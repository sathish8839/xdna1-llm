"""Decode linear-layer engine: GGUF Q4_K / Q6_K weights on the quantized GEMVs (kernels/designs/qgemv.py).

Each weight is repacked once at registration (xdna1_llm.q4k / q6k, lossless) into its own device buffer. One
context per (format, K), loaded from the largest N's xclbin; every N runs through its own instruction stream
(`q4_<N>x<K>.bin`), which costs nothing to switch. Moving between contexts costs ~1.45 ms per call on the
Windows driver, so the order of calls matters (see docs/benchmarks.md).

Weights whose parts share a format and K (for example ffn_gate + ffn_up) run as one GEMV with stacked rows.
"""

from __future__ import annotations

import os
import time

import numpy as np

from .. import q4k, q6k
from ..bf16 import to_bf16
from ..gguf_weights import as_parts
from .device import FROM_DEV, TO_DEV, Npu, mapped

LANES, RB = 8, 16
FORMATS = {"Q4_K": ("q4", q4k), "Q6_K": ("q6", q6k)}


class NpuQLinear:
    wants_head = True

    def __init__(self, npu: Npu | None = None, kernel_dir: str | None = None):
        from .linear import default_kernel_dir

        self.npu = npu or Npu()
        self.kdir = kernel_dir or default_kernel_dir()
        self.built = [f[:-7] for f in os.listdir(self.kdir) if f.endswith(".xclbin")]
        self.ctx: dict[tuple[str, int], dict] = {}
        self.mats: dict[str, list[dict]] = {}
        self.npu_ms, self.calls, self.repack_s = 0.0, 0, 0.0
        self.switches, self._last = 0, None  # calls that ran on a different context than the call before

    def _context(self, fmt: str, K: int) -> dict:
        if (fmt, K) not in self.ctx:
            sizes = [b for b in self.built if b.startswith(fmt + "_") and b.endswith(f"x{K}")]
            if not sizes:
                raise FileNotFoundError(f"no {fmt} GEMV for K={K} in {self.kdir} (build_kernels.sh <out> qgemv)")
            big = max(sizes, key=lambda b: int(b[3:].split("x")[0]))
            path = os.path.join(self.kdir, big)
            kern = self.npu.load(f"{fmt}_k{K}", path + ".xclbin", path + ".bin")
            _, TB, XT = FORMATS["Q4_K" if fmt == "q4" else "Q6_K"][1].layout(K)
            bo_x = self.npu.bo(XT * TB, kern.group(1))
            self.ctx[fmt, K] = dict(kern=kern, bo_x=bo_x, x=mapped(bo_x, np.uint8), xt_bytes=XT * TB)
        return self.ctx[fmt, K]

    def register(self, name: str, w):
        groups: list[tuple[str, list]] = []  # consecutive parts with the same type and K run as one GEMV
        for kind, data in as_parts(w):
            if kind not in FORMATS:
                raise ValueError(f"{name}: the decode GEMVs take Q4_K / Q6_K weights, got {kind}")
            if groups and groups[-1][0] == kind and groups[-1][1][0].shape[1] == data.shape[1]:
                groups[-1][1].append(data)
            else:
                groups.append((kind, [data]))
        mats = []
        for kind, datas in groups:
            fmt, mod = FORMATS[kind]
            raw = np.concatenate(datas, axis=0) if len(datas) > 1 else datas[0]
            N, K = raw.shape[0], raw.shape[1] // mod.BLOCK_BYTES * mod.QK
            if N % (LANES * RB):
                raise ValueError(f"{name}: {N} rows, need a multiple of {LANES * RB}")
            c = self._context(fmt, K)
            prog = f"{fmt}_{N}x{K}"
            if prog not in c["kern"].programs:
                insts = os.path.join(self.kdir, prog + ".bin")
                if not os.path.exists(insts):
                    raise FileNotFoundError(f"{name}: no instruction stream {insts} (kernels/build_kernels.sh)")
                c["kern"].add_program(prog, insts)
            t0 = time.perf_counter()
            tiles = mod.repack(raw)
            self.repack_s += time.perf_counter() - t0
            bo_w = self.npu.bo(tiles.nbytes, c["kern"].group(0))
            bo_w.write(tiles.tobytes(), 0)
            bo_w.sync(TO_DEV)
            bo_y = self.npu.bo(N * 4, c["kern"].group(2))
            mats.append(dict(ctx=c, prog=prog, N=N, K=K, bo_w=bo_w, bo_y=bo_y, y=mapped(bo_y, np.float32)))
        self.mats[name] = mats

    def __call__(self, name: str, x: np.ndarray) -> np.ndarray:
        """x (T, K) float32 -> (T, N) float32, one GEMV per row and weight group."""
        x = np.atleast_2d(x)
        t0 = time.perf_counter()
        out = np.empty((x.shape[0], sum(m["N"] for m in self.mats[name])), np.float32)
        for t in range(x.shape[0]):
            col = 0
            for m in self.mats[name]:
                c, K, N = m["ctx"], m["K"], m["N"]
                c["x"][: 2 * K] = to_bf16(x[t]).view(np.uint8)
                c["x"][-4:] = np.array([N // LANES // RB], np.int32).view(np.uint8)
                c["bo_x"].sync(TO_DEV)
                c["kern"].run(m["bo_w"], c["bo_x"], m["bo_y"], program=m["prog"])
                m["bo_y"].sync(FROM_DEV)
                out[t, col : col + N] = m["y"]
                col += N
                self.calls += 1
                self.switches += c is not self._last
                self._last = c
        self.npu_ms += (time.perf_counter() - t0) * 1e3
        return out
