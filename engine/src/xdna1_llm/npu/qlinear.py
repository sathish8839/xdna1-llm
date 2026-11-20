"""Decode linear-layer engine: GGUF Q4_K / Q6_K weights on the quantized GEMV (kernels/designs/qgemv.py).

Each weight is repacked once at registration (xdna1_llm.q4k / q6k, lossless) into its own device buffer. The
GEMV's core program takes the format and sizes at run time, so one hardware context serves every matrix; each
(format, N, K) only needs its own instruction stream (`q4_<N>x<K>.bin`), and switching those costs nothing.

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

FORMATS = {"Q4_K": (4, q4k), "Q6_K": (6, q6k)}


class NpuQLinear:
    wants_head = True

    def __init__(self, npu: Npu | None = None, kernel_dir: str | None = None):
        from .linear import default_kernel_dir

        self.npu = npu or Npu()
        self.kdir = kernel_dir or default_kernel_dir()
        built = sorted(f[:-7] for f in os.listdir(self.kdir) if f.startswith(("q4_", "q6_")) and f.endswith(".xclbin"))
        if not built:
            raise FileNotFoundError(f"no quantized GEMV in {self.kdir} (kernels/build_kernels.sh <out> qgemv)")
        path = os.path.join(self.kdir, built[0])  # every build has the same cores; any one makes the context
        self.kern = self.npu.load("qgemv", path + ".xclbin", path + ".bin")
        self.bo_x = self.npu.bo(2 * q4k.TILE_BYTES, self.kern.group(1))
        self.x = mapped(self.bo_x, np.uint8)
        self.mats: dict[str, list[dict]] = {}
        self.npu_ms, self.calls, self.repack_s = 0.0, 0, 0.0

    def register(self, name: str, w):
        groups: list[tuple[str, list]] = []  # consecutive parts with the same type and K run as one GEMV
        for kind, data in as_parts(w):
            if kind not in FORMATS:
                raise ValueError(f"{name}: the decode GEMV takes Q4_K / Q6_K weights, got {kind}")
            if groups and groups[-1][0] == kind and groups[-1][1][0].shape[1] == data.shape[1]:
                groups[-1][1].append(data)
            else:
                groups.append((kind, [data]))
        mats = []
        for kind, datas in groups:
            fmt, mod = FORMATS[kind]
            raw = np.concatenate(datas, axis=0) if len(datas) > 1 else datas[0]
            N, K = raw.shape[0], raw.shape[1] // mod.BLOCK_BYTES * mod.QK
            if N % (q4k.LANES * q4k.RB):
                raise ValueError(f"{name}: {N} rows, need a multiple of {q4k.LANES * q4k.RB}")
            prog = f"q{fmt}_{N}x{K}"
            if prog not in self.kern.programs:
                insts = os.path.join(self.kdir, prog + ".bin")
                if not os.path.exists(insts):
                    raise FileNotFoundError(f"{name}: no instruction stream {insts} (kernels/build_kernels.sh)")
                self.kern.add_program(prog, insts)
            t0 = time.perf_counter()
            tiles = mod.repack(raw)
            self.repack_s += time.perf_counter() - t0
            bo_w = self.npu.bo(tiles.nbytes, self.kern.group(0))
            bo_w.write(tiles.tobytes(), 0)
            bo_w.sync(TO_DEV)
            bo_y = self.npu.bo(N * 4, self.kern.group(2))
            mats.append(dict(fmt=fmt, prog=prog, N=N, K=K, bo_w=bo_w, bo_y=bo_y, y=mapped(bo_y, np.float32)))
        self.mats[name] = mats

    def __call__(self, name: str, x: np.ndarray) -> np.ndarray:
        """x (T, K) float32 -> (T, N) float32, one GEMV per row and weight group."""
        x = np.atleast_2d(x)
        t0 = time.perf_counter()
        out = np.empty((x.shape[0], sum(m["N"] for m in self.mats[name])), np.float32)
        for t in range(x.shape[0]):
            col = 0
            xbits = to_bf16(x[t])
            for m in self.mats[name]:
                self.x[:] = q4k.x_input(xbits, m["fmt"], m["N"], m["K"])
                self.bo_x.sync(TO_DEV)
                self.kern.run(m["bo_w"], self.bo_x, m["bo_y"], program=m["prog"])
                m["bo_y"].sync(FROM_DEV)
                out[t, col : col + m["N"]] = m["y"]
                col += m["N"]
                self.calls += 1
        self.npu_ms += (time.perf_counter() - t0) * 1e3
        return out
