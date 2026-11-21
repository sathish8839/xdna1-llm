"""Decode linear-layer engine: GGUF Q4_K / Q6_K weights on the quantized GEMV (kernels/designs/qgemv.py).

Each weight is repacked once at registration (xdna1_llm.q4k / q6k, lossless) into its own device buffer. The
GEMV's core program takes the format and sizes at run time, so one hardware context serves every matrix; each
(format, N, K) only needs its own instruction stream (`q4_<N>x<K>.bin`), and switching those costs nothing.

Two tile geometries (xdna1_llm.q4k.Geometry): `unified` runs Q4_K and Q6_K in one program (Q4_K_M models);
`q6only` streams no padding for all-Q6_K models (`q6only_<N>x<K>` builds) and is used when the model has no
other quantized type and those builds exist.

Weights whose parts share a format and K (for example ffn_gate + ffn_up) run as one GEMV with stacked rows.
An FFN's gate + up (registered with glu=True) is stored with its rows interleaved in blocks of 16, gate then up,
the layout the prompt GEMM's fused FFN needs (xdna1_llm.npu.qgemm, NpuQGemm.ffn); results come back in the
registered order.
"""

from __future__ import annotations

import os
import time
from functools import partial

import numpy as np

from .. import q4k, q6k
from ..bf16 import to_bf16
from ..gguf_weights import as_parts
from .device import FROM_DEV, TO_DEV, Npu, finish, mapped

FORMATS = {"Q4_K": (4, q4k), "Q6_K": (6, q6k)}
GLU_ROWS = 16  # rows per gate / up block of an interleaved FFN weight: half of a GEMM core's 32 outputs


def interleave(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """(n, B) gate and up weight rows -> (2n, B): blocks of GLU_ROWS rows, gate then up."""
    n, B = gate.shape
    return np.stack([gate.reshape(-1, GLU_ROWS, B), up.reshape(-1, GLU_ROWS, B)], axis=1).reshape(2 * n, B)


def deinterleave(y: np.ndarray) -> np.ndarray:
    """(..., 2n) outputs of an interleaved weight -> gate (n) | up (n)."""
    *lead, N = y.shape
    return y.reshape(*lead, -1, 2, GLU_ROWS).swapaxes(-3, -2).reshape(*lead, N)


def _builds(kdir: str, geo: q4k.Geometry) -> list[str]:
    prefixes = ("q4_", "q6_") if geo is q4k.UNIFIED else (geo.name + "_",)
    return sorted(f[:-7] for f in os.listdir(kdir) if f.startswith(prefixes) and f.endswith(".xclbin"))


class NpuQLinear:
    wants_head = True

    def __init__(
        self, npu: Npu | None = None, kernel_dir: str | None = None, formats: set[str] | None = None, cache=None
    ):
        """formats: the tensor types of the model (GGUFWeights.types()); an all-Q6_K model uses the q6only
        geometry when it is built. cache: a WeightCache for the repacked weights, or None."""
        from .linear import default_kernel_dir

        self.npu, self.cache = npu or Npu(), cache
        self.kdir = kernel_dir or default_kernel_dir()
        quant = set(formats or ()) & set(FORMATS)
        q6only = quant == {"Q6_K"} and _builds(self.kdir, q4k.Q6ONLY)
        self.geo = q4k.Q6ONLY if q6only else q4k.UNIFIED
        built = _builds(self.kdir, self.geo)
        if not built:
            raise FileNotFoundError(f"no quantized GEMV in {self.kdir} (kernels/build_kernels.sh <out> qgemv)")
        path = os.path.join(self.kdir, built[0])  # every build of a geometry has the same cores
        self.kern = self.npu.load("qgemv_" + self.geo.name, path + ".xclbin", path + ".bin")
        self.mats: dict[str, list[dict]] = {}
        self.npu_ms, self.calls, self.repack_s = 0.0, 0, 0.0

    def register(self, name: str, w, glu: bool = False):
        """glu: w is [gate, up] of an FFN; when the two share a format and shape, they are stored interleaved."""
        groups: list[tuple[str, list]] = []  # consecutive parts with the same type and K run as one GEMV
        for kind, data in as_parts(w):
            if kind not in FORMATS:
                raise ValueError(f"{name}: the decode GEMV takes Q4_K / Q6_K weights, got {kind}")
            if groups and groups[-1][0] == kind and groups[-1][1][0].shape[1] == data.shape[1]:
                groups[-1][1].append(data)
            else:
                groups.append((kind, [data]))
        mats = []
        for g, (kind, datas) in enumerate(groups):
            fmt, mod = FORMATS[kind]
            il = glu and len(groups) == 1 and len(datas) == 2 and datas[0].shape == datas[1].shape
            if il:
                raw = interleave(*datas)
            else:
                raw = np.concatenate(datas, axis=0) if len(datas) > 1 else datas[0]
            N, K = raw.shape[0], raw.shape[1] // mod.BLOCK_BYTES * mod.QK
            if N % (q4k.LANES * q4k.RB):
                raise ValueError(f"{name}: {N} rows, need a multiple of {q4k.LANES * q4k.RB}")
            prog = self.geo.program(fmt, N, K)
            if prog not in self.kern.programs:
                insts = os.path.join(self.kdir, prog + ".bin")
                if not os.path.exists(insts):
                    raise FileNotFoundError(f"{name}: no instruction stream {insts} (kernels/build_kernels.sh)")
                self.kern.add_program(prog, insts)
            t0 = time.perf_counter()
            make = partial(mod.repack, raw, self.geo)
            key = f"gemv-{self.geo.name}/{name}.{g}" + (".glu" if il else "")
            tiles = self.cache.get(key, make) if self.cache else make()
            self.repack_s += time.perf_counter() - t0
            bo_w = self.npu.bo(tiles.nbytes + self.geo.slack, self.kern.group(0))  # tiles are read past their end
            bo_w.write(tiles.tobytes(), 0)
            bo_w.sync(TO_DEV)
            bo_y = self.npu.bo(N * 4, self.kern.group(2))
            bo_x = self.npu.bo(self.geo.x_tiles * self.geo.tile, self.kern.group(1))  # own x: groups can queue
            x, y = mapped(bo_x, np.uint8), mapped(bo_y, np.float32)
            mats.append(dict(fmt=fmt, prog=prog, N=N, K=K, glu=il, bo_w=bo_w, bo_x=bo_x, x=x, bo_y=bo_y, y=y))
        self.mats[name] = mats

    def __call__(self, name: str, x: np.ndarray) -> np.ndarray:
        """x (T, K) float32 -> (T, N) float32. A weight with several groups (for example Q4_K q+k and a Q6_K v)
        queues all its GEMVs before waiting, so their dispatch latencies overlap."""
        x = np.atleast_2d(x)
        t0 = time.perf_counter()
        mats = self.mats[name]
        out = np.empty((x.shape[0], sum(m["N"] for m in mats)), np.float32)
        for t in range(x.shape[0]):
            xbits = to_bf16(x[t])
            runs = []
            for m in mats:
                m["x"][:] = q4k.x_input(xbits, m["fmt"], m["N"], m["K"], self.geo)
                m["bo_x"].sync(TO_DEV)
                runs.append(self.kern.start(m["bo_w"], m["bo_x"], m["bo_y"], program=m["prog"]))
            col = 0
            for m, r in zip(mats, runs, strict=True):
                finish(r, m["prog"])
                m["bo_y"].sync(FROM_DEV)
                out[t, col : col + m["N"]] = deinterleave(m["y"]) if m["glu"] else m["y"]
                col += m["N"]
            self.calls += len(mats)
        self.npu_ms += (time.perf_counter() - t0) * 1e3
        return out
