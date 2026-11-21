"""Prompt linear-layer engine: GGUF Q4_K / Q6_K weights on the W4A16 GEMM (kernels/designs/qgemm.py).

C[256, N] (fp32) = A[256, K] (bf16) . W[N, K]^T, reading W from the decode GEMV's device buffers (NpuQLinear):
the weights are on the device once, in one layout, for both prompt and decode. Registering a weight only loads
the instruction streams of its matrices (`qm4_<N>x<K>.bin`, `qm6only_<N>x<K>.bin`). The cores take the format
and sizes at run time and do not depend on the GEMV geometry, so one context runs every build.
"""

from __future__ import annotations

import os
import time

import ml_dtypes
import numpy as np

from .device import FROM_DEV, TO_DEV, finish, mapped
from .qlinear import NpuQLinear

M_TILE = 256


def program(geo, fmt: int, N: int, K: int) -> str:
    """Name of the GEMM build for one matrix: the GEMV's (geo.program) with qm in front of the format."""
    return "qm" + geo.program(fmt, N, K)[1:]


def builds(kdir: str, geo) -> list[str]:
    prefix = "qm6only_" if geo.name == "q6only" else ("qm4_", "qm6_")
    return sorted(f[:-7] for f in os.listdir(kdir) if f.startswith(prefix) and f.endswith(".xclbin"))


class NpuQGemm:
    wants_head = False

    def __init__(self, qlin: NpuQLinear):
        """qlin: the decode backend; its weights (registered first) are the ones this GEMM reads."""
        self.q, self.npu, self.geo = qlin, qlin.npu, qlin.geo
        built = builds(qlin.kdir, self.geo)
        if not built:
            raise FileNotFoundError(f"no W4A16 GEMM in {qlin.kdir} (kernels/build_kernels.sh <out> qgemm)")
        path = os.path.join(qlin.kdir, built[0])  # every build has the same cores
        self.kern = self.npu.load("qgemm", path + ".xclbin", path + ".bin")
        self.a: dict[int, tuple] = {}  # K -> (bo, mapped (256, K) bf16 bits)
        self.c: list[tuple] = []  # one output buffer per matrix of a weight, (bo, bytes)
        self.mats: dict[str, list[tuple[str, int]]] = {}  # name -> [(program, N)], in the GEMV's order
        self.npu_ms, self.calls = 0.0, 0

    def register(self, name: str, w=None):
        """w is ignored: the weights are the ones qlin registered under name."""
        mats = []
        for m in self.q.mats[name]:
            prog = program(self.geo, m["fmt"], m["N"], m["K"])
            if prog not in self.kern.programs:
                insts = os.path.join(self.q.kdir, prog + ".bin")
                if not os.path.exists(insts):
                    raise FileNotFoundError(f"{name}: no instruction stream {insts} (kernels/build_kernels.sh)")
                self.kern.add_program(prog, insts)
            if m["K"] not in self.a:
                bo = self.npu.bo(M_TILE * m["K"] * 2, self.kern.group(0))
                self.a[m["K"]] = (bo, mapped(bo, np.uint16, (M_TILE, m["K"])))
            mats.append((prog, m["N"]))
        for j, (_, N) in enumerate(mats):  # sized for the widest matrix in each position
            if j == len(self.c) or self.c[j][1] < M_TILE * N * 4:
                bo = (self.npu.bo(M_TILE * N * 4, self.kern.group(2)), M_TILE * N * 4)
                self.c[j : j + 1] = [bo]
        self.mats[name] = mats

    def __call__(self, name: str, x: np.ndarray) -> np.ndarray:
        """x (T <= 256, K) float32 -> (T, N) float32. A weight's matrices share A and run queued together."""
        T, K = x.shape
        if T > M_TILE:
            raise ValueError(f"{name}: {T} rows, at most {M_TILE} per call")
        t0 = time.perf_counter()
        bo_a, a = self.a[K]
        np.copyto(a[:T].view(ml_dtypes.bfloat16), x, casting="unsafe")  # rows past T are stale: unused rows
        bo_a.sync(TO_DEV, T * K * 2, 0)
        mats = self.mats[name]
        runs = [
            self.kern.start(bo_a, m["bo_w"], self.c[j][0], program=prog)
            for j, ((prog, _), m) in enumerate(zip(mats, self.q.mats[name], strict=True))
        ]
        outs = []
        for j, ((prog, N), r) in enumerate(zip(mats, runs, strict=True)):
            finish(r, prog)
            self.c[j][0].sync(FROM_DEV, T * N * 4, 0)
            outs.append(mapped(self.c[j][0], np.float32)[: T * N].reshape(T, N))
        y = np.concatenate(outs, axis=1) if len(outs) > 1 else outs[0].copy()
        self.npu_ms += (time.perf_counter() - t0) * 1e3
        self.calls += len(mats)
        return y
