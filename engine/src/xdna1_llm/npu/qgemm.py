"""Prompt linear-layer engine: GGUF Q4_K / Q6_K weights on the W4A16 GEMM (kernels/designs/qgemm.py).

C[256, N] (fp32) = A[256, K] (bf16) . W[N, K]^T, reading W from the decode GEMV's device buffers (NpuQLinear):
the weights are on the device once, in one layout, for both prompt and decode. Registering a weight only loads
the instruction streams of its matrices (`qm4_<N>x<K>.bin`, `qm6only_<N>x<K>.bin`). The cores take the format
and sizes at run time and do not depend on the GEMV geometry, so one context runs every build.

Host work per call is kept small: the activations are converted to bf16 on a few threads, and a single-matrix
result is returned as a view of its device buffer, not copied. Calls take turns between two sets of output
buffers, so a result stays valid until the call after next (LFM3 uses each one before its next call).

The FFN runs fused (ffn): gate + up, stored interleaved (NpuQLinear glu=True), goes to its `_glu` build, whose
cores write silu(gate) * up in bf16 into a device buffer that the down projection then reads as its A. The
15 MB gate + up result, the CPU's silu * up and the bf16 conversion of act are gone.
"""

from __future__ import annotations

import itertools
import os
import time
from concurrent.futures import ThreadPoolExecutor

import ml_dtypes
import numpy as np

from .device import FROM_DEV, TO_DEV, finish, mapped
from .qlinear import NpuQLinear, deinterleave

M_TILE = 256
THREADS = min(4, os.cpu_count() or 1)  # the bf16 conversion is compute-bound: ~2.7x faster on 4 threads
MIN_ROWS = 32  # per thread


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
        self.c: list[list[tuple]] = [[], []]  # two sets, used in turn: per matrix of a weight, (bo, bytes)
        self.turn = 0
        self.mats: dict[str, list[tuple[str, int]]] = {}  # name -> [(program, N)], in the GEMV's order
        self.glu: dict[str, tuple[str, int]] = {}  # interleaved gate + up -> (its _glu program, N)
        self.act = None  # (bo, bytes): the fused FFN's silu(gate) * up, only ever on the device
        self.npu_ms, self.calls = 0.0, 0
        self.pool = ThreadPoolExecutor(THREADS, thread_name_prefix="qgemm") if THREADS > 1 else None

    def _to_bf16(self, a: np.ndarray, x: np.ndarray):
        """x (float32) -> bf16 bits in a (round to nearest even), split by rows over the pool."""
        dst = a.view(ml_dtypes.bfloat16)
        n = min(THREADS, len(x) // MIN_ROWS) if self.pool else 1
        if n <= 1:
            np.copyto(dst, x, casting="unsafe")
            return
        bounds = np.linspace(0, len(x), n + 1).astype(int)
        jobs = [self.pool.submit(np.copyto, dst[i:j], x[i:j], casting="unsafe") for i, j in itertools.pairwise(bounds)]
        for f in jobs:
            f.result()

    def _add(self, prog: str) -> bool:
        if prog not in self.kern.programs:
            insts = os.path.join(self.q.kdir, prog + ".bin")
            if not os.path.exists(insts):
                return False
            self.kern.add_program(prog, insts)
        return True

    def register(self, name: str, w=None, glu: bool = False):
        """w and glu are ignored: the weights, and their layout, are the ones qlin registered under name."""
        mats = []
        for m in self.q.mats[name]:
            prog = program(self.geo, m["fmt"], m["N"], m["K"])
            if not self._add(prog):
                insts = os.path.join(self.q.kdir, prog + ".bin")
                raise FileNotFoundError(f"{name}: no instruction stream {insts} (kernels/build_kernels.sh)")
            if m["glu"] and self._add(prog + "_glu"):
                self.glu[name] = (prog + "_glu", m["N"])
                size = M_TILE * m["N"] * 4  # the program writes the whole C size (act at the start)
                if self.act is None or self.act[1] < size:
                    self.act = (self.npu.bo(size, self.kern.group(2)), size)
            if m["K"] not in self.a:
                bo = self.npu.bo(M_TILE * m["K"] * 2, self.kern.group(0))
                self.a[m["K"]] = (bo, mapped(bo, np.uint16, (M_TILE, m["K"])))
            mats.append((prog, m["N"]))
        for c in self.c:
            for j, (_, N) in enumerate(mats):  # sized for the widest matrix in each position
                if j == len(c) or c[j][1] < M_TILE * N * 4:
                    c[j : j + 1] = [(self.npu.bo(M_TILE * N * 4, self.kern.group(2)), M_TILE * N * 4)]
        self.mats[name] = mats

    def _load_a(self, name: str, x: np.ndarray):
        T, K = x.shape
        if T > M_TILE:
            raise ValueError(f"{name}: {T} rows, at most {M_TILE} per call")
        bo_a, a = self.a[K]
        self._to_bf16(a[:T], x)  # rows past T are stale: each output row reads only its own row of A
        bo_a.sync(TO_DEV, T * K * 2, 0)
        return bo_a

    def __call__(self, name: str, x: np.ndarray) -> np.ndarray:
        """x (T <= 256, K) float32 -> (T, N) float32, valid until the call after next. A weight's matrices
        share A and run queued together."""
        T = len(x)
        t0 = time.perf_counter()
        bo_a = self._load_a(name, x)
        mats, c = self.mats[name], self.c[self.turn]
        self.turn ^= 1
        runs = [
            self.kern.start(bo_a, m["bo_w"], c[j][0], program=prog)
            for j, ((prog, _), m) in enumerate(zip(mats, self.q.mats[name], strict=True))
        ]
        outs = []
        for j, ((prog, N), r, m) in enumerate(zip(mats, runs, self.q.mats[name], strict=True)):
            finish(r, prog)
            c[j][0].sync(FROM_DEV, T * N * 4, 0)
            y = mapped(c[j][0], np.float32)[: T * N].reshape(T, N)
            outs.append(deinterleave(y) if m["glu"] else y)
        y = np.concatenate(outs, axis=1) if len(outs) > 1 else outs[0]
        self.npu_ms += (time.perf_counter() - t0) * 1e3
        self.calls += len(mats)
        return y

    def ffn(self, gateup: str, down: str, x: np.ndarray) -> np.ndarray | None:
        """down(silu(gate(x)) * up(x)) for x (T <= 256, K) float32 -> (T, N_down) float32, valid until the call
        after next; None if these weights do not run fused (no _glu build, or not stored interleaved). The two
        programs run queued: the down projection reads the act that the first one leaves on the device."""
        if gateup not in self.glu or len(self.mats.get(down, ())) != 1:
            return None
        (gprog, N), ((dprog, D),) = self.glu[gateup], self.mats[down]
        if self.q.mats[down][0]["K"] != N // 2:
            return None
        T = len(x)
        t0 = time.perf_counter()
        bo_a = self._load_a(gateup, x)
        bo_act, c = self.act[0], self.c[self.turn][0][0]
        self.turn ^= 1
        r1 = self.kern.start(bo_a, self.q.mats[gateup][0]["bo_w"], bo_act, program=gprog)
        r2 = self.kern.start(bo_act, self.q.mats[down][0]["bo_w"], c, program=dprog)
        finish(r1, gprog)
        finish(r2, dprog)
        c.sync(FROM_DEV, T * D * 4, 0)
        self.npu_ms += (time.perf_counter() - t0) * 1e3
        self.calls += 2
        return mapped(c, np.float32)[: T * D].reshape(T, D)
