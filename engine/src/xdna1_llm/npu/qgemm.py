"""Prompt linear-layer engine: GGUF Q4_K / Q6_K weights on the W4A16 GEMM (kernels/designs/qgemm.py).

C[256, N] (fp32) = A[256, K] (bf16) . W[N, K]^T, reading W from the decode GEMV's device buffers (NpuQLinear):
the weights are on the device once, in one layout, for both prompt and decode. Registering a weight only loads
the instruction streams of its matrices (`qm4_<N>x<K>.bin`, `qm6only_<N>x<K>.bin`). The cores take the format
and sizes at run time and do not depend on the GEMV geometry, so one context runs every build.

Host work per call is kept small: the activations are converted to bf16 on a few threads, and a single-matrix
result is returned as a view of its device buffer, not copied.

A call can overlap the host's other work: `submit` starts it and returns, `collect` waits for its result. Calls
take turns between three sets of buffers (A, C, act), so two can be in flight while the caller still reads the
result of the one before: a result stays valid until three more calls have been submitted.

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
SETS = 3  # buffer sets, used in turn


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
        # per set: "a": K -> (bo, mapped (256, K) bf16 bits); "c": per matrix of a weight, (bo, bytes);
        # "act": (bo, bytes), the fused FFN's silu(gate) * up, only ever on the device; "y": float32, the result of
        # a weight with several matrices, their outputs side by side (reused: a fresh 3 MB array page-faults)
        self.sets = [{"a": {}, "c": [], "act": None, "y": np.empty(0, np.float32)} for _ in range(SETS)]
        self.turn = 0
        self.mats: dict[str, list[tuple[str, int]]] = {}  # name -> [(program, N)], in the GEMV's order
        self.glu: dict[str, tuple[str, int]] = {}  # interleaved gate + up -> (its _glu program, N)
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
            glu = m["glu"] and self._add(prog + "_glu")
            if glu:
                self.glu[name] = (prog + "_glu", m["N"])
            for s in self.sets:
                size = M_TILE * m["N"] * 4  # the _glu program writes the whole C size (act at the start)
                if glu and (s["act"] is None or s["act"][1] < size):
                    s["act"] = (self.npu.bo(size, self.kern.group(2)), size)
                if m["K"] not in s["a"]:
                    bo = self.npu.bo(M_TILE * m["K"] * 2, self.kern.group(0))
                    s["a"][m["K"]] = (bo, mapped(bo, np.uint16, (M_TILE, m["K"])))
            mats.append((prog, m["N"]))
        for c in (s["c"] for s in self.sets):
            for j, (_, N) in enumerate(mats):  # sized for the widest matrix in each position
                if j == len(c) or c[j][1] < M_TILE * N * 4:
                    c[j : j + 1] = [(self.npu.bo(M_TILE * N * 4, self.kern.group(2)), M_TILE * N * 4)]
        size = M_TILE * sum(N for _, N in mats)
        for s in self.sets:
            if len(mats) > 1 and s["y"].size < size:
                s["y"] = np.empty(size, np.float32)
        self.mats[name] = mats

    def _load_a(self, s: dict, name: str, x: np.ndarray):
        T, K = x.shape
        if T > M_TILE:
            raise ValueError(f"{name}: {T} rows, at most {M_TILE} per call")
        bo_a, a = s["a"][K]
        self._to_bf16(a[:T], x)  # rows past T are stale: each output row reads only its own row of A
        bo_a.sync(TO_DEV, T * K * 2, 0)
        return bo_a

    def _set(self) -> dict:
        s = self.sets[self.turn]
        self.turn = (self.turn + 1) % SETS
        return s

    def submit(self, name: str, x: np.ndarray):
        """Start x (T <= 256, K) float32 times weight name^T; collect() returns the (T, N) float32 result. A
        weight's matrices share A and run queued together."""
        t0 = time.perf_counter()
        s = self._set()
        bo_a = self._load_a(s, name, x)
        runs, reads = [], []
        for j, ((prog, N), m) in enumerate(zip(self.mats[name], self.q.mats[name], strict=True)):
            bo_c = s["c"][j][0]
            runs.append((prog, self.kern.start(bo_a, m["bo_w"], bo_c, program=prog)))
            reads.append((N, bo_c, m["glu"]))
        y = s["y"][: len(x) * sum(N for N, _, _ in reads)].reshape(len(x), -1) if len(reads) > 1 else None
        self.npu_ms += (time.perf_counter() - t0) * 1e3
        self.calls += len(runs)
        return len(x), runs, reads, y

    def submit_ffn(self, gateup: str, down: str, x: np.ndarray):
        """Start down(silu(gate(x)) * up(x)) for x (T <= 256, K) float32; collect() returns the (T, N_down)
        float32 result. None if these weights do not run fused (no _glu build, or not stored interleaved). The two
        programs run queued: the down projection reads the act that the first one leaves on the device."""
        if gateup not in self.glu or len(self.mats.get(down, ())) != 1:
            return None
        (gprog, N), ((dprog, D),) = self.glu[gateup], self.mats[down]
        if self.q.mats[down][0]["K"] != N // 2:
            return None
        t0 = time.perf_counter()
        s = self._set()
        bo_a = self._load_a(s, gateup, x)
        bo_act, bo_c = s["act"][0], s["c"][0][0]
        r1 = self.kern.start(bo_a, self.q.mats[gateup][0]["bo_w"], bo_act, program=gprog)
        r2 = self.kern.start(bo_act, self.q.mats[down][0]["bo_w"], bo_c, program=dprog)
        self.npu_ms += (time.perf_counter() - t0) * 1e3
        self.calls += 2
        return len(x), [(gprog, r1), (dprog, r2)], [(D, bo_c, False)], None

    def collect(self, pending) -> np.ndarray:
        """Wait for a submitted call; returns its result, a view of the call's buffers."""
        t0 = time.perf_counter()
        T, runs, reads, y = pending
        for prog, r in runs:
            finish(r, prog)
        outs = []
        for N, bo_c, glu in reads:
            bo_c.sync(FROM_DEV, T * N * 4, 0)
            c = mapped(bo_c, np.float32)[: T * N].reshape(T, N)
            outs.append(deinterleave(c) if glu else c)
        y = outs[0] if y is None else np.concatenate(outs, axis=1, out=y)
        self.npu_ms += (time.perf_counter() - t0) * 1e3
        return y

    def __call__(self, name: str, x: np.ndarray) -> np.ndarray:
        """x (T <= 256, K) float32 -> (T, N) float32: submit and collect."""
        return self.collect(self.submit(name, x))

    def ffn(self, gateup: str, down: str, x: np.ndarray) -> np.ndarray | None:
        """The fused FFN, submitted and collected; None if it does not run fused (see submit_ffn)."""
        p = self.submit_ffn(gateup, down, x)
        return None if p is None else self.collect(p)
