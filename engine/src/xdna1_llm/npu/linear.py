"""Chunked linear-layer engine: every projection runs as 1024-wide output chunks on one of two fixed-tile GEMMs.

    k2048: C[256x1024] f32 = A[256x2048] bf16 @ B[2048x1024] bf16
    k7680: C[256x1024] f32 = A[256x7680] bf16 @ B[7680x1024] bf16

Each chunk's weights get their own device buffer, uploaded once at registration. One context per K, so a forward
pass only switches context when it moves between K=2048 and K=7680 projections.
Activations are padded to 256 rows (prefill of up to 256 tokens per call).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from ..bf16 import to_bf16
from ..gguf_weights import as_parts
from .device import COMPLETED, FROM_DEV, TO_DEV, Npu, mapped

M_TILE, N_CHUNK = 256, 1024
SUPPORTED_K = (2048, 7680)


def default_kernel_dir() -> str:
    """$XDNA1_LLM_KERNELS, else kernels/build in a source checkout, else %LOCALAPPDATA%/xdna1-llm/kernels."""
    if os.environ.get("XDNA1_LLM_KERNELS"):
        return os.environ["XDNA1_LLM_KERNELS"]
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "kernels" / "build"
        if candidate.is_dir():
            return str(candidate)
    return str(Path(os.environ.get("LOCALAPPDATA", Path.home())) / "xdna1-llm" / "kernels")


class NpuLinear:
    wants_head = False  # the LM head stays on the CPU in fp32 for prefill

    def __init__(self, npu: Npu | None = None, kernel_dir: str | None = None, ks=SUPPORTED_K):
        self.npu = npu or Npu()
        kdir = kernel_dir or default_kernel_dir()
        self.k = {}
        for K in ks:
            kern = self.npu.load(
                f"gemm_k{K}",
                os.path.join(kdir, f"gemm_256x{K}x1024_bf16_f32.xclbin"),
                os.path.join(kdir, f"gemm_256x{K}x1024_bf16_f32.bin"),
            )
            bo_a = self.npu.bo(M_TILE * K * 2, kern.group(0))
            bo_c = [self.npu.bo(M_TILE * N_CHUNK * 4, kern.group(2)) for _ in range(16)]
            self.k[K] = dict(
                kern=kern,
                bo_a=bo_a,
                bo_c=bo_c,
                rows=M_TILE,
                a=mapped(bo_a, np.uint16, (M_TILE, K)),
                c=[mapped(b, np.float32, (M_TILE, N_CHUNK)) for b in bo_c],
            )
        self.chunks: dict[str, tuple[int, list]] = {}
        self.npu_ms = 0.0
        self.calls = 0

    def register(self, name: str, w):
        """w: bf16 bits, shape (out, in) as stored in GGUF (or BF16 tensors whose rows are stacked); out must be a
        multiple of 1024."""
        parts = as_parts(w)
        if any(kind != "BF16" for kind, _ in parts):
            raise ValueError(f"{name}: the prefill GEMM engine takes BF16 weights, got {[k for k, _ in parts]}")
        wbits = np.concatenate([data for _, data in parts], axis=0)
        out, K = wbits.shape
        if out % N_CHUNK or K not in self.k:
            raise ValueError(f"{name}: shape {wbits.shape} needs out % {N_CHUNK} == 0 and in in {tuple(self.k)}")
        kern = self.k[K]["kern"]
        bos = []
        for j in range(out // N_CHUNK):
            B = np.ascontiguousarray(wbits[j * N_CHUNK : (j + 1) * N_CHUNK, :].T)  # (K, 1024) row-major
            bo = self.npu.bo(B.nbytes, kern.group(1))
            bo.write(B.tobytes(), 0)
            bo.sync(TO_DEV)
            bos.append(bo)
        self.chunks[name] = (K, bos)

    def __call__(self, name: str, x: np.ndarray) -> np.ndarray:
        K, bos = self.chunks[name]
        T = x.shape[0]
        if T > M_TILE or x.shape[1] != K:
            raise ValueError(f"{name}: input {x.shape}, expected (<= {M_TILE}, {K})")
        st = self.k[K]
        t0 = time.perf_counter()
        st["a"][:T] = to_bf16(x)
        if st["rows"] > T:
            st["a"][T:] = 0
        st["rows"] = T
        st["bo_a"].sync(TO_DEV)
        runs = [st["kern"].start(st["bo_a"], wbo, st["bo_c"][j]) for j, wbo in enumerate(bos)]  # queue all, then wait
        for j, r in enumerate(runs):
            s = r.wait()
            if s != COMPLETED:
                raise RuntimeError(f"{name} chunk {j}: {s}")
        outs = []
        for j in range(len(bos)):
            st["bo_c"][j].sync(FROM_DEV)
            outs.append(st["c"][j][:T])
        y = np.concatenate(outs, axis=1)
        self.npu_ms += (time.perf_counter() - t0) * 1e3
        self.calls += len(bos)
        return y
