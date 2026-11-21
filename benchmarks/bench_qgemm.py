"""Benchmark the W4A16 prompt GEMM (kernels/designs/qgemm.py) on random GGUF Q4_K / Q6_K weights.

    python benchmarks/bench_qgemm.py <kernel_dir> qm4_2048x2048 qm4_15360x2048 qm6_2048x7680 qm4_15360x2048_glu

Each shape (qm<4|6>_NxK) runs on 256 random rows and is checked against numpy: A (bf16) times the weights
dequantized and rounded to bf16, the product the bf16 GEMM computes. A _glu shape reads its rows as gate and
up interleaved in 16-row blocks and is checked against bf16(silu(A . gate^T) * (A . up^T)). One context runs
every shape.
"""

import os
import sys
import time

import numpy as np
from bench_qgemv import random_blocks

from xdna1_llm import q4k, q6k
from xdna1_llm.bf16 import from_bf16, to_bf16
from xdna1_llm.npu.device import FROM_DEV, TO_DEV, Npu, hard_exit, mapped


def silu(x):
    return x / (1.0 + np.exp(-x))


M = 256


def main():
    kdir, tags = sys.argv[1], sys.argv[2:]
    geo, rng = q4k.UNIFIED, np.random.default_rng(0)
    npu = Npu()
    path = lambda tag, ext: os.path.join(kdir, tag + ext)  # noqa: E731
    kern = npu.load("qgemm", path(tags[0], ".xclbin"), path(tags[0], ".bin"))
    for tag in tags:
        fmt, (N, K), glu = int(tag[2]), map(int, tag.split("_")[1].split("x")), tag.endswith("_glu")
        mod = {4: q4k, 6: q6k}[fmt]
        if tag not in kern.programs:
            kern.add_program(tag, path(tag, ".bin"))
        raw = random_blocks(f"q{fmt}", N, K, rng)
        tiles = mod.repack(raw, geo)
        a = to_bf16(rng.standard_normal((M, K)).astype(np.float32))
        bo_a, bo_c = npu.bo(a.nbytes, kern.group(0)), npu.bo(M * N * 4, kern.group(2))
        bo_w = npu.bo(tiles.nbytes + geo.slack, kern.group(1))
        bo_w.write(tiles.tobytes(), 0)
        bo_w.sync(TO_DEV)
        bo_a.write(a.tobytes(), 0)
        bo_a.sync(TO_DEV)
        ts = []
        for _ in range(12):
            t0 = time.perf_counter()
            kern.run(bo_a, bo_w, bo_c, program=tag)
            ts.append(time.perf_counter() - t0)
        bo_c.sync(FROM_DEV)
        wq = from_bf16(to_bf16(mod.dequantize(raw))).astype(np.float64)
        if glu:
            c = from_bf16(mapped(bo_c, np.uint16)[: M * N // 2].reshape(M, N // 2)).astype(np.float64)
            blocks = wq.reshape(N // 32, 2, 16, K)
            gate, up = (blocks[:, i].reshape(N // 2, K) for i in (0, 1))
            af = from_bf16(a).astype(np.float64)
            ref = from_bf16(to_bf16((silu(af @ gate.T) * (af @ up.T)).astype(np.float32))).astype(np.float64)
        else:
            c = mapped(bo_c, np.float32, (M, N)).copy()
            ref = from_bf16(a).astype(np.float64) @ wq.T
        err = float(np.max(np.abs(c - ref)) / np.max(np.abs(ref)))
        med = float(np.median(ts[2:]))
        print(f"{tag:16s} {med * 1e3:7.3f} ms  {2 * M * N * K / med / 1e12:5.2f} TFLOP/s  max rel err {err:.1e}")
    hard_exit(0)


if __name__ == "__main__":
    main()
