"""Benchmark the 4-bit decode GEMV (kernels/designs/q4gemv.py) against an fp32 numpy dequantization.

    python benchmarks/bench_q4gemv.py <kernel_dir> 2048 1024x2048 2048x2048 6144x2048 15360x2048

Weights are synthetic Q4_K-style values: q in 0..15 with one (scale, min) per row per 32 weights. One context
per K (the largest N's xclbin); each N runs through its own instruction stream.
"""

import os
import sys
import time

import numpy as np

from xdna1_llm.bf16 import from_bf16, to_bf16
from xdna1_llm.npu.device import FROM_DEV, TO_DEV, Npu, hard_exit

LANES, RB, G, N_MAX = 8, 16, 32, 15360


def pack(q: np.ndarray, s: np.ndarray, m: np.ndarray) -> np.ndarray:
    """q (N, K) uint8 in 0..15, s/m (N, K/32) bf16 bits -> the kernel's tile layout, tiles of 16 rows in order."""
    N, K = q.shape
    qb = q.reshape(N // RB, RB, K // 2, 2)  # (tile, row, k pair, lo/hi)
    nib = (qb[..., 0] | (qb[..., 1] << 4)).astype(np.uint8).transpose(0, 2, 1)  # (tile, k pair, row)
    sm = np.stack([s.reshape(N // RB, RB, K // G), m.reshape(N // RB, RB, K // G)], axis=2)  # (tile,row,2,g)
    sm = sm.transpose(0, 3, 2, 1)  # (tile, g, s/m, row)
    return np.concatenate([nib.reshape(N // RB, -1), sm.reshape(N // RB, -1).view(np.uint8)], axis=1).reshape(-1)


def main():
    kdir, K, shapes = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    tb = K // 2 * RB + K // G * 2 * RB * 2
    npu = Npu()
    rng = np.random.default_rng(0)
    path = lambda tag, ext: os.path.join(kdir, tag + ext)  # noqa: E731
    k = npu.load(f"q4_k{K}", path(f"q4_{shapes[-1]}", ".xclbin"), path(f"q4_{shapes[-1]}", ".bin"))
    bo_x = npu.bo(tb, k.group(1))
    bo_y = npu.bo(N_MAX * 4, k.group(2))
    y_all = np.frombuffer(bo_y.map(), dtype=np.float32).reshape(LANES, N_MAX // LANES)
    x = to_bf16(rng.standard_normal(K).astype(np.float32))
    for shape in shapes:
        N = int(shape.split("x")[0])
        k.add_program(shape, path(f"q4_{shape}", ".bin"))
        q = rng.integers(0, 16, size=(N, K), dtype=np.uint8)
        s = to_bf16(rng.uniform(0.001, 0.004, size=(N, K // G)).astype(np.float32))
        m = to_bf16(rng.uniform(0.0, 0.03, size=(N, K // G)).astype(np.float32))
        w = pack(q, s, m)
        tile = np.zeros(tb, np.uint8)
        tile[: 2 * K] = x.view(np.uint8)
        tile[2 * K : 2 * K + 4] = np.array([N // LANES // RB], np.int32).view(np.uint8)
        bo_w = npu.bo(w.nbytes, k.group(0))
        bo_w.write(w.tobytes(), 0)
        bo_w.sync(TO_DEV)
        bo_x.write(tile.tobytes(), 0)
        bo_x.sync(TO_DEV)
        ts = []
        for _ in range(52):
            t0 = time.perf_counter()
            k.run(bo_w, bo_x, bo_y, program=shape)
            ts.append(time.perf_counter() - t0)
        bo_y.sync(FROM_DEV)
        y = y_all[:, : N // LANES].reshape(-1)
        wf = np.repeat(from_bf16(s), G, axis=1) * q - np.repeat(from_bf16(m), G, axis=1)
        ref = wf.astype(np.float64) @ from_bf16(x).astype(np.float64)
        err = float(np.max(np.abs(y - ref)) / np.max(np.abs(ref)))
        med = float(np.median(ts[2:]))
        print(
            f"q4 {shape:12s} {w.nbytes / 2**20:6.1f} MB  {med * 1e3:7.3f} ms  {w.nbytes / med / 1e9:6.2f} GB/s  "
            f"{N * K / med / 1e9:5.1f} Gweights/s  max rel err {err:.1e}"
        )
    hard_exit(0)


if __name__ == "__main__":
    main()
