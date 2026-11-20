"""Benchmark the decode GEMV (kernels/designs/gemv.py) on the NPU against numpy.

    python benchmarks/bench_gemv.py <kernel_dir> 2048 1024x2048 2048x2048 6144x2048 15360x2048

Loads one context per K (from the xclbin of the largest N) and runs every N through its own instruction
stream. Reports the median time over 50 runs, the weight bandwidth and the max relative error.
"""

import os
import sys
import time

import numpy as np

from xdna1_llm.bf16 import from_bf16, to_bf16
from xdna1_llm.npu.device import FROM_DEV, TO_DEV, Npu, hard_exit

LANES, TILE_BYTES, N_MAX = 8, 16 * 1024, 15360


def layout(K):
    R = max(1, TILE_BYTES // (2 * K))
    return R, (K if R > 1 else K + 32)


def main():
    kdir, K, shapes = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    R, KS = layout(K)
    npu = Npu()
    rng = np.random.default_rng(0)
    path = lambda tag, ext: os.path.join(kdir, tag + ext)  # noqa: E731
    k = npu.load(f"gemv_k{K}", path(f"gemv_{shapes[-1]}", ".xclbin"), path(f"gemv_{shapes[-1]}", ".bin"))
    bo_x = npu.bo(R * KS * 2, k.group(1))
    bo_y = npu.bo(N_MAX * 4, k.group(2))
    y_all = np.frombuffer(bo_y.map(), dtype=np.float32).reshape(LANES, N_MAX // LANES)
    x = to_bf16(rng.standard_normal(K).astype(np.float32))
    for shape in shapes:
        N = int(shape.split("x")[0])
        k.add_program(shape, path(f"gemv_{shape}", ".bin"))
        w = to_bf16(rng.standard_normal((N, K)).astype(np.float32) * 0.02)
        w_pad = np.zeros((N, KS), np.uint16)
        w_pad[:, :K] = w
        tile = np.zeros(R * KS, np.uint16)
        tile[:K] = x
        tile[K : K + 2] = np.array([N // LANES // R], np.int32).view(np.uint16)
        bo_w = npu.bo(w_pad.nbytes, k.group(0))
        bo_w.write(w_pad.tobytes(), 0)
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
        ref = from_bf16(w) @ from_bf16(x)
        err = float(np.max(np.abs(y - ref)) / np.max(np.abs(ref)))
        med = float(np.median(ts[2:]))
        print(
            f"gemv {shape:12s} {w.nbytes / 2**20:6.1f} MB  {med * 1e3:7.3f} ms  {w.nbytes / med / 1e9:6.2f} GB/s  "
            f"max rel err {err:.1e}"
        )
    hard_exit(0)


if __name__ == "__main__":
    main()
