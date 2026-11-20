"""Benchmark the decode GEMV kernels (kernels/designs/gemv.py) on the NPU against numpy.

    python benchmarks/bench_gemv.py <kernel_dir> gemv_2048x2048 gemv_2048x7680 ...

Reports the median time over 50 runs, the weight bandwidth (N*K*2 bytes per run) and the max relative error.
"""

import os
import sys
import time

import numpy as np

from xdna1_llm.bf16 import from_bf16, to_bf16
from xdna1_llm.npu.device import FROM_DEV, TO_DEV, Npu, hard_exit

TILE_BYTES = 16 * 1024


def main():
    kdir, tags = sys.argv[1], sys.argv[2:]
    npu = Npu()
    rng = np.random.default_rng(0)
    for tag in tags:
        N, K = map(int, tag.split("_")[1].split("x"))
        R = max(1, TILE_BYTES // (2 * K))
        k = npu.load(tag, os.path.join(kdir, tag + ".xclbin"), os.path.join(kdir, tag + ".bin"))
        w = to_bf16(rng.standard_normal((N, K)).astype(np.float32) * 0.02)
        x = to_bf16(rng.standard_normal(K).astype(np.float32))
        bo_w, bo_x, bo_y = npu.bo(w.nbytes, k.group(0)), npu.bo(R * K * 2, k.group(1)), npu.bo(N * 4, k.group(2))
        bo_w.write(w.tobytes(), 0)
        bo_x.write(np.concatenate([x, np.zeros(R * K - K, np.uint16)]).tobytes(), 0)
        bo_w.sync(TO_DEV)
        bo_x.sync(TO_DEV)
        ts = []
        for _ in range(52):
            t0 = time.perf_counter()
            k.run(bo_w, bo_x, bo_y)
            ts.append(time.perf_counter() - t0)
        bo_y.sync(FROM_DEV)
        y = np.frombuffer(bytes(bo_y.read(N * 4, 0)), dtype=np.float32)
        ref = from_bf16(w) @ from_bf16(x)
        err = float(np.max(np.abs(y - ref)) / np.max(np.abs(ref)))
        med = float(np.median(ts[2:]))
        print(
            f"{tag:18s} {w.nbytes / 2**20:6.1f} MB  {med * 1e3:7.3f} ms  {w.nbytes / med / 1e9:6.2f} GB/s  "
            f"max rel err {err:.1e}"
        )
    hard_exit(0)


if __name__ == "__main__":
    main()
