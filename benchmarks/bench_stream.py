"""Measure DDR -> NPU streaming bandwidth with the stream_bw probe kernels (kernels/designs/stream_bw.py).

    python benchmarks/bench_stream.py <kernel_dir> bw_l1 bw_l8 ...

Each kernel takes (A: the bytes to stream, C: 16 words per lane). Reports the median GB/s over 20 runs.
"""

import os
import sys
import time

import numpy as np

from xdna1_llm.npu.device import FROM_DEV, TO_DEV, Npu, hard_exit

OUT_WORDS = 16


def main():
    kdir, tags = sys.argv[1], sys.argv[2:]
    npu = Npu()
    for tag in tags:
        k = npu.load(tag, os.path.join(kdir, tag + ".xclbin"), os.path.join(kdir, tag + ".bin"))
        lanes = int(tag.split("_l")[1].split("_")[0])
        total = int(tag.split("_mb")[1]) * 1024 * 1024 if "_mb" in tag else 16 * 1024 * 1024
        bo_a = npu.bo(total, k.group(0))
        bo_c = npu.bo(lanes * OUT_WORDS * 4, k.group(1))
        a = np.arange(total // 4, dtype=np.int32)
        bo_a.write(a.tobytes(), 0)
        bo_a.sync(TO_DEV)
        ts = []
        for _ in range(22):
            t0 = time.perf_counter()
            k.run(bo_a, bo_c)
            ts.append(time.perf_counter() - t0)
        bo_c.sync(FROM_DEV)
        c = np.frombuffer(bytes(bo_c.read(lanes * OUT_WORDS * 4, 0)), dtype=np.int32)
        med = float(np.median(ts[2:]))
        print(
            f"{tag:24s} {total / 2**20:6.1f} MB  {med * 1e3:8.3f} ms  {total / med / 1e9:6.2f} GB/s  "
            f"last-tile words {c[::OUT_WORDS].tolist()}"
        )
    hard_exit(0)


if __name__ == "__main__":
    main()
