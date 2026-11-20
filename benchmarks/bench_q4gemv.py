"""Benchmark the 4-bit decode GEMV (kernels/designs/q4gemv.py) on Q4_K weights.

    python benchmarks/bench_q4gemv.py <kernel_dir> 2048x2048 15360x2048 2048x7680
    python benchmarks/bench_q4gemv.py <kernel_dir> --gguf model.gguf blk.0.ffn_gate.weight ...

Shapes (NxK) use random Q4_K blocks; --gguf uses real tensors. Results are checked against the fp64 product
with the GGUF dequantization. One context per K (the largest N's xclbin); each N runs through its own
instruction stream.
"""

import os
import sys
import time

import numpy as np

from xdna1_llm import q4k
from xdna1_llm.bf16 import from_bf16, to_bf16
from xdna1_llm.npu.device import FROM_DEV, TO_DEV, Npu, hard_exit

LANES = 8


def random_q4k(n: int, k: int, rng) -> np.ndarray:
    raw = rng.integers(0, 256, size=(n, k // q4k.QK, q4k.BLOCK_BYTES), dtype=np.uint8)
    raw[..., 0:2] = rng.uniform(1e-4, 1e-3, size=(n, k // q4k.QK)).astype(np.float16)[..., None].view(np.uint8)
    raw[..., 2:4] = rng.uniform(0, 1e-3, size=(n, k // q4k.QK)).astype(np.float16)[..., None].view(np.uint8)
    return raw.reshape(n, -1)


def main():
    args = sys.argv[1:]
    kdir = args.pop(0)
    rng = np.random.default_rng(0)
    jobs = []  # (label, N, K, raw)
    if args[0] == "--gguf":
        from xdna1_llm.gguf_weights import GGUFWeights

        w = GGUFWeights(args[1])
        for name in args[2:]:
            kind, raw = w.quant(name)
            assert kind == "Q4_K", (name, kind)
            jobs.append((name, raw.shape[0], raw.shape[1] // q4k.BLOCK_BYTES * q4k.QK, raw))
    else:
        for shape in args:
            n, k = map(int, shape.split("x"))
            jobs.append((shape, n, k, random_q4k(n, k, rng)))

    npu = Npu()
    path = lambda tag, ext: os.path.join(kdir, tag + ext)  # noqa: E731
    shapes = [f[3:-7] for f in os.listdir(kdir) if f.startswith("q4_") and f.endswith(".xclbin")]
    kernels = {}
    for label, N, K, raw in jobs:
        shape = f"{N}x{K}"
        if K not in kernels:
            big = max((s for s in shapes if s.endswith(f"x{K}")), key=lambda s: int(s.split("x")[0]))
            kernels[K] = npu.load(f"q4_k{K}", path(f"q4_{big}", ".xclbin"), path(f"q4_{big}", ".bin"))
        k = kernels[K]
        if shape not in k.programs:
            k.add_program(shape, path(f"q4_{shape}", ".bin"))
        _, TB, XT = q4k.layout(K)
        x = to_bf16(rng.standard_normal(K).astype(np.float32))
        tiles = q4k.repack(raw)
        xin = np.zeros(XT * TB, np.uint8)
        xin[: 2 * K] = x.view(np.uint8)
        xin[-4:] = np.array([N // LANES // q4k.RB], np.int32).view(np.uint8)
        bo_w, bo_x, bo_y = npu.bo(tiles.nbytes, k.group(0)), npu.bo(xin.nbytes, k.group(1)), npu.bo(N * 4, k.group(2))
        bo_w.write(tiles.tobytes(), 0)
        bo_w.sync(TO_DEV)
        bo_x.write(xin.tobytes(), 0)
        bo_x.sync(TO_DEV)
        ts = []
        for _ in range(52):
            t0 = time.perf_counter()
            k.run(bo_w, bo_x, bo_y, program=shape)
            ts.append(time.perf_counter() - t0)
        bo_y.sync(FROM_DEV)
        y = np.frombuffer(bo_y.map(), dtype=np.float32, count=N).copy()
        xf = from_bf16(x).astype(np.float64)
        ref = np.concatenate([q4k.dequantize(raw[i : i + 4096]).astype(np.float64) @ xf for i in range(0, N, 4096)])
        err = float(np.max(np.abs(y - ref)) / np.max(np.abs(ref)))
        med = float(np.median(ts[2:]))
        print(
            f"q4 {label:32s} {shape:12s} {tiles.nbytes / 2**20:6.1f} MB  {med * 1e3:7.3f} ms  "
            f"{N * K / med / 1e9:5.1f} Gweights/s  max rel err {err:.1e}"
        )
    hard_exit(0)


if __name__ == "__main__":
    main()
