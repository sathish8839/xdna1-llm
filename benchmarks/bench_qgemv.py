"""Benchmark the quantized decode GEMVs (kernels/designs/qgemv.py) on GGUF Q4_K / Q6_K weights.

    python benchmarks/bench_qgemv.py <kernel_dir> q4_2048x2048 q4_15360x2048 q6_2048x7680
    python benchmarks/bench_qgemv.py <kernel_dir> --gguf model.gguf blk.0.ffn_gate.weight blk.0.ffn_down.weight

Shapes (q4_NxK / q6_NxK) use random blocks; --gguf uses real tensors (Q4_K or Q6_K). Results are checked
against the fp64 product with the GGUF dequantization. One context per (format, K), from the largest N's
xclbin; each N runs through its own instruction stream.
"""

import os
import sys
import time

import numpy as np

from xdna1_llm import q4k, q6k
from xdna1_llm.bf16 import from_bf16, to_bf16
from xdna1_llm.npu.device import FROM_DEV, TO_DEV, Npu, hard_exit

LANES, RB = 8, 16
FORMATS = {"q4": q4k, "q6": q6k}


def random_blocks(fmt: str, n: int, k: int, rng) -> np.ndarray:
    mod = FORMATS[fmt]
    raw = rng.integers(0, 256, size=(n, k // mod.QK, mod.BLOCK_BYTES), dtype=np.uint8)
    scale = lambda lo, hi: rng.uniform(lo, hi, size=(n, k // mod.QK)).astype(np.float16)[..., None].view(np.uint8)  # noqa: E731
    if fmt == "q4":
        raw[..., 0:2], raw[..., 2:4] = scale(1e-4, 1e-3), scale(0, 1e-3)
    else:
        raw[..., 208:210] = scale(1e-5, 1e-4)
    return raw.reshape(n, -1)


def main():
    args = sys.argv[1:]
    kdir = args.pop(0)
    rng = np.random.default_rng(0)
    jobs = []  # (label, format, N, K, raw)
    if args[0] == "--gguf":
        from xdna1_llm.gguf_weights import GGUFWeights

        w = GGUFWeights(args[1])
        for name in args[2:]:
            kind, raw = w.quant(name)
            fmt = {"Q4_K": "q4", "Q6_K": "q6"}[kind]
            mod = FORMATS[fmt]
            jobs.append((name, fmt, raw.shape[0], raw.shape[1] // mod.BLOCK_BYTES * mod.QK, raw))
    else:
        for tag in args:
            fmt, shape = tag.split("_")
            n, k = map(int, shape.split("x"))
            jobs.append((tag, fmt, n, k, random_blocks(fmt, n, k, rng)))

    npu = Npu()
    path = lambda tag, ext: os.path.join(kdir, tag + ext)  # noqa: E731
    built = [f[:-7] for f in os.listdir(kdir) if f.endswith(".xclbin")]
    kernels = {}
    for label, fmt, N, K, raw in jobs:
        mod, tag = FORMATS[fmt], f"{fmt}_{N}x{K}"
        if (fmt, K) not in kernels:
            big = max(
                (b for b in built if b.startswith(fmt) and b.endswith(f"x{K}")), key=lambda b: int(b[3:].split("x")[0])
            )
            kernels[fmt, K] = npu.load(f"{fmt}_k{K}", path(big, ".xclbin"), path(big, ".bin"))
        k = kernels[fmt, K]
        if tag not in k.programs:
            k.add_program(tag, path(tag, ".bin"))
        _, TB, XT = mod.layout(K)
        x = to_bf16(rng.standard_normal(K).astype(np.float32))
        tiles = mod.repack(raw)
        xin = np.zeros(XT * TB, np.uint8)
        xin[: 2 * K] = x.view(np.uint8)
        xin[-4:] = np.array([N // LANES // RB], np.int32).view(np.uint8)
        bo_w, bo_x, bo_y = npu.bo(tiles.nbytes, k.group(0)), npu.bo(xin.nbytes, k.group(1)), npu.bo(N * 4, k.group(2))
        bo_w.write(tiles.tobytes(), 0)
        bo_w.sync(TO_DEV)
        bo_x.write(xin.tobytes(), 0)
        bo_x.sync(TO_DEV)
        ts = []
        for _ in range(52):
            t0 = time.perf_counter()
            k.run(bo_w, bo_x, bo_y, program=tag)
            ts.append(time.perf_counter() - t0)
        bo_y.sync(FROM_DEV)
        y = np.frombuffer(bo_y.map(), dtype=np.float32, count=N).copy()
        xf = from_bf16(x).astype(np.float64)
        ref = np.concatenate([mod.dequantize(raw[i : i + 4096]).astype(np.float64) @ xf for i in range(0, N, 4096)])
        err = float(np.max(np.abs(y - ref)) / np.max(np.abs(ref)))
        med = float(np.median(ts[2:]))
        print(
            f"{label:32s} {tag:15s} {tiles.nbytes / 2**20:6.1f} MB  {med * 1e3:7.3f} ms  "
            f"{N * K / med / 1e9:5.1f} Gweights/s  max rel err {err:.1e}"
        )
    hard_exit(0)


if __name__ == "__main__":
    main()
