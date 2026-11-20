"""Benchmark LFM3-1.7B linear-layer shapes on the XDNA1 NPU (Windows, pyxrt) against the CPU.

Kernels come from kernels/build_kernels.sh <out> shapes (mlir-aie whole_array GEMM, npu1, 4x4 tiles):
  pf_*  prefill   M=256 tokens, bf16 in/out
  dc_*  decode    M=64 (stand-in for M=1: the time is dominated by streaming the K x N weight matrix)

Phases:
  1. Open one hw_context per kernel, all at once, to probe the driver's concurrent-context limit.
  2. Per-shape correctness + latency (each kernel alone, its context kept open).
  3. Round-robin across all open contexts (the access pattern of a real forward pass).
  4. CPU baseline: numpy fp32 matmul at the same shapes.
Writes results to bench_results.json.
"""

import json
import os
import sys
import time

import numpy as np
import pyxrt as xrt

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
KDIR = os.environ.get(
    "XDNA1_LLM_KERNELS", os.path.join(REPO, "kernels", "build")
)  # build with: build_kernels.sh <dir> shapes
SHAPES = {  # name: (K, N, count per LFM3 forward pass)
    "attn_qo": (2048, 2048, 11 * 2 + 13),  # q,o in 11 attn layers + out_proj in 13 conv layers
    "kv_fused": (2048, 1024, 11),
    "conv_in": (2048, 6144, 13),
    "ffn_gateup": (2048, 15360, 24),
    "ffn_down": (7680, 2048, 24),
    "lm_head": (2048, 12800, 10),  # 128000-vocab head split into 10 chunks (one 128000-wide kernel exceeds DMA limits)
}
ITERS = int(os.environ.get("ITERS", "20"))
SYNC_TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
SYNC_FROM = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE


def to_bf16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + (0x7FFF + ((u >> 16) & 1))) >> 16).astype(np.uint16)


def from_bf16(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


class NpuGemm:
    def __init__(self, dev, tag, M, K, N, dtype):
        self.tag, self.M, self.K, self.N, self.dtype = tag, M, K, N, dtype
        xb = xrt.xclbin(os.path.join(KDIR, tag + ".xclbin"))
        dev.register_xclbin(xb)
        self.ctx = xrt.hw_context(dev, xb.get_uuid())
        self.kernel = xrt.kernel(self.ctx, xb.get_kernels()[0].get_name())
        self.insts = np.fromfile(os.path.join(KDIR, tag + ".bin"), dtype=np.uint32)
        k = self.kernel
        isz, osz = (2, 2) if dtype == "bf16" else (1, 4)
        self.bo_i = xrt.bo(dev, self.insts.nbytes, xrt.bo.cacheable, k.group_id(1))
        self.bo_a = xrt.bo(dev, M * K * isz, xrt.bo.host_only, k.group_id(3))
        self.bo_b = xrt.bo(dev, K * N * isz, xrt.bo.host_only, k.group_id(4))
        self.bo_c = xrt.bo(dev, M * N * osz, xrt.bo.host_only, k.group_id(5))
        self.bo_i.write(self.insts.tobytes(), 0)
        self.bo_i.sync(SYNC_TO)

    def load(self, A, B):
        self.bo_a.write(A.tobytes(), 0)
        self.bo_b.write(B.tobytes(), 0)
        self.bo_a.sync(SYNC_TO)
        self.bo_b.sync(SYNC_TO)

    def run(self):
        st = self.kernel(3, self.bo_i, len(self.insts), self.bo_a, self.bo_b, self.bo_c).wait()
        if st != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"{self.tag}: {st}")

    def result(self):
        self.bo_c.sync(SYNC_FROM)
        if self.dtype == "bf16":
            raw = np.frombuffer(bytes(self.bo_c.read(self.M * self.N * 2, 0)), dtype=np.uint16)
            return from_bf16(raw).reshape(self.M, self.N)
        raw = np.frombuffer(bytes(self.bo_c.read(self.M * self.N * 4, 0)), dtype=np.int32)
        return raw.reshape(self.M, self.N)


def timeit(fn, iters):
    fn()  # warm-up
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), float(np.min(ts))


def main():
    dev = xrt.device(0)
    plan = []
    for name, (K, N, cnt) in SHAPES.items():
        plan.append((f"pf_{name}", 256, K, N, "bf16", name, cnt))
        plan.append((f"dc_{name}", 64, K, N, "bf16", name, cnt))
    plan.append(("dc_conv_in_i8", 64, 2048, 6144, "i8", "conv_in", 13))
    plan = [p for p in plan if os.path.exists(os.path.join(KDIR, p[0] + ".xclbin"))]
    only = [t for t in os.environ.get("TAGS", "").split(",") if t]
    if only:  # the Windows driver allows only 5 concurrent contexts, so run subsets per process
        plan = [p for p in plan if p[0] in only]

    # Phase 1: open every context at once
    print(f"== phase 1: opening {len(plan)} hw_contexts concurrently")
    gemms, failed = {}, {}
    for tag, M, K, N, dt, _name, _cnt in plan:
        try:
            gemms[tag] = NpuGemm(dev, tag, M, K, N, dt)
            print(f"  ctx #{len(gemms):2d} OK   {tag}")
        except Exception as e:  # noqa: BLE001
            failed[tag] = str(e)
            print(f"  ctx FAIL       {tag}: {e}")

    rng = np.random.default_rng(0)
    res = {"contexts_opened": len(gemms), "contexts_failed": failed, "shapes": {}}

    # Phase 2: per-shape correctness + latency
    print("== phase 2: per-shape correctness and latency")
    for tag, M, K, N, dt, name, cnt in plan:
        g = gemms.get(tag)
        if g is None:
            continue
        if dt == "bf16":
            A = to_bf16(rng.standard_normal((M, K), dtype=np.float32) * 0.1)
            B = to_bf16(rng.standard_normal((K, N), dtype=np.float32) * 0.1)
            ref = from_bf16(A) @ from_bf16(B)
        else:
            A = rng.integers(-8, 8, (M, K), dtype=np.int8)
            B = rng.integers(-8, 8, (K, N), dtype=np.int8)
            ref = A.astype(np.int32) @ B.astype(np.int32)
        g.load(A, B)
        try:
            med, mn = timeit(g.run, ITERS)
        except Exception as e:  # noqa: BLE001
            print(f"  {tag:18s} RUN FAIL: {e}")
            res["shapes"][tag] = {"error": str(e)}
            continue
        C = g.result()
        err = float(np.abs(C - ref).max() / (np.abs(ref).max() + 1e-9))
        wbytes = K * N * (2 if dt == "bf16" else 1)
        gflops = 2 * M * K * N / (med * 1e-3) / 1e9
        gbps = wbytes / (med * 1e-3) / 1e9
        ok = err < (2e-2 if dt == "bf16" else 1e-6)
        print(
            f"  {tag:18s} M={M:3d} K={K:5d} N={N:6d} {dt:4s} {med:8.3f} ms  {gflops:7.1f} GFLOP/s  "
            f"weights {gbps:5.1f} GB/s  rel_err={err:.2e} {'PASS' if ok else 'FAIL'}"
        )
        res["shapes"][tag] = dict(
            M=M,
            K=K,
            N=N,
            dtype=dt,
            name=name,
            count=cnt,
            median_ms=med,
            min_ms=mn,
            gflops=gflops,
            weight_gbps=gbps,
            rel_err=err,
            ok=bool(ok),
        )

    # Phase 3: round-robin across all contexts (forward-pass access pattern)
    print("== phase 3: round-robin across all open contexts")
    order = [t for t in res["shapes"] if "error" not in res["shapes"][t]]
    rr = {t: [] for t in order}
    for _ in range(5):
        for t in order:
            t0 = time.perf_counter()
            gemms[t].run()
            rr[t].append((time.perf_counter() - t0) * 1e3)
    for t in order:
        alone = res["shapes"][t]["median_ms"]
        mixed = float(np.median(rr[t][1:]))
        res["shapes"][t]["roundrobin_ms"] = mixed
        print(f"  {t:18s} alone {alone:8.3f} ms   round-robin {mixed:8.3f} ms   (+{mixed - alone:6.3f} ms)")

    # Phase 4: CPU baseline (numpy fp32)
    print("== phase 4: CPU numpy fp32 baseline")
    for name, (K, N, cnt) in SHAPES.items():
        if os.environ.get("SKIP_CPU"):
            break
        B = rng.standard_normal((K, N), dtype=np.float32)
        for M, key in ((256, "cpu_pf"), (1, "cpu_dc")):
            A = rng.standard_normal((M, K), dtype=np.float32)
            med, _ = timeit(lambda A=A, B=B: A @ B, 5 if name == "lm_head" else 10)
            res["shapes"].setdefault(key + "_" + name, {}).update(
                dict(M=M, K=K, N=N, name=name, count=cnt, median_ms=med, gflops=2 * M * K * N / (med * 1e-3) / 1e9)
            )
            print(f"  {key}_{name:12s} M={M:3d} {med:8.3f} ms  {2 * M * K * N / (med * 1e-3) / 1e9:7.1f} GFLOP/s")

    out = os.environ.get("OUT", "bench_results.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # skip pyxrt teardown, which access-violates on interpreter exit
