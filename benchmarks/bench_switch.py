"""Is the round-robin penalty from switching hw_contexts, or from switching weight buffers?

Same kernel (pf_kv_fused, 256x2048x1024 bf16), three patterns:
  same      : one weight BO, repeated calls
  rotate_bo : 8 different weight BOs, same context (how a chunked forward pass would run)
  two_ctx   : alternate between two contexts of the same xclbin
"""

import os
import time

import numpy as np
import pyxrt as xrt

KDIR = os.environ.get(
    "XDNA1_LLM_KERNELS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels", "build")
)
TAG, M, K, N = "pf_kv_fused", 256, 2048, 1024
TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE

dev = xrt.device(0)
xb = xrt.xclbin(os.path.join(KDIR, TAG + ".xclbin"))
dev.register_xclbin(xb)
insts = np.fromfile(os.path.join(KDIR, TAG + ".bin"), dtype=np.uint32)


def setup(ctx):
    k = xrt.kernel(ctx, xb.get_kernels()[0].get_name())
    bi = xrt.bo(dev, insts.nbytes, xrt.bo.cacheable, k.group_id(1))
    bi.write(insts.tobytes(), 0)
    bi.sync(TO)
    ba = xrt.bo(dev, M * K * 2, xrt.bo.host_only, k.group_id(3))
    bc = xrt.bo(dev, M * N * 2, xrt.bo.host_only, k.group_id(5))
    return k, bi, ba, bc


ctx1 = xrt.hw_context(dev, xb.get_uuid())
ctx2 = xrt.hw_context(dev, xb.get_uuid())
k1, bi1, ba1, bc1 = setup(ctx1)
k2, bi2, ba2, bc2 = setup(ctx2)
rng = np.random.default_rng(0)
wbos = []
for _ in range(8):
    b = xrt.bo(dev, K * N * 2, xrt.bo.host_only, k1.group_id(4))
    b.write(rng.integers(0, 2**15, K * N, dtype=np.uint16).tobytes(), 0)
    b.sync(TO)
    wbos.append(b)


def bench(label, step, n=64):
    step(0)
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        step(i)
        ts.append((time.perf_counter() - t0) * 1e3)
    print(f"  {label:10s} median {np.median(ts):.3f} ms   min {np.min(ts):.3f} ms")


bench("same", lambda i: k1(3, bi1, len(insts), ba1, wbos[0], bc1).wait())
bench("rotate_bo", lambda i: k1(3, bi1, len(insts), ba1, wbos[i % 8], bc1).wait())
bench(
    "two_ctx",
    lambda i: (k1 if i % 2 == 0 else k2)(
        3, bi1 if i % 2 == 0 else bi2, len(insts), ba1 if i % 2 == 0 else ba2, wbos[0], bc1 if i % 2 == 0 else bc2
    ).wait(),
)
os._exit(0)
