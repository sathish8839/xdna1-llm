"""On-device tests: need an XDNA1 NPU, pyxrt, and kernels from `kernels/build_kernels.sh <out> engine` / `smoke` /
`qgemv`."""

import os

import numpy as np
import pytest

pytest.importorskip("pyxrt", reason="pyxrt not importable (copy pyxrt.pyd from the XRT Windows SDK into the venv)")

from xdna1_llm.bf16 import from_bf16, to_bf16  # noqa: E402
from xdna1_llm.npu.device import FROM_DEV, TO_DEV  # noqa: E402
from xdna1_llm.npu.linear import NpuLinear, default_kernel_dir  # noqa: E402

pytestmark = pytest.mark.npu


def test_vector_scalar_mul(npu):
    kdir = default_kernel_dir()
    xclbin = os.path.join(kdir, "smoke_vector_scalar_mul.xclbin")
    if not os.path.exists(xclbin):
        pytest.skip("smoke kernel not built (kernels/build_kernels.sh <out> smoke)")
    n, factor = 4096, 3
    k = npu.load("smoke", xclbin, os.path.join(kdir, "smoke_vector_scalar_mul.bin"))
    bo_in, bo_out, bo_f = npu.bo(n * 2, k.group(0)), npu.bo(n * 2, k.group(1)), npu.bo(4, k.group(2))
    x = np.arange(1, n + 1, dtype=np.int16)
    bo_in.write(x.tobytes(), 0)
    bo_f.write(np.array([factor], np.int32).tobytes(), 0)
    for b in (bo_in, bo_f, bo_out):
        b.sync(TO_DEV)
    k.run(bo_in, bo_out, bo_f)
    bo_out.sync(FROM_DEV)
    y = np.frombuffer(bytes(bo_out.read(n * 2, 0)), dtype=np.int16)
    np.testing.assert_array_equal(y, (x.astype(np.int32) * factor).astype(np.int16))


def test_npu_linear_matches_cpu(npu):
    rng = np.random.default_rng(0)
    lin = NpuLinear(npu, ks=(2048,))
    w = to_bf16(rng.standard_normal((2048, 2048)).astype(np.float32) * 0.02)
    lin.register("w", w)
    x = rng.standard_normal((37, 2048)).astype(np.float32)
    ref = from_bf16(to_bf16(x)) @ from_bf16(w).T
    y = lin("w", x)
    assert y.shape == (37, 2048)
    np.testing.assert_allclose(y, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("kind", ["Q4_K", "Q6_K"])
def test_npu_qlinear_matches_gguf_dequantization(npu, kind):
    from xdna1_llm import q4k, q6k
    from xdna1_llm.npu.qlinear import NpuQLinear

    fmt, mod = {"Q4_K": ("q4", q4k), "Q6_K": ("q6", q6k)}[kind]
    if not os.path.exists(os.path.join(default_kernel_dir(), f"{fmt}_512x2048.bin")):
        pytest.skip(f"{fmt} GEMV not built (kernels/build_kernels.sh <out> qgemv)")
    rng = np.random.default_rng(0)
    n, k = 512, 2048
    raw = rng.integers(0, 256, size=(n, k // 256, mod.BLOCK_BYTES), dtype=np.uint8)
    scales = rng.uniform(1e-4, 1e-3, size=(n, k // 256, 2)).astype(np.float16).view(np.uint8)
    if kind == "Q4_K":
        raw[..., 0:4] = scales
    else:
        raw[..., 208:210] = scales[..., :2]
    raw = raw.reshape(n, -1)
    lin = NpuQLinear(npu)
    lin.register("w", (kind, raw))
    x = rng.standard_normal((2, k)).astype(np.float32)
    ref = from_bf16(to_bf16(x)).astype(np.float64) @ mod.dequantize(raw).T.astype(np.float64)
    np.testing.assert_allclose(lin("w", x), ref, rtol=0, atol=1e-4 * np.abs(ref).max())
