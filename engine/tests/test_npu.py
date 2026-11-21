"""On-device tests: need an XDNA1 NPU, pyxrt, and kernels from `kernels/build_kernels.sh <out> engine` / `smoke` /
`qgemv` / `qgemm`."""

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


@pytest.mark.parametrize(
    ("kind", "geometry", "n"), [("Q4_K", "unified", 512), ("Q6_K", "unified", 512), ("Q6_K", "q6only", 2048)]
)
def test_npu_qlinear_matches_gguf_dequantization(npu, kind, geometry, n):
    from xdna1_llm import q4k, q6k
    from xdna1_llm.npu.qlinear import NpuQLinear

    mod = {"Q4_K": q4k, "Q6_K": q6k}[kind]
    geo = {g.name: g for g in q4k.GEOMETRIES}[geometry]
    k = 2048
    prog = geo.program(4 if kind == "Q4_K" else 6, n, k)
    if not os.path.exists(os.path.join(default_kernel_dir(), prog + ".bin")):
        pytest.skip(f"{prog} not built (kernels/build_kernels.sh <out> qgemv)")
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, size=(n, k // 256, mod.BLOCK_BYTES), dtype=np.uint8)
    scales = rng.uniform(1e-4, 1e-3, size=(n, k // 256, 2)).astype(np.float16).view(np.uint8)
    if kind == "Q4_K":
        raw[..., 0:4] = scales
    else:
        raw[..., 208:210] = scales[..., :2]
    raw = raw.reshape(n, -1)
    lin = NpuQLinear(npu, formats={kind} if geometry == "q6only" else None)
    assert lin.geo is geo
    lin.register("w", (kind, raw))
    x = rng.standard_normal((2, k)).astype(np.float32)
    ref = from_bf16(to_bf16(x)).astype(np.float64) @ mod.dequantize(raw).T.astype(np.float64)
    np.testing.assert_allclose(lin("w", x), ref, rtol=0, atol=1e-4 * np.abs(ref).max())


def random_quant(kind, n, k, rng):
    from xdna1_llm import q4k, q6k

    mod = {"Q4_K": q4k, "Q6_K": q6k}[kind]
    raw = rng.integers(0, 256, size=(n, k // 256, mod.BLOCK_BYTES), dtype=np.uint8)
    scales = rng.uniform(1e-4, 1e-3, size=(n, k // 256, 2)).astype(np.float16).view(np.uint8)
    if kind == "Q4_K":
        raw[..., 0:4] = scales
    else:
        raw[..., 208:210] = scales[..., :2]
    return raw.reshape(n, -1), mod


@pytest.mark.parametrize(("kinds", "geometry"), [(("Q4_K", "Q6_K"), "unified"), (("Q6_K",), "q6only")])
def test_npu_qgemm_matches_bf16_weights(npu, kinds, geometry):
    """The prompt GEMM on the decode GEMV's weights: A (bf16) times the weights dequantized and rounded to bf16.
    Two parts of different formats (q + v of an attention layer) run as two matrices of one weight."""
    from xdna1_llm.npu.qgemm import NpuQGemm, program
    from xdna1_llm.npu.qlinear import NpuQLinear

    lin = NpuQLinear(npu, formats=set(kinds))
    n, k = (512, 2048) if geometry == "unified" else (2048, 2048)
    progs = [program(lin.geo, 4 if kd == "Q4_K" else 6, n, k) for kd in kinds]
    missing = [p for p in progs if not os.path.exists(os.path.join(lin.kdir, p + ".bin"))]
    if missing:
        pytest.skip(f"{missing} not built (kernels/build_kernels.sh <out> qgemm)")
    rng = np.random.default_rng(1)
    parts = [random_quant(kd, n, k, rng) for kd in kinds]
    lin.register("w", [(kd, raw) for kd, (raw, _) in zip(kinds, parts, strict=True)])
    gemm = NpuQGemm(lin)
    gemm.register("w")
    wq = np.concatenate([from_bf16(to_bf16(mod.dequantize(raw))) for raw, mod in parts]).astype(np.float64)
    for T in (256, 37):  # a full tile, then fewer rows over the stale ones
        x = rng.standard_normal((T, k)).astype(np.float32)
        ref = from_bf16(to_bf16(x)).astype(np.float64) @ wq.T
        y = gemm("w", x)
        assert y.shape == (T, n * len(kinds))
        np.testing.assert_allclose(y, ref, rtol=0, atol=1e-3 * np.abs(ref).max())
    kept = y.copy()  # a result stays valid through the next call (two output buffer sets, used in turn)
    gemm("w", rng.standard_normal((T, k)).astype(np.float32))
    np.testing.assert_array_equal(y, kept)


@pytest.mark.parametrize(("kinds", "geometry"), [(("Q4_K", "Q6_K"), "unified"), (("Q6_K", "Q6_K"), "q6only")])
def test_npu_qgemm_fused_ffn(npu, kinds, geometry):
    """The fused FFN (gate + up with silu * up on the cores, then down on the act left on the device) against
    numpy, and the interleaved gate + up through the unfused paths: the prompt GEMM and the decode GEMV."""
    from xdna1_llm.npu.qgemm import NpuQGemm, program
    from xdna1_llm.npu.qlinear import NpuQLinear

    lin = NpuQLinear(npu, formats=set(kinds))
    d, ff = 2048, 7680
    gu, dn = kinds
    fused = program(lin.geo, 4 if gu == "Q4_K" else 6, 2 * ff, d) + "_glu"
    if not os.path.exists(os.path.join(lin.kdir, fused + ".bin")):
        pytest.skip(f"{fused} not built (kernels/build_kernels.sh <out> qgemm)")
    rng = np.random.default_rng(2)
    (g, mod), (u, _), (w2, mod2) = (random_quant(kd, n, k, rng) for kd, n, k in [(gu, ff, d), (gu, ff, d), (dn, d, ff)])
    lin.register("gateup", [(gu, g), (gu, u)], glu=True)
    lin.register("down", (dn, w2))
    gemm = NpuQGemm(lin)
    gemm.register("gateup")
    gemm.register("down")
    deq = lambda raw, m: from_bf16(to_bf16(m.dequantize(raw))).astype(np.float64)  # noqa: E731
    wg, wu, wd = deq(g, mod), deq(u, mod), deq(w2, mod2)
    for T in (256, 37):
        x = 4 * rng.standard_normal((T, d)).astype(np.float32)
        xb = from_bf16(to_bf16(x)).astype(np.float64)
        G, U = xb @ wg.T, xb @ wu.T
        np.testing.assert_allclose(gemm("gateup", x), np.hstack([G, U]), rtol=0, atol=1e-3 * np.abs(G).max())
        act = from_bf16(to_bf16((G / (1 + np.exp(-np.maximum(G, -80))) * U).astype(np.float32))).astype(np.float64)
        ref = act @ wd.T
        y = gemm.ffn("gateup", "down", x)
        assert y.shape == (T, d)
        np.testing.assert_allclose(y, ref, rtol=0, atol=5e-3 * np.abs(ref).max())
    ref = xb[:2] @ np.vstack([mod.dequantize(g), mod.dequantize(u)]).T.astype(np.float64)  # GEMV: exact weights
    np.testing.assert_allclose(lin("gateup", x[:2]), ref, rtol=0, atol=1e-4 * np.abs(ref).max())
