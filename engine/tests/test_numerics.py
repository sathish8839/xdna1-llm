import numpy as np
import pytest

from xdna1_llm.bf16 import from_bf16, to_bf16
from xdna1_llm.cpu_linear import CpuLinear
from xdna1_llm.models.lfm3 import LCONV, LFM3, apply_rope, rms, rope_tables

rng = np.random.default_rng(0)


def test_bf16_roundtrip_exact_for_representable_values():
    x = np.array([0.0, 1.0, -2.5, 3.140625, 65280.0, -1e-3], np.float32)
    x = from_bf16(to_bf16(x))  # snap to bf16 once
    np.testing.assert_array_equal(from_bf16(to_bf16(x)), x)


def test_bf16_round_to_nearest_even():
    # 1 + 2^-8 is exactly halfway between bf16 neighbours 1.0 and 1 + 2^-7; ties go to the even mantissa (1.0)
    assert from_bf16(to_bf16(np.float32(1 + 2**-8)))[()] == 1.0
    assert from_bf16(to_bf16(np.float32(1 + 3 * 2**-8)))[()] == 1 + 2**-6


def test_bf16_relative_error_bound():
    x = rng.standard_normal(10_000).astype(np.float32)
    rel = np.abs(from_bf16(to_bf16(x)) - x) / np.abs(x)
    assert rel.max() <= 2**-8


def test_rms_norm():
    x = rng.standard_normal((4, 64)).astype(np.float32)
    w = rng.standard_normal(64).astype(np.float32)
    ref = x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-5) * w
    np.testing.assert_allclose(rms(x, w), ref, rtol=1e-5, atol=1e-6)


def test_rope_position_zero_is_identity():
    cos, sin = rope_tables(4, 128, 1e6, 32)
    x = rng.standard_normal((4, 2, 128)).astype(np.float32)
    np.testing.assert_allclose(apply_rope(x, cos, sin)[0], x[0], atol=1e-6)


def test_rope_partial_rotary_leaves_tail_pairs_untouched():
    hd, pairs = 128, 32
    cos, sin = rope_tables(8, hd, 1e6, pairs)
    x = rng.standard_normal((8, 2, hd)).astype(np.float32)
    y = apply_rope(x, cos, sin)
    half = hd // 2
    frozen = np.r_[pairs:half, half + pairs : hd]
    np.testing.assert_allclose(y[..., frozen], x[..., frozen], atol=1e-6)
    assert not np.allclose(y[1:, :, :pairs], x[1:, :, :pairs])


def test_rope_preserves_pair_norms_and_relative_positions():
    hd = 64
    cos, sin = rope_tables(16, hd, 1e4, hd // 2)
    q = np.tile(rng.standard_normal((1, 1, hd)).astype(np.float32), (16, 1, 1))
    k = np.tile(rng.standard_normal((1, 1, hd)).astype(np.float32), (16, 1, 1))
    rq, rk = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    np.testing.assert_allclose(np.linalg.norm(rq, axis=-1), np.linalg.norm(q, axis=-1), rtol=1e-5)
    dots = [float(rq[i + 3, 0] @ rk[i, 0]) for i in range(10)]  # depends only on the offset (3)
    np.testing.assert_allclose(dots, dots[0], rtol=1e-4, atol=1e-4)


def test_cpu_linear_matches_fp32_matmul_on_bf16_weights():
    w = rng.standard_normal((16, 32)).astype(np.float32)
    x = rng.standard_normal((3, 32)).astype(np.float32)
    lin = CpuLinear()
    lin.register("w", to_bf16(w))
    np.testing.assert_allclose(lin("w", x), x @ from_bf16(to_bf16(w)).T, rtol=1e-6)


def test_short_conv_matches_naive_causal_convolution():
    d, T = 8, 11
    m = object.__new__(LFM3)  # exercise _conv without loading a GGUF
    m.d, m.cpu_ms = d, 0.0
    m.p = {"b.conv": rng.standard_normal((d, LCONV)).astype(np.float32)}

    class Identity:
        def __call__(self, name, x):
            return x

    m.lin = Identity()
    P = rng.standard_normal((T, 3 * d)).astype(np.float32)
    out = m._conv("b.", P)

    B, C, X = P[:, :d], P[:, d : 2 * d], P[:, 2 * d :]
    z = B * X
    ref = np.zeros_like(z)
    for t in range(T):
        for k in range(LCONV):  # tap LCONV-1 multiplies the newest input
            src = t - (LCONV - 1 - k)
            if src >= 0:
                ref[t] += m.p["b.conv"][:, k] * z[src]
    np.testing.assert_allclose(out, C * ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("T", [1, 5])
def test_conv_is_causal(T):
    d = 4
    m = object.__new__(LFM3)
    m.d, m.cpu_ms = d, 0.0
    m.p = {"b.conv": rng.standard_normal((d, LCONV)).astype(np.float32)}
    m.lin = lambda name, x: x
    P = rng.standard_normal((T + 3, 3 * d)).astype(np.float32)
    full = m._conv("b.", P)
    prefix = m._conv("b.", P[:T])
    np.testing.assert_allclose(full[:T], prefix, rtol=1e-6)
