"""Q6_K unpacking and the NPU tile layout, checked against the gguf package's reference dequantizer."""

import numpy as np
import pytest
from gguf import GGMLQuantizationType
from gguf.quants import dequantize as gguf_dequantize

from xdna1_llm import q4k, q6k
from xdna1_llm.bf16 import from_bf16


def random_q6k(n: int, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(n, k // q6k.QK, q6k.BLOCK_BYTES), dtype=np.uint8)
    raw[..., 208:210] = rng.uniform(1e-5, 1e-4, size=(n, k // q6k.QK)).astype(np.float16)[..., None].view(np.uint8)
    return raw.reshape(n, -1)


def unpack_tiles(tiles: np.ndarray, n: int, k: int, geo=q4k.UNIFIED) -> np.ndarray:
    """Read the kernel's tile layout back into float weights (what qgemv.cc computes, in numpy)."""
    KC, nc = q6k.layout(k, geo)
    RB, kp = q6k.RB, nc * KC
    t = tiles.reshape(n // RB, nc, geo.tile)
    nib = t[..., : KC * 8].reshape(*t.shape[:2], KC // 2, RB)
    lo = np.stack([nib & 0x0F, nib >> 4], axis=-1)  # (block, chunk, k pair, row, k in pair)
    lo = lo.transpose(0, 3, 1, 2, 4).reshape(n, kp)
    hb = t[..., KC * 8 : KC * 12].reshape(*t.shape[:2], KC // 16, 4, RB)  # (block, chunk, group, j, row)
    h = np.stack([(hb >> s) & 3 for s in (0, 2, 4, 6)], axis=-1).reshape(*hb.shape, 2, 2)  # (.., half, k in pair)
    h = h.transpose(0, 4, 1, 2, 5, 3, 6).reshape(n, kp)  # (block, row, chunk, group, half, j, k in pair)
    q = lo | (h << 4)
    rest = t[..., KC * 12 :].copy().view(np.uint16)
    ng = KC // q6k.G
    sc = from_bf16(rest[..., : ng * RB]).reshape(*t.shape[:2], ng, RB).transpose(0, 3, 1, 2).reshape(n, -1)
    dd = (
        from_bf16(rest[..., ng * RB : (ng + KC // q6k.QK * 2) * RB])
        .reshape(*t.shape[:2], KC // q6k.QK, 2, RB)
        .astype(np.float64)
    )
    d = (dd[..., 0, :] + dd[..., 1, :]).transpose(0, 3, 1, 2).reshape(n, -1)
    s = np.repeat(d, q6k.QK // q6k.G, axis=1) * sc
    return (np.repeat(s, q6k.G, axis=1) * (q.astype(np.float64) - 32)).astype(np.float32)[:, :k]


def test_dequantize_matches_gguf():
    raw = random_q6k(32, 512)
    ref = gguf_dequantize(raw, GGMLQuantizationType.Q6_K)
    np.testing.assert_array_equal(q6k.dequantize(raw), ref)


@pytest.mark.parametrize("geo", q4k.GEOMETRIES, ids=lambda g: g.name)
@pytest.mark.parametrize("k", [512, 2048, 7680])
def test_repack_is_lossless(k, geo):
    n = 32
    raw = random_q6k(n, k, seed=k)
    np.testing.assert_array_equal(unpack_tiles(q6k.repack(raw, geo), n, k, geo), q6k.dequantize(raw))


def test_q6only_tiles_have_no_padding():
    assert q6k.repack(random_q6k(16, 2048), q4k.Q6ONLY).size == 4 * q4k.Q6ONLY.tile
    assert q6k.layout(7680, q4k.Q6ONLY) == (512, 15)
