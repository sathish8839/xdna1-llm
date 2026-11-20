"""Q4_K unpacking and the NPU tile layout, checked against the gguf package's reference dequantizer."""

import numpy as np
import pytest
from gguf import GGMLQuantizationType
from gguf.quants import dequantize as gguf_dequantize

from xdna1_llm import q4k
from xdna1_llm.bf16 import from_bf16


def random_q4k(n: int, k: int, seed: int = 0) -> np.ndarray:
    """Random Q4_K rows with sane fp16 scales (every other byte pattern is a valid block)."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(n, k // q4k.QK, q4k.BLOCK_BYTES), dtype=np.uint8)
    raw[..., 0:2] = rng.uniform(1e-4, 1e-3, size=(n, k // q4k.QK)).astype(np.float16)[..., None].view(np.uint8)
    raw[..., 2:4] = rng.uniform(0, 1e-3, size=(n, k // q4k.QK)).astype(np.float16)[..., None].view(np.uint8)
    return raw.reshape(n, -1)


def unpack_tiles(tiles: np.ndarray, n: int, k: int) -> np.ndarray:
    """Read the kernel's tile layout back into float weights (what q4gemv.cc computes, in numpy)."""
    KC, TB, _ = q4k.layout(k)
    ngc, nsc = KC // q4k.G, KC // q4k.QK
    t = tiles.reshape(n // q4k.RB, k // KC, TB)
    nib = t[..., : KC * 8].reshape(*t.shape[:2], KC // 2, q4k.RB)
    q = np.stack([nib & 0x0F, nib >> 4], axis=-1).transpose(0, 3, 1, 2, 4).reshape(n, k)  # (row, k)
    rest = t[..., KC * 8 :].copy().view(np.uint16)
    scm = from_bf16(rest[..., : ngc * 2 * q4k.RB]).reshape(*t.shape[:2], ngc, 2, q4k.RB)
    dd = from_bf16(rest[..., ngc * 2 * q4k.RB :]).reshape(*t.shape[:2], nsc, 4, q4k.RB).astype(np.float64)
    sc, mn = (scm[..., i, :].transpose(0, 3, 1, 2).reshape(n, -1) for i in (0, 1))
    d = (dd[..., 0, :] + dd[..., 1, :]).transpose(0, 3, 1, 2).reshape(n, -1)
    dmin = (dd[..., 2, :] + dd[..., 3, :]).transpose(0, 3, 1, 2).reshape(n, -1)
    s = np.repeat(d, 8, axis=1) * sc
    m = np.repeat(dmin, 8, axis=1) * mn
    return (np.repeat(s, q4k.G, axis=1) * q - np.repeat(m, q4k.G, axis=1)).astype(np.float32)


def test_dequantize_matches_gguf():
    raw = random_q4k(32, 512)
    ref = gguf_dequantize(raw, GGMLQuantizationType.Q4_K)
    np.testing.assert_array_equal(q4k.dequantize(raw), ref)


@pytest.mark.parametrize("k", [512, 2048, 7680])
def test_repack_is_lossless(k):
    n = 32
    raw = random_q4k(n, k, seed=k)
    np.testing.assert_array_equal(unpack_tiles(q4k.repack(raw), n, k), q4k.dequantize(raw))
