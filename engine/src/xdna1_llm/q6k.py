"""GGUF Q6_K weights -> the 6-bit NPU GEMV tile layout (kernels/designs/qgemv.cc).

Q6_K stores each row in super-blocks of 256 weights (210 bytes): 128 bytes with the low 4 bits of each value,
64 bytes with the high 2 bits, sixteen int8 scales sc (one per 16 weights) and an fp16 scale d.
A weight is d * sc * (q - 32) with q in 0..63. The kernel uses the same math, so repacking only moves bits
around: nothing is rounded.
"""

from __future__ import annotations

import numpy as np

from . import q4k
from .bf16 import to_bf16

QK = 256  # weights per Q6_K super-block
BLOCK_BYTES = 210
RB, G = 16, 16  # kernel: rows per block, weights per scale group


def unpack(raw: np.ndarray):
    """Q6_K rows, uint8 (N, K/256*210) -> q uint8 (N, K) in 0..63, sc int8 (N, K/16), d float32 (N, K/256),
    with w = d * sc * (q - 32)."""
    N = raw.shape[0]
    b = raw.reshape(N, -1, BLOCK_BYTES)
    nb = b.shape[1]
    # each half super-block (128 weights) takes 64 bytes of low nibbles (low then high nibble = weights 0-63 and
    # 64-127) and 32 bytes of high bits (bit pairs 0, 2, 4, 6 = weights 0-31, 32-63, 64-95, 96-127)
    ql = b[..., :128].reshape(N, nb, 2, 1, 64) >> np.array([0, 4], np.uint8).reshape(2, 1)
    qh = b[..., 128:192].reshape(N, nb, 2, 1, 32) >> np.array([0, 2, 4, 6], np.uint8).reshape(4, 1)
    q = (ql & 0x0F).reshape(N, nb, 2, 4, 32) | ((qh & 0x03) << 4).reshape(N, nb, 2, 4, 32)
    sc = b[..., 192:208].copy().view(np.int8).reshape(N, -1)
    d = b[..., 208:210].copy().view(np.float16)[..., 0].astype(np.float32)
    return q.reshape(N, -1), sc, d


def dequantize(raw: np.ndarray) -> np.ndarray:
    """Q6_K rows -> float32 (N, K)."""
    q, sc, d = unpack(raw)
    s = np.repeat(d, QK // G, axis=1) * sc
    return np.repeat(s, G, axis=1) * (q.astype(np.float32) - 32)


def layout(K: int, geo: q4k.Geometry = q4k.UNIFIED) -> tuple[int, int]:
    """(columns per tile, tiles per 16-row block)."""
    return q4k.layout(K, 6, geo)


def repack(raw: np.ndarray, geo: q4k.Geometry = q4k.UNIFIED) -> np.ndarray:
    """Q6_K rows, uint8 (N, K/256*210) -> kernel tiles, ordered (16-row block, k chunk). Each tile holds
    super-blocks of 256 columns, q4k.SB_BYTES[6] each:
    low nibbles: per k pair, 16 bytes (row i: q[2p] & 15 | (q[2p+1] & 15) << 4), as in the Q4_K tiles
    high bits:   per 16 k, 64 bytes; byte i (row i % 16, j = i // 16) holds the high 2 bits of
                 k = 2j, 2j+1, 8+2j, 9+2j in bits 0-1, 2-3, 4-5, 6-7 (the lane order of 64-byte nibble loads)
    scales:      per 16 k, sc for 16 rows (bf16, exact)
    d:           d_hi, d_lo for 16 rows (bf16, d_hi + d_lo == d exactly)
    Columns past K are zero."""
    q, sc, d = unpack(raw)
    N, K = q.shape
    KC, nc = layout(K, geo)
    pad = nc * KC - K
    q = np.pad(q, ((0, 0), (0, pad)))
    sc = np.pad(sc, ((0, 0), (0, pad // G)))
    d = np.pad(d, ((0, 0), (0, pad // QK)))
    nb = N // RB

    qb = q.reshape(nb, RB, nc, KC // 2, 2) & 0x0F
    nib = (qb[..., 0] | (qb[..., 1] << 4)).astype(np.uint8).transpose(0, 2, 3, 1)  # (block, chunk, k pair, row)

    h = (q >> 4).reshape(nb, RB, nc, KC // 16, 2, 4, 2)  # (block, row, chunk, group, half, j, k in pair)
    hb = h[..., 0, :, 0] | (h[..., 0, :, 1] << 2) | (h[..., 1, :, 0] << 4) | (h[..., 1, :, 1] << 6)
    hb = hb.astype(np.uint8).transpose(0, 2, 3, 4, 1)  # (block, chunk, group, j, row)

    scv = q4k.rows_last([to_bf16(sc.astype(np.float32))], nb, nc, KC // G)
    dd = q4k.rows_last([*q4k._hi_lo(d)], nb, nc, KC // QK)
    return q4k.sb_tiles([nib.reshape(nb, nc, -1), hb.reshape(nb, nc, -1), scv, dd], 6, geo)
