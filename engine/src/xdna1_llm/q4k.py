"""GGUF Q4_K weights -> the 4-bit NPU GEMV tile layout (kernels/designs/q4gemv.cc).

Q4_K stores each row in super-blocks of 256 weights (144 bytes): an fp16 scale d, an fp16 min dmin, twelve
bytes of packed 6-bit scales sc and mins mn (one pair per 32 weights) and 128 bytes of 4-bit values q.
A weight is d * sc * q - dmin * mn. The kernel uses the same math, so repacking only moves bytes around and
unpacks the 6-bit fields: nothing is rounded.
"""

from __future__ import annotations

import numpy as np

from .bf16 import from_bf16, to_bf16

QK = 256  # weights per Q4_K super-block
BLOCK_BYTES = 144
RB, G = 16, 32  # kernel: rows per block, weights per scale group


def unpack(raw: np.ndarray):
    """Q4_K rows, uint8 (N, K/256*144) -> q uint8 (N, K), sc and mn uint8 (N, K/32), d and dmin float32
    (N, K/256), with w = d * sc * q - dmin * mn."""
    N = raw.shape[0]
    b = raw.reshape(N, -1, BLOCK_BYTES)
    d = b[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float32)
    dmin = b[..., 2:4].copy().view(np.float16)[..., 0].astype(np.float32)
    sc12 = b[..., 4:16].reshape(N, -1, 3, 4)
    lo, hi, mix = sc12[..., 0, :], sc12[..., 1, :], sc12[..., 2, :]
    # 8 six-bit scales and mins: the first four in the low 6 bits of bytes 0-3 / 4-7, the last four split
    # between the nibbles of bytes 8-11 and the top 2 bits of bytes 0-7
    sc = np.concatenate([lo & 0x3F, (mix & 0x0F) | ((lo >> 2) & 0x30)], axis=-1).reshape(N, -1)
    mn = np.concatenate([hi & 0x3F, (mix >> 4) | ((hi >> 2) & 0x30)], axis=-1).reshape(N, -1)
    # 32-byte chunk j holds group 2j in its low nibbles and group 2j+1 in its high nibbles
    qs = b[..., 16:].reshape(N, -1, 4, 1, 32) >> np.array([0, 4], np.uint8).reshape(2, 1)
    q = (qs & 0x0F).reshape(N, -1)
    return q, sc, mn, d, dmin


def dequantize(raw: np.ndarray) -> np.ndarray:
    """Q4_K rows -> float32 (N, K)."""
    q, sc, mn, d, dmin = unpack(raw)
    s = np.repeat(d, 8, axis=1) * sc
    m = np.repeat(dmin, 8, axis=1) * mn
    return np.repeat(s, G, axis=1) * q - np.repeat(m, G, axis=1)


def layout(K: int) -> tuple[int, int, int]:
    """(k per tile, tile bytes, tiles that carry x); must match kernels/designs/q4gemv.py."""
    KC = K if K <= 2048 else max(d for d in range(QK, 1537, QK) if K % d == 0)
    TB = KC // 2 * RB + KC // G * 2 * RB * 2 + KC // QK * 4 * RB * 2
    return KC, TB, -(-(2 * K + 4) // TB)


def _hi_lo(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hi = to_bf16(v)
    return hi, to_bf16(v - from_bf16(hi))


def repack(raw: np.ndarray) -> np.ndarray:
    """Q4_K rows, uint8 (N, K/256*144) -> kernel tiles, ordered (16-row block, k chunk). Each tile:
    nibbles (k pair-major, 16 rows), then per group sc and mn for 16 rows, then per super-block d_hi, d_lo,
    dmin_hi, dmin_lo for 16 rows (all bf16)."""
    q, sc, mn, d, dmin = unpack(raw)
    N, K = q.shape
    KC = layout(K)[0]
    nb, nc, ngc, nsc = N // RB, K // KC, KC // G, KC // QK

    qb = q.reshape(nb, RB, nc, KC // 2, 2)
    nib = (qb[..., 0] | (qb[..., 1] << 4)).astype(np.uint8).transpose(0, 2, 3, 1)  # (block, chunk, k pair, row)

    def rows_last(parts, n):  # list of (N, ...) arrays -> (block, chunk, n, len(parts), row) bf16 bytes
        a = np.stack([p.reshape(nb, RB, nc, n) for p in parts], axis=-1)
        return np.ascontiguousarray(a.transpose(0, 2, 3, 4, 1)).view(np.uint8).reshape(nb, nc, -1)

    scm = rows_last([to_bf16(sc.astype(np.float32)), to_bf16(mn.astype(np.float32))], ngc)
    dd = rows_last([*_hi_lo(d), *_hi_lo(dmin)], nsc)
    return np.concatenate([nib.reshape(nb, nc, -1), scm, dd], axis=2).reshape(-1)
