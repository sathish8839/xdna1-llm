"""GGUF Q4_K weights -> the 4-bit NPU GEMV tile layout (kernels/designs/qgemv.cc).

Q4_K stores each row in super-blocks of 256 weights (144 bytes): an fp16 scale d, an fp16 min dmin, twelve
bytes of packed 6-bit scales sc and mins mn (one pair per 32 weights) and 128 bytes of 4-bit values q.
A weight is d * sc * q - dmin * mn. The kernel uses the same math, so repacking only moves bytes around and
unpacks the 6-bit fields: nothing is rounded.
"""

from __future__ import annotations

from dataclasses import dataclass

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


# Tile geometry, shared by both formats; must match the kernel build (kernels/designs/qgemv.py --geometry).
# A row of K columns takes ceil(K / KC) tiles and the last one may be partly used (its padding is streamed too).
LANES, K_MAX = 8, 7680
# Bytes per super-block of 16 rows (256 columns), multiples of 64 so the kernels' vector loads stay aligned.
# Q4_K: 2048 nibbles + 512 scales/mins + 128 d/dmin; Q6_K: 2048 nibbles + 1024 high bits + 512 scales + 64 d.
SB_BYTES = {4: 2688, 6: 3648}


@dataclass(frozen=True)
class Geometry:
    name: str
    tile: int  # bytes per tile
    kc4: int  # columns per tile, Q4_K
    kc6: int  # columns per tile, Q6_K

    def kc(self, fmt: int) -> int:
        return self.kc4 if fmt == 4 else self.kc6

    def tile_bytes(self, fmt: int) -> int:
        """Bytes of one tile of this format in the repacked weights: whole super-blocks, back to back. The
        kernel moves `tile` bytes per tile, so a shorter tile is read with the start of the next one after it
        (ignored), and the weight buffer needs `tile` bytes of slack at its end (`slack`)."""
        return self.kc(fmt) // QK * SB_BYTES[fmt]

    @property
    def slack(self) -> int:
        return self.tile

    @property
    def x_tiles(self) -> int:  # x (K_MAX bf16) and the header travel in this many tiles
        return -(-2 * K_MAX // self.tile)

    def program(self, fmt: int, N: int, K: int) -> str:
        """Name of the kernel build (xclbin / instruction stream) for one matrix shape."""
        if self.name == "unified":
            return f"q{fmt}_{N}x{K}"
        if fmt != 6:
            raise ValueError(f"the {self.name} geometry only runs Q6_K")
        return f"{self.name}_{N}x{K}"


# unified: one program for Q4_K (1024 columns, 10752 bytes used) and Q6_K (768 columns, 10944 bytes), so a
# Q4_K_M model runs in one context; K = 2048 Q6_K rows stream 12.5 % padding.
UNIFIED = Geometry("unified", 10944, 1024, 768)
# q6only: Q6_K in 512-column tiles (7296 bytes), which divide K = 2048 and 7680 exactly: for all-Q6_K models.
Q6ONLY = Geometry("q6only", 7296, 512, 512)
GEOMETRIES = (UNIFIED, Q6ONLY)
TILE_BYTES, KC = UNIFIED.tile, {4: UNIFIED.kc4, 6: UNIFIED.kc6}


def layout(K: int, fmt: int = 4, geo: Geometry = UNIFIED) -> tuple[int, int]:
    """(columns per tile, tiles per 16-row block)."""
    return geo.kc(fmt), -(-K // geo.kc(fmt))


def x_input(xbits: np.ndarray, fmt: int, N: int, K: int, geo: Geometry = UNIFIED, glu: bool = False) -> np.ndarray:
    """The kernel's x input (geo.x_tiles tiles): x as bf16 bits, then [blocks per lane, tiles per block,
    format + 256 * glu, K] as int32 in the last 16 bytes. glu: an FFN gate + up whose cores emit silu(gate) * up
    (kernels/designs/qgemv.cc)."""
    buf = np.zeros(geo.x_tiles * geo.tile, np.uint8)
    buf[: 2 * K] = np.asarray(xbits, np.uint16).view(np.uint8)
    head = [N // LANES // RB, layout(K, fmt, geo)[1], fmt + 256 * glu, K]
    buf[-16:] = np.array(head, np.int32).view(np.uint8)
    return buf


def sb_tiles(parts: list[np.ndarray], fmt: int, geo: Geometry = UNIFIED) -> np.ndarray:
    """Per-tile byte sections, each (blocks, tiles, n) with n split evenly over the tile's super-blocks ->
    (blocks * tiles * geo.tile_bytes(fmt),): every super-block's sections together."""
    nsb = geo.kc(fmt) // QK
    body = np.concatenate([p.reshape(*p.shape[:2], nsb, -1) for p in parts], axis=3)
    assert body.shape[3] == SB_BYTES[fmt], body.shape
    return body.reshape(-1)


def _hi_lo(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hi = to_bf16(v)
    return hi, to_bf16(v - from_bf16(hi))


def repack(raw: np.ndarray, geo: Geometry = UNIFIED) -> np.ndarray:
    """Q4_K rows, uint8 (N, K/256*144) -> kernel tiles, ordered (16-row block, k chunk). Each tile holds
    super-blocks of 256 columns, SB_BYTES[4] each: nibbles (k pair-major, 16 rows), then per group sc and mn
    for 16 rows, then d_hi, d_lo, dmin_hi, dmin_lo for 16 rows (all bf16). Columns past K are zero."""
    q, sc, mn, d, dmin = unpack(raw)
    N, K = q.shape
    kc, nc = layout(K, 4, geo)
    pad = nc * kc - K
    q = np.pad(q, ((0, 0), (0, pad)))
    sc, mn = (np.pad(a, ((0, 0), (0, pad // G))) for a in (sc, mn))
    d, dmin = (np.pad(a, ((0, 0), (0, pad // QK))) for a in (d, dmin))
    nb, ngc, nsc = N // RB, kc // G, kc // QK

    qb = q.reshape(nb, RB, nc, kc // 2, 2)
    nib = (qb[..., 0] | (qb[..., 1] << 4)).astype(np.uint8).transpose(0, 2, 3, 1)  # (block, chunk, k pair, row)
    scm = rows_last([to_bf16(sc.astype(np.float32)), to_bf16(mn.astype(np.float32))], nb, nc, ngc)
    dd = rows_last([*_hi_lo(d), *_hi_lo(dmin)], nb, nc, nsc)
    return sb_tiles([nib.reshape(nb, nc, -1), scm, dd], 4, geo)


def rows_last(parts, nb, nc, n):
    """(N, nc * n) arrays -> bytes (block, chunk, n, len(parts), row): per item, the parts for 16 rows each."""
    a = np.stack([p.reshape(nb, RB, nc, n) for p in parts], axis=-1)
    return np.ascontiguousarray(a.transpose(0, 2, 3, 4, 1)).view(np.uint8).reshape(nb, nc, -1)
