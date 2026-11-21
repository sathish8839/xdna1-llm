// Quantized GEMV for one decode token on an npu1 core: y = W . x (fp32 out, bf16 x) with W in GGUF Q4_K or Q6_K.
// The format, the row length K and the sizes arrive at run time, so one core program serves every LFM3 matrix and
// decode runs in a single hardware context (switching contexts costs ~1.45 ms per call on the Windows driver).
//
//   Q4_K: W[r, k] = d[r, b] * sc[r, g] * q[r, k] - dmin[r, b] * mn[r, g]   q in 0..15, g = k / 32, b = k / 256
//   Q6_K: W[r, k] = d[r, b] * sc[r, g] * (q[r, k] - 32)                   q in 0..63, g = k / 16
// Both are the formats' own math, so results match llama.cpp's dequantization to fp32 rounding.
//
// Every weight tile is TILE_BYTES and covers 16 rows (vector lanes = rows) and KC4 (Q4_K) or KC6 (Q6_K) columns.
// The geometry is a build option (xdna1_llm.q4k.Geometry): the unified build sizes both formats to (nearly) the
// same tile so a Q4_K_M model runs in one program; the q6only build uses 512-column Q6_K tiles, which divide
// LFM3's K = 2048 and 7680, so no padding is streamed. A row of K columns takes ceil(K / KC) tiles; the last one
// may be partly used. A tile holds super-blocks of 256 columns, SB4 / SB6 bytes each, back to back in memory
// across tiles too (the W4A16 GEMM streams them one at a time). The core always receives TILE_BYTES, so a
// shorter Q4_K tile arrives with the start of the next one after it, which it ignores. Each super-block
// (xdna1_llm/q4k.py and q6k.py write this):
//   Q4_K: nibbles   128 x 16 B: chunk p holds q[i, 2p] | q[i, 2p+1] << 4 for rows i = 0..15
//         groups    per 32 k: sc[16], mn[16] (bf16, small integers, exact)
//         d         d_hi[16], d_lo[16], dmin_hi[16], dmin_lo[16] (bf16, hi + lo == the fp16 exactly)
//   Q6_K: nibbles   128 x 16 B: the low 4 bits, as above
//         high bits per 16 k, 64 B: byte i (row i % 16, j = i / 16) holds the high 2 bits of k = 2j, 2j+1,
//                   8+2j, 9+2j (relative to the group's 16 k) in bits 0-1, 2-3, 4-5, 6-7: the lane order of the
//                   two 64-byte nibble vectors of the group, so merging needs no shuffles
//         groups    per 16 k: sc[16] (bf16)
//         d         d_hi[16], d_lo[16]
// x arrives first, over X_TILES tiles: x as bf16 from byte 0, and a header in the last 16 bytes of the last tile:
// [16-row blocks for this lane, tiles per block, format (4 or 6) + 256 * glu, K].
//
// glu = 1 (FFN gate + up, rows stored in 16-row blocks gate, up, gate, ...): after each up block the core writes
// bf16(silu(gate) * up) of the pair's 16 rows to the first 32 bytes of the block's output (silu_lut.h); the
// gate block's output is left as it is. The host's DMA pattern (qgemv.py --glu) puts those 32 bytes where the
// down projection reads its x, and the rest elsewhere.
//
// Quantized values become bf16 exactly with the exponent trick (0x4300 | q) == 128 + q; the offset (128, plus 32
// for Q6_K) times sum(x) is removed per group. AIE2 has no fast fp32 multiply, so every scaling step is a native
// bf16 MAC on fp32 values split into bf16 high and low parts.
//
// Speed (the loops are bound by instruction latency, not MACs): bit masks run on full 64-byte vectors; bytes
// become bf16 with one shuffle (zip with 0x43); x is read as scalars from memory and broadcast; each group's MACs
// go to 4 independent accumulators; and the group loop is unrolled twice so two groups' chains overlap.
//
// IRON compiles one object per ExternalFunction, so each function is built on its own (QG_FN) and they share
// state through globals.
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef TILE_BYTES  // unified geometry: Q4_K 4 super-blocks per tile (10752 bytes used), Q6_K 3 (10944)
#define TILE_BYTES 10944
#define KC4 1024
#define KC6 768
#endif
#define K_MAX 7680
#define X_TILES ((2 * K_MAX + TILE_BYTES - 1) / TILE_BYTES)
#define RB 16
#define QK 256
#define SB4 2688  // bytes per super-block (xdna1_llm.q4k.SB_BYTES); a Q4_K tile is shorter than TILE_BYTES
#define SB6 3648

#if QG_FN == 1
#include "silu_lut.h"  // glu(): bf16(silu(g) * u)
#endif

extern "C" {

#if QG_FN == 0
int32_t qg_fmt, qg_k, qg_tiles, qg_chunk;  // format, row length, tiles per block, next tile of the current block
int32_t qg_glu, qg_odd;                    // glu mode; the current block is an up block
alignas(aie::vector_decl_align) float qg_gate[RB];  // glu: the gate block's outputs (vector loads need the alignment)
float qg_acc[RB];                          // partial outputs of the current block
float qg_xoff[K_MAX / 16];                 // per group: (128 or 160) * sum(x)
bfloat16 qg_xs_hi[K_MAX / 32], qg_xs_lo[K_MAX / 32];  // Q4_K, per group: sum(x) as bf16 high + low parts
static int32_t xpart;

void qg_load_x(const uint8_t *__restrict src, bfloat16 *__restrict x, int32_t *__restrict hdr) {
  const bfloat16 *s = (const bfloat16 *)src;
  const int begin = xpart * (TILE_BYTES / 2), end = begin + TILE_BYTES / 2 < K_MAX ? begin + TILE_BYTES / 2 : K_MAX;
  for (int i = begin; i < end; i += 32)
    aie::store_v(x + i, aie::load_v<32>(s + (i - begin)));
  if (++xpart < X_TILES)
    return;
  xpart = 0;
  const int32_t *h = (const int32_t *)(src + TILE_BYTES - 16);
  hdr[0] = h[0];
  hdr[1] = h[1];
  qg_tiles = h[1];
  qg_fmt = h[2] & 0xFF;
  qg_glu = h[2] >> 8;
  qg_odd = 0;
  qg_k = h[3];
  if (qg_fmt == 4) {
    for (int g = 0; g < qg_k / 32; g++) {
      aie::accum<accfloat, 32> a;
      a.from_vector(aie::load_v<32>(x + g * 32));
      float xs = aie::reduce_add(a.to_vector<float>());
      qg_xoff[g] = 128.0f * xs;
      qg_xs_hi[g] = (bfloat16)xs;
      qg_xs_lo[g] = (bfloat16)(xs - (float)qg_xs_hi[g]);
    }
  } else {
    for (int g = 0; g < qg_k / 16; g++) {
      aie::accum<accfloat, 16> a;
      a.from_vector(aie::load_v<16>(x + g * 16));
      qg_xoff[g] = 160.0f * aie::reduce_add(a.to_vector<float>());
    }
  }
  qg_chunk = 0;
  aie::store_v(qg_acc, aie::zeros<float, RB>());
}
#else
extern int32_t qg_fmt, qg_k, qg_tiles, qg_chunk;
extern int32_t qg_glu, qg_odd;
extern float qg_gate[RB];
extern float qg_acc[RB];
extern float qg_xoff[K_MAX / 16];
extern bfloat16 qg_xs_hi[K_MAX / 32], qg_xs_lo[K_MAX / 32];

typedef aie::vector<bfloat16, 16> vbf16_t;
typedef aie::accum<accfloat, 16> accf16_t;

static inline void split(aie::vector<float, 16> v, vbf16_t &hi, vbf16_t &lo) {
  hi = accf16_t(v).to_vector<bfloat16>();
  accf16_t h;
  h.from_vector(hi);
  lo = accf16_t(aie::sub(v, h.to_vector<float>())).to_vector<bfloat16>();
}

// 64 quantized values (bytes) -> two vectors of 32 bf16 128 + q: zipping in 0x43 as each high byte is one
// shuffle per output vector (unpack + or would be two ALU operations, on half-width vectors)
static inline void to_bf16_zip(aie::vector<uint8_t, 64> q, aie::vector<bfloat16, 32> &a, aie::vector<bfloat16, 32> &b) {
  const auto [lo, hi] = aie::interleave_zip(q, aie::broadcast<uint8_t, 64>(0x43), 1);
  a = aie::vector_cast<bfloat16>(lo);
  b = aie::vector_cast<bfloat16>(hi);
}

// lo: lanes 0-15 rows at k = xp[0], lanes 16-31 at xp[2]; hi: xp[1] and xp[3]. mac_elem_16_2 adds lanes i and
// i + 16 into accumulator lane i, so one instruction does 32 multiply-adds. x is read as scalars from memory
// (load slots) rather than extracted from a vector register.
static inline v16accfloat quad(v16accfloat a, aie::vector<bfloat16, 32> lo, aie::vector<bfloat16, 32> hi,
                               const bfloat16 *__restrict xp) {
  aie::vector<bfloat16, 32> x02 = aie::concat(aie::broadcast<bfloat16, 16>(xp[0]), aie::broadcast<bfloat16, 16>(xp[2]));
  aie::vector<bfloat16, 32> x13 = aie::concat(aie::broadcast<bfloat16, 16>(xp[1]), aie::broadcast<bfloat16, 16>(xp[3]));
  a = mac_elem_16_2(lo.to_native(), x02.to_native(), a);
  return mac_elem_16_2(hi.to_native(), x13.to_native(), a);
}

// sum of independent accumulators (one MAC chain each, so their latencies overlap)
static inline aie::vector<float, 16> sum4(v16accfloat a0, v16accfloat a1, v16accfloat a2, v16accfloat a3) {
  return aie::add(aie::add(accf16_t(a0), accf16_t(a1)), aie::add(accf16_t(a2), accf16_t(a3))).to_vector<float>();
}

static inline accf16_t q4_tile(const uint8_t *__restrict w, const bfloat16 *__restrict x, accf16_t acc_y, int nsb) {
  const aie::vector<uint8_t, 64> m0f = aie::broadcast<uint8_t, 64>(0x0F);
  const int g0 = qg_chunk * (KC4 / 32);
  for (int b = 0; b < nsb; b++) {
    const uint8_t *__restrict wb = w + b * SB4;
    const bfloat16 *__restrict scm = (const bfloat16 *)(wb + QK / 2 * RB);
    accf16_t acc_s = aie::zeros<accfloat, 16>(), acc_m = acc_s;  // sum of sc * dot, mn * xsum
    _Pragma("clang loop unroll_count(2)")
    for (int gb = 0; gb < 8; gb++) {
      const int gc = b * 8 + gb, g = g0 + gc;
      const uint8_t *__restrict wg = wb + gb * 16 * RB;
      const bfloat16 *__restrict xg = x + g * 32;
      v16accfloat a[4];
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < 4; i++) {  // 64 bytes = k pairs j = 0..3 (then row): even k low, odd k high
        const aie::vector<uint8_t, 64> v = aie::load_v<64>(wg + i * 4 * RB);
        aie::vector<bfloat16, 32> ea, eb, oa, ob;
        to_bf16_zip(aie::bit_and(v, m0f), ea, eb);
        to_bf16_zip(aie::downshift(v, 4), oa, ob);
        a[i] = quad(quad(aie::zeros<accfloat, 16>().to_native(), ea, oa, xg + i * 8), eb, ob, xg + i * 8 + 4);
      }
      vbf16_t dot_hi, dot_lo;
      split(aie::sub(sum4(a[0], a[1], a[2], a[3]), qg_xoff[g]), dot_hi, dot_lo);
      vbf16_t sc = aie::load_v<16>(scm + gb * 2 * RB), mn = aie::load_v<16>(scm + gb * 2 * RB + RB);
      acc_s = aie::mac(aie::mac(acc_s, sc, dot_hi), sc, dot_lo);
      acc_m = aie::mac(aie::mac(acc_m, mn, qg_xs_hi[g]), mn, qg_xs_lo[g]);
    }
    // y += d * S - dmin * M (the lo * lo terms are below fp32 rounding)
    const bfloat16 *__restrict db = scm + 8 * 2 * RB;
    vbf16_t d_hi = aie::load_v<16>(db), d_lo = aie::load_v<16>(db + RB);
    vbf16_t dm_hi = aie::load_v<16>(db + 2 * RB), dm_lo = aie::load_v<16>(db + 3 * RB);
    vbf16_t s_hi, s_lo, m_hi, m_lo;
    split(acc_s.to_vector<float>(), s_hi, s_lo);
    split(acc_m.to_vector<float>(), m_hi, m_lo);
    acc_y = aie::mac(aie::mac(aie::mac(acc_y, d_hi, s_hi), d_hi, s_lo), d_lo, s_hi);
    acc_y = aie::msc(aie::msc(aie::msc(acc_y, dm_hi, m_hi), dm_hi, m_lo), dm_lo, m_hi);
  }
  return acc_y;
}

static inline accf16_t q6_tile(const uint8_t *__restrict w, const bfloat16 *__restrict x, accf16_t acc_y, int nsb) {
  const aie::vector<uint8_t, 64> m0f = aie::broadcast<uint8_t, 64>(0x0F), m30 = aie::broadcast<uint8_t, 64>(0x30);
  const int g0 = qg_chunk * (KC6 / 16);
  for (int b = 0; b < nsb; b++) {
    const uint8_t *__restrict wb = w + b * SB6;
    const uint8_t *__restrict hbits = wb + QK / 2 * RB;
    const bfloat16 *__restrict scv = (const bfloat16 *)(hbits + QK / 16 * 64);
    const bfloat16 *__restrict dd = scv + QK / 16 * RB;
    accf16_t acc_s = aie::zeros<accfloat, 16>();  // sum of sc * dot
    _Pragma("clang loop unroll_count(2)")
    for (int gb = 0; gb < 16; gb++) {
      const int gc = b * 16 + gb, g = g0 + gc;
      const bfloat16 *__restrict xg = x + g * 16;
      // 16 k x 16 rows: n0 holds k 0-7 (lanes: k pair j = 0..3, then row), n1 k 8-15, h their high bits
      const uint8_t *__restrict wg = wb + gb * 8 * RB;
      const aie::vector<uint8_t, 64> n0 = aie::load_v<64>(wg), n1 = aie::load_v<64>(wg + 64);
      const aie::vector<uint8_t, 64> h = aie::load_v<64>(hbits + gb * 64);
      aie::vector<bfloat16, 32> e0a, e0b, o0a, o0b, e1a, e1b, o1a, o1b;  // even / odd k, lanes 0-31 / 32-63
      to_bf16_zip(aie::bit_or(aie::bit_and(n0, m0f), aie::bit_and(aie::upshift(h, 4), m30)), e0a, e0b);
      to_bf16_zip(aie::bit_or(aie::downshift(n0, 4), aie::bit_and(aie::upshift(h, 2), m30)), o0a, o0b);
      to_bf16_zip(aie::bit_or(aie::bit_and(n1, m0f), aie::bit_and(h, m30)), e1a, e1b);
      to_bf16_zip(aie::bit_or(aie::downshift(n1, 4), aie::bit_and(aie::downshift(h, 2), m30)), o1a, o1b);
      const v16accfloat z = aie::zeros<accfloat, 16>().to_native();
      const v16accfloat a0 = quad(z, e0a, o0a, xg), a1 = quad(z, e0b, o0b, xg + 4);
      const v16accfloat a2 = quad(z, e1a, o1a, xg + 8), a3 = quad(z, e1b, o1b, xg + 12);
      vbf16_t dot_hi, dot_lo;
      split(aie::sub(sum4(a0, a1, a2, a3), qg_xoff[g]), dot_hi, dot_lo);
      vbf16_t sc = aie::load_v<16>(scv + gb * RB);
      acc_s = aie::mac(aie::mac(acc_s, sc, dot_hi), sc, dot_lo);
    }
    vbf16_t d_hi = aie::load_v<16>(dd), d_lo = aie::load_v<16>(dd + RB);
    vbf16_t s_hi, s_lo;
    split(acc_s.to_vector<float>(), s_hi, s_lo);
    acc_y = aie::mac(aie::mac(aie::mac(acc_y, d_hi, s_hi), d_hi, s_lo), d_lo, s_hi);
  }
  return acc_y;
}

void qg_rows(const uint8_t *__restrict w, const bfloat16 *__restrict x, float *__restrict y) {
  accf16_t acc_y;
  acc_y.from_vector(aie::load_v<16>(qg_acc));
  const int sbt = qg_fmt == 4 ? KC4 / QK : KC6 / QK, left = qg_k / QK - qg_chunk * sbt;
  const int nsb = left < sbt ? left : sbt;  // super-blocks used in this tile (the last tile of a row may be short)
  acc_y = qg_fmt == 4 ? q4_tile(w, x, acc_y, nsb) : q6_tile(w, x, acc_y, nsb);
  if (++qg_chunk < qg_tiles) {
    aie::store_v(qg_acc, acc_y.to_vector<float>());
  } else if (!qg_glu) {  // block done: emit its 16 outputs
    aie::store_v(y, acc_y.to_vector<float>());
    aie::store_v(qg_acc, aie::zeros<float, RB>());
    qg_chunk = 0;
  } else {  // glu: keep a gate block, emit bf16(silu(gate) * up) after its up block
    if (qg_odd) {  // rounded to nearest even, as the host rounds x; the GEMV's own splits keep their mode
      const aie::rounding_mode mode = aie::get_rounding();
      aie::set_rounding(aie::rounding_mode::conv_even);
      aie::store_v((bfloat16 *)y, glu(aie::load_v<16>(qg_gate), acc_y.to_vector<float>()));
      aie::set_rounding(mode);
    } else
      aie::store_v(qg_gate, acc_y.to_vector<float>());
    qg_odd ^= 1;
    aie::store_v(qg_acc, aie::zeros<float, RB>());
    qg_chunk = 0;
  }
}
#endif

}  // extern "C"
