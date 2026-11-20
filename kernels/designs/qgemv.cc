// Quantized GEMV for one decode token on an npu1 core: y = W . x (fp32 out, bf16 x) with W in GGUF Q4_K or Q6_K.
// The format, the row length K and the sizes arrive at run time, so one core program serves every LFM3 matrix and
// decode runs in a single hardware context (switching contexts costs ~1.45 ms per call on the Windows driver).
//
//   Q4_K: W[r, k] = d[r, b] * sc[r, g] * q[r, k] - dmin[r, b] * mn[r, g]   q in 0..15, g = k / 32, b = k / 256
//   Q6_K: W[r, k] = d[r, b] * sc[r, g] * (q[r, k] - 32)                   q in 0..63, g = k / 16
// Both are the formats' own math, so results match llama.cpp's dequantization to fp32 rounding.
//
// Every weight tile is TILE_BYTES and covers 16 rows (vector lanes = rows) and KC4 (Q4_K) or KC6 (Q6_K) columns;
// the two sizes are chosen so the tiles are (nearly) the same size. A row of K columns takes ceil(K / KC) tiles;
// the last one may be partly used. Inside a tile (xdna1_llm/q4k.py and q6k.py write this):
//   Q4_K: nibbles   KC4/2 x 16 B: chunk p holds q[i, 2p] | q[i, 2p+1] << 4 for rows i = 0..15
//         groups    per 32 k: sc[16], mn[16] (bf16, small integers, exact)
//         blocks    per 256 k: d_hi[16], d_lo[16], dmin_hi[16], dmin_lo[16] (bf16, hi + lo == the fp16 exactly)
//   Q6_K: nibbles   KC6/2 x 16 B: the low 4 bits, as above
//         high bits KC6/8 x 32 B: byte i (row i % 16, side s = i / 16) holds the high 2 bits of k = 2s, 2s+1,
//                   2s+4, 2s+5 (relative to the 8 k) in bits 0-1, 2-3, 4-5, 6-7: the lane order of the nibble
//                   vectors below, so no shuffles are needed
//         groups    per 16 k: sc[16] (bf16)
//         blocks    per 256 k: d_hi[16], d_lo[16]
// x arrives first, over two tiles: x as bf16 from byte 0, and a header in the last 16 bytes of the second tile:
// [16-row blocks for this lane, tiles per block, format (4 or 6), K].
//
// Quantized values become bf16 exactly with the exponent trick (0x4300 | q) == 128 + q; the offset (128, plus 32
// for Q6_K) times sum(x) is removed per group. AIE2 has no fast fp32 multiply, so every scaling step is a native
// bf16 MAC on fp32 values split into bf16 high and low parts.
//
// IRON compiles one object per ExternalFunction, so each function is built on its own (QG_FN) and they share
// state through globals.
#include <aie_api/aie.hpp>
#include <stdint.h>

#define TILE_BYTES 10944
#define K_MAX 7680
#define RB 16
#define QK 256
#define KC4 1024  // Q4_K: 4 super-blocks per tile, 10752 bytes used
#define KC6 768   // Q6_K: 3 super-blocks per tile, 10944 bytes used

extern "C" {

#if QG_FN == 0
int32_t qg_fmt, qg_k, qg_tiles, qg_chunk;  // format, row length, tiles per block, next tile of the current block
float qg_acc[RB];                          // partial outputs of the current block
float qg_xoff[K_MAX / 16];                 // per group: (128 or 160) * sum(x)
bfloat16 qg_xs_hi[K_MAX / 32], qg_xs_lo[K_MAX / 32];  // Q4_K, per group: sum(x) as bf16 high + low parts
static int32_t xpart;

void qg_load_x(const uint8_t *__restrict src, bfloat16 *__restrict x, int32_t *__restrict hdr) {
  const bfloat16 *s = (const bfloat16 *)src;
  const int begin = xpart * (TILE_BYTES / 2), end = begin + TILE_BYTES / 2 < K_MAX ? begin + TILE_BYTES / 2 : K_MAX;
  for (int i = begin; i < end; i += 32)
    aie::store_v(x + i, aie::load_v<32>(s + (i - begin)));
  if (++xpart < 2)
    return;
  xpart = 0;
  const int32_t *h = (const int32_t *)(src + TILE_BYTES - 16);
  hdr[0] = h[0];
  hdr[1] = h[1];
  qg_tiles = h[1];
  qg_fmt = h[2];
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

// 32 quantized values (bytes) -> bf16 128 + q
static inline aie::vector<bfloat16, 32> to_bf16(aie::vector<uint8_t, 32> q) {
  return aie::vector_cast<bfloat16>(aie::bit_or(aie::unpack(q), aie::broadcast<uint16_t, 32>(0x4300)));
}

// lo: lanes 0-15 rows at k = kk, lanes 16-31 at kk + 2; hi: kk + 1 and kk + 3. mac_elem_16_2 adds lanes i and
// i + 16 into accumulator lane i, so one instruction does 32 multiply-adds.
extern "C++" {  // templates need C++ linkage
template <unsigned N>
static inline v16accfloat quad(v16accfloat a, aie::vector<uint8_t, 32> lo, aie::vector<uint8_t, 32> hi,
                               const aie::vector<bfloat16, N> &xv, int kk) {
  aie::vector<bfloat16, 32> x02 = aie::concat(aie::broadcast<bfloat16, 16>(xv[kk]), aie::broadcast<bfloat16, 16>(xv[kk + 2]));
  aie::vector<bfloat16, 32> x13 = aie::concat(aie::broadcast<bfloat16, 16>(xv[kk + 1]), aie::broadcast<bfloat16, 16>(xv[kk + 3]));
  a = mac_elem_16_2(to_bf16(lo).to_native(), x02.to_native(), a);
  return mac_elem_16_2(to_bf16(hi).to_native(), x13.to_native(), a);
}
}  // extern "C++"

static inline accf16_t q4_tile(const uint8_t *__restrict w, const bfloat16 *__restrict x, accf16_t acc_y, int nsb) {
  const bfloat16 *__restrict scm = (const bfloat16 *)(w + KC4 / 2 * RB);
  const bfloat16 *__restrict dd = scm + KC4 / 32 * 2 * RB;
  const aie::vector<uint8_t, 32> m0f = aie::broadcast<uint8_t, 32>(0x0F);
  const int g0 = qg_chunk * (KC4 / 32);
  for (int b = 0; b < nsb; b++) {
    accf16_t acc_s = aie::zeros<accfloat, 16>(), acc_m = acc_s;  // sum of sc * dot, mn * xsum
    for (int gb = 0; gb < 8; gb++) {
      const int gc = b * 8 + gb, g = g0 + gc;
      const uint8_t *__restrict wg = w + gc * 16 * RB;
      const aie::vector<bfloat16, 32> xg = aie::load_v<32>(x + g * 32);
      v16accfloat a = aie::zeros<accfloat, 16>().to_native();
      _Pragma("clang loop unroll(full)")
      for (int kk = 0; kk < 32; kk += 4) {
        aie::vector<uint8_t, 32> v = aie::load_v<32>(wg + kk / 2 * RB);
        a = quad(a, aie::bit_and(v, m0f), aie::downshift(v, 4), xg, kk);
      }
      vbf16_t dot_hi, dot_lo;
      split(aie::sub(accf16_t(a).to_vector<float>(), qg_xoff[g]), dot_hi, dot_lo);
      vbf16_t sc = aie::load_v<16>(scm + gc * 2 * RB), mn = aie::load_v<16>(scm + gc * 2 * RB + RB);
      acc_s = aie::mac(aie::mac(acc_s, sc, dot_hi), sc, dot_lo);
      acc_m = aie::mac(aie::mac(acc_m, mn, qg_xs_hi[g]), mn, qg_xs_lo[g]);
    }
    // y += d * S - dmin * M (the lo * lo terms are below fp32 rounding)
    const bfloat16 *__restrict db = dd + b * 4 * RB;
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
  const uint8_t *__restrict hbits = w + KC6 / 2 * RB;
  const bfloat16 *__restrict scv = (const bfloat16 *)(hbits + KC6 / 8 * 32);
  const bfloat16 *__restrict dd = scv + KC6 / 16 * RB;
  const aie::vector<uint8_t, 32> m0f = aie::broadcast<uint8_t, 32>(0x0F), m30 = aie::broadcast<uint8_t, 32>(0x30);
  const int g0 = qg_chunk * (KC6 / 16);
  for (int b = 0; b < nsb; b++) {
    accf16_t acc_s = aie::zeros<accfloat, 16>();  // sum of sc * dot
    for (int gb = 0; gb < 16; gb++) {
      const int gc = b * 16 + gb, g = g0 + gc;
      const aie::vector<bfloat16, 16> xg = aie::load_v<16>(x + g * 16);
      v16accfloat a = aie::zeros<accfloat, 16>().to_native();
      _Pragma("clang loop unroll(full)")
      for (int k8 = 0; k8 < 16; k8 += 8) {
        const int kc = gc * 16 + k8;  // k within the tile
        aie::vector<uint8_t, 32> hv = aie::load_v<32>(hbits + kc / 8 * 32);
        aie::vector<uint8_t, 32> v0 = aie::load_v<32>(w + kc / 2 * RB), v1 = aie::load_v<32>(w + (kc + 4) / 2 * RB);
        a = quad(a, aie::bit_or(aie::bit_and(v0, m0f), aie::bit_and(aie::upshift(hv, 4), m30)),
                 aie::bit_or(aie::downshift(v0, 4), aie::bit_and(aie::upshift(hv, 2), m30)), xg, k8);
        a = quad(a, aie::bit_or(aie::bit_and(v1, m0f), aie::bit_and(hv, m30)),
                 aie::bit_or(aie::downshift(v1, 4), aie::bit_and(aie::downshift(hv, 2), m30)), xg, k8 + 4);
      }
      vbf16_t dot_hi, dot_lo;
      split(aie::sub(accf16_t(a).to_vector<float>(), qg_xoff[g]), dot_hi, dot_lo);
      vbf16_t sc = aie::load_v<16>(scv + gc * RB);
      acc_s = aie::mac(aie::mac(acc_s, sc, dot_hi), sc, dot_lo);
    }
    vbf16_t d_hi = aie::load_v<16>(dd + b * 2 * RB), d_lo = aie::load_v<16>(dd + b * 2 * RB + RB);
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
  } else {  // block done: emit its 16 outputs
    aie::store_v(y, acc_y.to_vector<float>());
    aie::store_v(qg_acc, aie::zeros<float, RB>());
    qg_chunk = 0;
  }
}
#endif

}  // extern "C"
