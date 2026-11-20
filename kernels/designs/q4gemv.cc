// 4-bit GEMV for one decode token on an npu1 core, on GGUF Q4_K weights: y = W . x, fp32 output, with
//   W[r, k] = d[r, b] * sc[r, g] * q[r, k] - dmin[r, b] * mn[r, g]
// q in 0..15, sc and mn 6-bit integers per group of 32 k (g = k / 32), d and dmin fp16 per super-block of 256 k
// (b = k / 256). This is Q4_K's own math, so results match llama.cpp's dequantization to fp32 rounding.
//
// Compile-time: Q4_K (row length), Q4_KC (k per weight tile, a multiple of 256; K is split into Q4_K / Q4_KC
// chunks), Q4_TB (tile bytes), Q4_XT (tiles that carry x).
//
// A weight tile covers 16 rows (vector lanes = rows) and Q4_KC columns:
//   nibbles:      Q4_KC/2 chunks of 16 bytes; chunk p holds q[i, 2p] | q[i, 2p+1] << 4 for rows i = 0..15
//   groups:       per group, sc[16] then mn[16] as bf16 (small integers, exact)
//   super-blocks: per 256 k, d_hi[16], d_lo[16], dmin_hi[16], dmin_lo[16] as bf16 (hi + lo == the fp16 exactly)
// x arrives first, over Q4_XT tiles (x as bf16, then zero padding; the last 4 bytes of the last tile hold the
// number of 16-row blocks this lane will receive).
//
// q is turned into bf16 exactly with the exponent trick (0x4300 | q) == 128 + q, accumulated in fp32, and the
// 128 * sum(x) offset is removed per group, so no integer-to-float conversion is needed. AIE2 has no fast fp32
// multiply, so every product is a native bf16 MAC with fp32 values split into bf16 high and low parts.
//
// IRON compiles one object per ExternalFunction, so each function is built on its own (Q4_FN) and they share
// state through globals.
#include <aie_api/aie.hpp>
#include <stdint.h>

#define RB 16
#define G 32
#define SB 8                // groups per super-block
#define NG (Q4_K / G)       // groups in a row
#define NGC (Q4_KC / G)     // groups in a tile
#define NSC (NGC / SB)      // super-blocks in a tile
#define NC (Q4_K / Q4_KC)   // tiles per 16-row block
#define NIB_BYTES (Q4_KC / 2 * RB)

extern "C" {

#if Q4_FN == 0
float q4_xsum128[NG];                 // 128 * sum of x over each group
bfloat16 q4_xs_hi[NG], q4_xs_lo[NG];  // sum of x over each group, as bf16 high + low parts
int32_t q4_chunk;                     // next k chunk of the current block
float q4_acc[RB];                     // partial outputs of the current block
static int32_t xpart;

void q4_load_x(const uint8_t *__restrict src, bfloat16 *__restrict x, int32_t *__restrict blocks) {
  const int begin = xpart * (Q4_TB / 2);  // in bf16 elements
  const int end = begin + Q4_TB / 2 < Q4_K ? begin + Q4_TB / 2 : Q4_K;
  const bfloat16 *s = (const bfloat16 *)src;
  for (int i = begin; i < end; i += 32)
    aie::store_v(x + i, aie::load_v<32>(s + (i - begin)));
  if (++xpart < Q4_XT)
    return;
  for (int g = 0; g < NG; g++) {
    aie::accum<accfloat, 32> a;
    a.from_vector(aie::load_v<32>(x + g * G));
    float xs = aie::reduce_add(a.to_vector<float>());
    q4_xsum128[g] = 128.0f * xs;
    q4_xs_hi[g] = (bfloat16)xs;
    q4_xs_lo[g] = (bfloat16)(xs - (float)q4_xs_hi[g]);
  }
  blocks[0] = *(const int32_t *)(src + Q4_TB - 4);
  xpart = 0;
  q4_chunk = 0;
  aie::store_v(q4_acc, aie::zeros<float, RB>());
}
#else
extern float q4_xsum128[NG];
extern bfloat16 q4_xs_hi[NG], q4_xs_lo[NG];
extern int32_t q4_chunk;
extern float q4_acc[RB];

static inline void split(aie::vector<float, 16> v, aie::vector<bfloat16, 16> &hi, aie::vector<bfloat16, 16> &lo) {
  hi = aie::accum<accfloat, 16>(v).to_vector<bfloat16>();
  aie::accum<accfloat, 16> h;
  h.from_vector(hi);
  lo = aie::accum<accfloat, 16>(aie::sub(v, h.to_vector<float>())).to_vector<bfloat16>();
}

void q4_rows(const uint8_t *__restrict w, const bfloat16 *__restrict x, float *__restrict y) {
  const bfloat16 *__restrict scm = (const bfloat16 *)(w + NIB_BYTES);
  const bfloat16 *__restrict dd = scm + NGC * 2 * RB;
  const aie::vector<uint8_t, 32> mask = aie::broadcast<uint8_t, 32>(0x0F);
  const aie::vector<uint16_t, 32> magic = aie::broadcast<uint16_t, 32>(0x4300);
  const int g0 = q4_chunk * NGC;
  aie::accum<accfloat, 16> acc_y;
  acc_y.from_vector(aie::load_v<16>(q4_acc));
  for (int b = 0; b < NSC; b++) {
    aie::accum<accfloat, 16> acc_s = aie::zeros<accfloat, 16>(), acc_m = acc_s;  // sum of sc * dot, mn * xsum
    for (int gb = 0; gb < SB; gb++) {
      const int gc = b * SB + gb, g = g0 + gc;
      // mac_elem_16_2 adds lanes i and i + 16 into accumulator lane i: one instruction covers rows at k and
      // k + 2 (low nibbles) or k + 1 and k + 3 (high nibbles), 32 multiply-adds.
      v16accfloat a = aie::zeros<accfloat, 16>().to_native();
      const uint8_t *__restrict wg = w + gc * (G / 2) * RB;
      const aie::vector<bfloat16, 32> xv = aie::load_v<32>(x + g * G);
      _Pragma("clang loop unroll(full)")
      for (int kk = 0; kk < G; kk += 4) {
        aie::vector<uint8_t, 32> v = aie::load_v<32>(wg + kk / 2 * RB);
        aie::vector<bfloat16, 32> lb = aie::vector_cast<bfloat16>(aie::bit_or(aie::unpack(aie::bit_and(v, mask)), magic));
        aie::vector<bfloat16, 32> hb = aie::vector_cast<bfloat16>(aie::bit_or(aie::unpack(aie::downshift(v, 4)), magic));
        aie::vector<bfloat16, 32> x02 = aie::concat(aie::broadcast<bfloat16, 16>(xv[kk]), aie::broadcast<bfloat16, 16>(xv[kk + 2]));
        aie::vector<bfloat16, 32> x13 = aie::concat(aie::broadcast<bfloat16, 16>(xv[kk + 1]), aie::broadcast<bfloat16, 16>(xv[kk + 3]));
        a = mac_elem_16_2(lb.to_native(), x02.to_native(), a);
        a = mac_elem_16_2(hb.to_native(), x13.to_native(), a);
      }
      aie::vector<bfloat16, 16> dot_hi, dot_lo;
      split(aie::sub(aie::accum<accfloat, 16>(a).to_vector<float>(), q4_xsum128[g]), dot_hi, dot_lo);
      aie::vector<bfloat16, 16> sc = aie::load_v<16>(scm + gc * 2 * RB), mn = aie::load_v<16>(scm + gc * 2 * RB + RB);
      acc_s = aie::mac(aie::mac(acc_s, sc, dot_hi), sc, dot_lo);
      acc_m = aie::mac(aie::mac(acc_m, mn, q4_xs_hi[g]), mn, q4_xs_lo[g]);
    }
    // y += d * S - dmin * M (the lo * lo terms are below fp32 rounding)
    const bfloat16 *__restrict db = dd + b * 4 * RB;
    aie::vector<bfloat16, 16> d_hi = aie::load_v<16>(db), d_lo = aie::load_v<16>(db + RB);
    aie::vector<bfloat16, 16> dm_hi = aie::load_v<16>(db + 2 * RB), dm_lo = aie::load_v<16>(db + 3 * RB);
    aie::vector<bfloat16, 16> s_hi, s_lo, m_hi, m_lo;
    split(acc_s.to_vector<float>(), s_hi, s_lo);
    split(acc_m.to_vector<float>(), m_hi, m_lo);
    acc_y = aie::mac(aie::mac(aie::mac(acc_y, d_hi, s_hi), d_hi, s_lo), d_lo, s_hi);
    acc_y = aie::msc(aie::msc(aie::msc(acc_y, dm_hi, m_hi), dm_hi, m_lo), dm_lo, m_hi);
  }
  if (++q4_chunk < NC) {
    aie::store_v(q4_acc, acc_y.to_vector<float>());
  } else {  // block done: emit its 16 outputs
    aie::store_v(y, acc_y.to_vector<float>());
    aie::store_v(q4_acc, aie::zeros<float, RB>());
    q4_chunk = 0;
  }
}
#endif

}  // extern "C"
