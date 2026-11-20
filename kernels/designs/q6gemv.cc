// 6-bit GEMV for one decode token on an npu1 core, on GGUF Q6_K weights: y = W . x, fp32 output, with
//   W[r, k] = d[r, b] * sc[r, g] * (q[r, k] - 32)
// q in 0..63, sc an int8 per group of 16 k (g = k / 16), d fp16 per super-block of 256 k (b = k / 256). This is
// Q6_K's own math, so results match llama.cpp's dequantization to fp32 rounding.
//
// Compile-time: Q6_K (row length), Q6_KC (k per weight tile, a multiple of 256), Q6_TB (tile bytes),
// Q6_XT (tiles that carry x). The surrounding design (lanes, x, chunking, output streaming) is the same as the
// 4-bit kernel; see q4gemv.cc and qgemv.py.
//
// A weight tile covers 16 rows (vector lanes = rows) and Q6_KC columns:
//   low nibbles: Q6_KC/2 chunks of 16 bytes; chunk p holds q[i, 2p] & 15 | (q[i, 2p+1] & 15) << 4, rows 0..15
//   high bits:   Q6_KC/8 chunks of 32 bytes; byte i (row i % 16, side s = i / 16) holds the high 2 bits of
//                k = 2s, 2s+1, 2s+4, 2s+5 (relative to the chunk) in bits 0-1, 2-3, 4-5, 6-7. That is exactly
//                the lane order of the low/high-nibble vectors below, so no lane shuffles are needed.
//   scales:      per group, sc[16] as bf16 (small integers, exact)
//   super-block: per 256 k, d_hi[16], d_lo[16] as bf16 (hi + lo == the fp16 exactly)
//
// 6-bit values become bf16 exactly with the exponent trick (0x4300 | q) == 128 + q; the 160 * sum(x) offset
// (128 from the trick, 32 from Q6_K) is removed per group.
#include <aie_api/aie.hpp>
#include <stdint.h>

#define RB 16
#define G 16
#define SB 16               // groups per super-block
#define NG (Q6_K / G)       // groups in a row
#define NGC (Q6_KC / G)     // groups in a tile
#define NSC (NGC / SB)      // super-blocks in a tile
#define NC (Q6_K / Q6_KC)   // tiles per 16-row block
#define NIB_BYTES (Q6_KC / 2 * RB)
#define HI_BYTES (Q6_KC / 8 * 32)

extern "C" {

#if Q6_FN == 0
float q6_xsum160[NG];  // 160 * sum of x over each group
int32_t q6_chunk;      // next k chunk of the current block
float q6_acc[RB];      // partial outputs of the current block
static int32_t xpart;

void q6_load_x(const uint8_t *__restrict src, bfloat16 *__restrict x, int32_t *__restrict blocks) {
  const int begin = xpart * (Q6_TB / 2);  // in bf16 elements
  const int end = begin + Q6_TB / 2 < Q6_K ? begin + Q6_TB / 2 : Q6_K;
  const bfloat16 *s = (const bfloat16 *)src;
  for (int i = begin; i < end; i += 32)
    aie::store_v(x + i, aie::load_v<32>(s + (i - begin)));
  if (++xpart < Q6_XT)
    return;
  for (int g = 0; g < NG; g++) {
    aie::accum<accfloat, 16> a;
    a.from_vector(aie::load_v<16>(x + g * G));
    q6_xsum160[g] = 160.0f * aie::reduce_add(a.to_vector<float>());
  }
  blocks[0] = *(const int32_t *)(src + Q6_TB - 4);
  xpart = 0;
  q6_chunk = 0;
  aie::store_v(q6_acc, aie::zeros<float, RB>());
}
#else
extern float q6_xsum160[NG];
extern int32_t q6_chunk;
extern float q6_acc[RB];

static inline void split(aie::vector<float, 16> v, aie::vector<bfloat16, 16> &hi, aie::vector<bfloat16, 16> &lo) {
  hi = aie::accum<accfloat, 16>(v).to_vector<bfloat16>();
  aie::accum<accfloat, 16> h;
  h.from_vector(hi);
  lo = aie::accum<accfloat, 16>(aie::sub(v, h.to_vector<float>())).to_vector<bfloat16>();
}

// 32 bytes of 6-bit values (already merged) -> bf16 128 + q
static inline aie::vector<bfloat16, 32> to_bf16(aie::vector<uint8_t, 32> q) {
  return aie::vector_cast<bfloat16>(aie::bit_or(aie::unpack(q), aie::broadcast<uint16_t, 32>(0x4300)));
}

static inline v16accfloat quad(v16accfloat a, aie::vector<uint8_t, 32> lo, aie::vector<uint8_t, 32> hi,
                               const aie::vector<bfloat16, 16> &xv, int kk) {
  // lanes 0-15: rows at k = kk (lo) / kk + 1 (hi); lanes 16-31: k = kk + 2 / kk + 3. mac_elem_16_2 adds lane i
  // and i + 16 into accumulator lane i.
  aie::vector<bfloat16, 32> x02 = aie::concat(aie::broadcast<bfloat16, 16>(xv[kk]), aie::broadcast<bfloat16, 16>(xv[kk + 2]));
  aie::vector<bfloat16, 32> x13 = aie::concat(aie::broadcast<bfloat16, 16>(xv[kk + 1]), aie::broadcast<bfloat16, 16>(xv[kk + 3]));
  a = mac_elem_16_2(to_bf16(lo).to_native(), x02.to_native(), a);
  return mac_elem_16_2(to_bf16(hi).to_native(), x13.to_native(), a);
}

void q6_rows(const uint8_t *__restrict w, const bfloat16 *__restrict x, float *__restrict y) {
  const uint8_t *__restrict hbits = w + NIB_BYTES;
  const bfloat16 *__restrict scv = (const bfloat16 *)(w + NIB_BYTES + HI_BYTES);
  const bfloat16 *__restrict dd = scv + NGC * RB;
  const aie::vector<uint8_t, 32> m0f = aie::broadcast<uint8_t, 32>(0x0F), m30 = aie::broadcast<uint8_t, 32>(0x30);
  const int g0 = q6_chunk * NGC;
  aie::accum<accfloat, 16> acc_y;
  acc_y.from_vector(aie::load_v<16>(q6_acc));
  for (int b = 0; b < NSC; b++) {
    aie::accum<accfloat, 16> acc_s = aie::zeros<accfloat, 16>();  // sum of sc * dot
    for (int gb = 0; gb < SB; gb++) {
      const int gc = b * SB + gb, g = g0 + gc;
      v16accfloat a = aie::zeros<accfloat, 16>().to_native();
      const aie::vector<bfloat16, 16> xv = aie::load_v<16>(x + g * G);
      _Pragma("clang loop unroll(full)")
      for (int k8 = 0; k8 < G; k8 += 8) {
        const int kc = gc * G + k8;  // k within the tile
        aie::vector<uint8_t, 32> hv = aie::load_v<32>(hbits + kc / 8 * 32);
        aie::vector<uint8_t, 32> v0 = aie::load_v<32>(w + kc / 2 * RB), v1 = aie::load_v<32>(w + (kc + 4) / 2 * RB);
        a = quad(a, aie::bit_or(aie::bit_and(v0, m0f), aie::bit_and(aie::upshift(hv, 4), m30)),
                 aie::bit_or(aie::downshift(v0, 4), aie::bit_and(aie::upshift(hv, 2), m30)), xv, k8);
        a = quad(a, aie::bit_or(aie::bit_and(v1, m0f), aie::bit_and(hv, m30)),
                 aie::bit_or(aie::downshift(v1, 4), aie::bit_and(aie::downshift(hv, 2), m30)), xv, k8 + 4);
      }
      aie::vector<bfloat16, 16> dot_hi, dot_lo;
      split(aie::sub(aie::accum<accfloat, 16>(a).to_vector<float>(), q6_xsum160[g]), dot_hi, dot_lo);
      aie::vector<bfloat16, 16> sc = aie::load_v<16>(scv + gc * RB);
      acc_s = aie::mac(aie::mac(acc_s, sc, dot_hi), sc, dot_lo);
    }
    // y += d * S (the lo * lo term is below fp32 rounding)
    aie::vector<bfloat16, 16> d_hi = aie::load_v<16>(dd + b * 2 * RB), d_lo = aie::load_v<16>(dd + b * 2 * RB + RB);
    aie::vector<bfloat16, 16> s_hi, s_lo;
    split(acc_s.to_vector<float>(), s_hi, s_lo);
    acc_y = aie::mac(aie::mac(aie::mac(acc_y, d_hi, s_hi), d_hi, s_lo), d_lo, s_hi);
  }
  if (++q6_chunk < NC) {
    aie::store_v(q6_acc, acc_y.to_vector<float>());
  } else {  // block done: emit its 16 outputs
    aie::store_v(y, acc_y.to_vector<float>());
    aie::store_v(q6_acc, aie::zeros<float, RB>());
    q6_chunk = 0;
  }
}
#endif

}  // extern "C"
