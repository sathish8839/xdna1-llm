// 4-bit GEMV for one decode token on an npu1 core: y = W . x with W[r, k] = s[r, g] * q[r, k] - m[r, g],
// q in 0..15, one (s, m) pair per row per group of 32 k (g = k / 32), as in GGUF Q4_K after expanding its
// 6-bit scales. fp32 output. Compile-time: Q4_K (row length).
//
// A weight tile covers 16 rows (vector lanes = rows) and all K:
//   nibbles: K/2 chunks of 16 bytes; chunk p holds q[i, 2p] | q[i, 2p+1] << 4 for rows i = 0..15
//   scales:  per group, s[16] then m[16] as bf16
// q is turned into bf16 exactly with the exponent trick (0x4300 | q) == 128 + q, accumulated in fp32, and the
// 128 * sum(x) offset is removed per group, so no integer-to-float conversion is needed.
#include <aie_api/aie.hpp>
#include <stdint.h>

#define RB 16
#define G 32
#define NG (Q4_K / G)
#define NIB_BYTES (Q4_K / 2 * RB)

extern "C" {

#if Q4_FN == 0
int32_t q4_pos;    // next output row
float q4_xsum[NG];        // 128 * sum of x over each group
bfloat16 q4_xs_hi[NG], q4_xs_lo[NG];  // sum of x, as bf16 high + low parts

void q4_load_x(const uint8_t *__restrict src, bfloat16 *__restrict x, int32_t *__restrict tiles) {
  const bfloat16 *xs = (const bfloat16 *)src;
  for (int i = 0; i < Q4_K; i += 32)
    aie::store_v(x + i, aie::load_v<32>(xs + i));
  for (int g = 0; g < NG; g++) {
    aie::accum<accfloat, 32> a;
    a.from_vector(aie::load_v<32>(xs + g * G));
    float xs = aie::reduce_add(a.to_vector<float>());
    q4_xsum[g] = 128.0f * xs;
    q4_xs_hi[g] = (bfloat16)xs;
    q4_xs_lo[g] = (bfloat16)(xs - (float)q4_xs_hi[g]);
  }
  tiles[0] = *(const int32_t *)(src + 2 * Q4_K);
  q4_pos = 0;
}
#else
extern int32_t q4_pos;
extern float q4_xsum[NG];
extern bfloat16 q4_xs_hi[NG], q4_xs_lo[NG];

void q4_rows(const uint8_t *__restrict w, const bfloat16 *__restrict x, float *__restrict y) {
  const bfloat16 *__restrict sm = (const bfloat16 *)(w + NIB_BYTES);
  const aie::vector<uint8_t, 32> mask = aie::broadcast<uint8_t, 32>(0x0F);
  const aie::vector<uint16_t, 32> magic = aie::broadcast<uint16_t, 32>(0x4300);
  aie::accum<accfloat, 16> acc_y = aie::zeros<accfloat, 16>();
  for (int g = 0; g < NG; g++) {
    // mac_elem_16_2 adds lanes i and i + 16 into accumulator lane i: one instruction covers rows at k and k + 2
    // (low nibbles) or k + 1 and k + 3 (high nibbles), 32 multiply-adds.
    v16accfloat a = aie::zeros<accfloat, 16>().to_native();
    const uint8_t *__restrict wg = w + g * (G / 2) * RB;
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
    // y += s * (acc - 128 * xsum) - m * xsum. AIE2 emulates fp32 multiplies, so both products use native bf16
    // MACs with the fp32 factor split into a bf16 high and low part (the two terms cancel, so plain bf16 would
    // not be accurate enough).
    aie::vector<float, 16> dot = aie::sub(aie::accum<accfloat, 16>(a).to_vector<float>(), q4_xsum[g]);
    aie::vector<bfloat16, 16> dot_hi = aie::accum<accfloat, 16>(dot).to_vector<bfloat16>();
    aie::accum<accfloat, 16> hi_f;
    hi_f.from_vector(dot_hi);
    aie::vector<bfloat16, 16> dot_lo = aie::accum<accfloat, 16>(aie::sub(dot, hi_f.to_vector<float>())).to_vector<bfloat16>();
    aie::vector<bfloat16, 16> s = aie::load_v<16>(sm + g * 2 * RB), m = aie::load_v<16>(sm + g * 2 * RB + RB);
    acc_y = aie::mac(acc_y, s, dot_hi);
    acc_y = aie::mac(acc_y, s, dot_lo);
    acc_y = aie::msc(acc_y, m, q4_xs_hi[g]);
    acc_y = aie::msc(acc_y, m, q4_xs_lo[g]);
  }
  aie::store_v(y + q4_pos, acc_y.to_vector<float>());
  q4_pos += RB;
}
#endif

}  // extern "C"
