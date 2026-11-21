// W4A16 GEMM on npu1, the core side: turns one 64-column slice of GGUF Q4_K / Q6_K weights into the bf16 B
// operand of mlir-aie's matmul kernel (mm.cc), which multiplies it with a 64x64 bf16 activation tile.
//
// The weights are the decode GEMV's repacked tiles (qgemv.cc, xdna1_llm/q4k.py and q6k.py), read one super-block
// (256 columns of 16 rows) at a time: each B object holds two super-blocks of the same columns, for rows 0-15
// and 16-31 of the core's 32 outputs, SB_STRIDE bytes apart (a Q4_K super-block is shorter, the rest is ignored).
// One object serves four calls, for its 64-column slices in turn.
//
// Values: Q4_K w = d*sc*q - dmin*mn, Q6_K w = d*sc*(q - 32). Per group and row the kernel forms s = d*sc and
// the offset in fp32 (exact: d is the fp16 as bf16 high + low parts, sc a small integer), then for every value
// w = s * (128 + q) - (offset + 128 s), with 128 + q made exactly by the exponent trick (as in qgemv.cc) and s
// split into bf16 high + low parts, and rounds w to bf16 to nearest even: the same bf16 weight the CPU gets by
// rounding its fp32 dequantization, up to ties.
//
// Layout: a 64-byte load of nibbles holds 8 k x 16 rows; unpacked it gives four vectors of 32 bf16, each two k
// of 16 rows ([k0 | k2], [k1 | k3], [k4 | k6], [k5 | k7]). mm.cc reads B (64 k x 32 n) as 8 x 4 micro-tiles,
// row-major inside (lane = k * 4 + n) and ordered (k / 8, n / 4). Three rounds of interleaving 4-, 8- and
// 16-lane runs turn the four vectors into the four micro-tiles of those 8 k and 16 rows.
#include <aie_api/aie.hpp>
#include <stdint.h>

#define RB 16
#define SB_STRIDE 3648  // bytes per super-block in a B object (xdna1_llm.q4k.SB_BYTES[6])
#define N_CORE 32       // output columns (weight rows) per core
#define KS 64           // columns per slice

typedef aie::vector<bfloat16, 16> bf16x16;
typedef aie::vector<bfloat16, 32> bf16x32;
typedef aie::vector<float, 16> f32x16;
typedef aie::accum<accfloat, 16> facc16;

static int32_t slice;  // which 64-column slice of the current B object

// 64 bytes -> two vectors of 32 bf16 128 + q (qgemv.cc's to_bf16_zip)
static inline void to_bf16(aie::vector<uint8_t, 64> q, bf16x32 &a, bf16x32 &b) {
  const auto [lo, hi] = aie::interleave_zip(q, aie::broadcast<uint8_t, 64>(0x43), 1);
  a = aie::vector_cast<bfloat16>(lo);
  b = aie::vector_cast<bfloat16>(hi);
}

static inline void split(f32x16 v, bf16x16 &hi, bf16x16 &lo) {
  hi = facc16(v).to_vector<bfloat16>();
  facc16 h;
  h.from_vector(hi);
  lo = facc16(aie::sub(v, h.to_vector<float>())).to_vector<bfloat16>();
}

struct Scale {  // one group's per-row scale, for vectors of [16 rows | 16 rows]
  bf16x32 hi, lo;
  aie::vector<float, 32> neg_off;
};

static inline Scale scale(f32x16 s, f32x16 off) {
  Scale r;
  bf16x16 hi, lo;
  split(s, hi, lo);
  r.hi = aie::concat(hi, hi);
  r.lo = aie::concat(lo, lo);
  const f32x16 n = aie::neg(off);
  r.neg_off = aie::concat(n, n);
  return r;
}

static inline bf16x16 times(bf16x16 v, float p2) {  // v * a power of two: exact in bf16
  return aie::mul(v, (bfloat16)p2).to_vector<bfloat16>();
}

static inline bf16x32 apply(bf16x32 q, const Scale &s) {
  aie::accum<accfloat, 32> acc;
  acc.from_vector(s.neg_off);
  acc = aie::mac(acc, q, s.hi);
  acc = aie::mac(acc, q, s.lo);
  return acc.to_vector<bfloat16>();
}

// a = [k0 | k2], b = [k1 | k3], c = [k4 | k6], d = [k5 | k7] (16 rows each) -> the four 8 x 4 micro-tiles of
// rows 0-3, 4-7, 8-11, 12-15, stored at out
static inline void emit(bf16x32 a, bf16x32 b, bf16x32 c, bf16x32 d, const Scale &s, bfloat16 *__restrict out) {
  a = apply(a, s), b = apply(b, s), c = apply(c, s), d = apply(d, s);
  const auto [p, q] = aie::interleave_zip(a, b, 4);  // [k0 r0-3, k1 r0-3, k0 r4-7, ...], same for k2, k3
  const auto [u, v] = aie::interleave_zip(c, d, 4);
  const auto [e, f] = aie::interleave_zip(p, q, 8);  // [k0-3 r0-3, k0-3 r4-7], [k0-3 r8-11, k0-3 r12-15]
  const auto [g, h] = aie::interleave_zip(u, v, 8);
  const auto [m0, m1] = aie::interleave_zip(e, g, 16);  // [k0-7 r0-3], [k0-7 r4-7]
  const auto [m2, m3] = aie::interleave_zip(f, h, 16);
  aie::store_v(out, m0);
  aie::store_v(out + 32, m1);
  aie::store_v(out + 64, m2);
  aie::store_v(out + 96, m3);
}

// one super-block of 16 rows, the 64 columns of slice sl -> out (micro-tiles of these rows in a 64 x 32 B)
static inline void q4_slice(const uint8_t *__restrict wb, int sl, bfloat16 *__restrict out) {
  const bfloat16 *__restrict scm = (const bfloat16 *)(wb + 128 * RB);
  const bfloat16 *__restrict dd = scm + 8 * 2 * RB;
  const bf16x16 d_hi = aie::load_v<16>(dd), d_lo = aie::load_v<16>(dd + RB);
  const bf16x16 dm_hi = aie::load_v<16>(dd + 2 * RB), dm_lo = aie::load_v<16>(dd + 3 * RB);
  const aie::vector<uint8_t, 64> m0f = aie::broadcast<uint8_t, 64>(0x0F);
  for (int gi = 0; gi < 2; gi++) {  // two groups of 32 columns
    const int g = sl * 2 + gi;
    const bf16x16 sc = aie::load_v<16>(scm + g * 2 * RB), mn = aie::load_v<16>(scm + g * 2 * RB + RB);
    const f32x16 s = aie::mac(aie::mul(d_hi, sc), d_lo, sc).to_vector<float>();
    const bf16x16 sc128 = times(sc, 128.0f);  // offset dmin * mn + 128 s
    const facc16 off = aie::mac(aie::mac(aie::mac(aie::mul(dm_hi, mn), dm_lo, mn), d_hi, sc128), d_lo, sc128);
    const Scale sv = scale(s, off.to_vector<float>());
    for (int i = 0; i < 4; i++) {  // 8 columns each
      const aie::vector<uint8_t, 64> v = aie::load_v<64>(wb + (g * 4 + i) * 4 * RB);
      bf16x32 a, b, c, d;
      to_bf16(aie::bit_and(v, m0f), a, c);
      to_bf16(aie::downshift(v, 4), b, d);
      emit(a, b, c, d, sv, out + (gi * 4 + i) * 8 * 32);
    }
  }
}

static inline void q6_slice(const uint8_t *__restrict wb, int sl, bfloat16 *__restrict out) {
  const uint8_t *__restrict hbits = wb + 128 * RB;
  const bfloat16 *__restrict scv = (const bfloat16 *)(hbits + 16 * 64);
  const bfloat16 *__restrict dd = scv + 16 * RB;
  const bf16x16 d_hi = aie::load_v<16>(dd), d_lo = aie::load_v<16>(dd + RB);
  const aie::vector<uint8_t, 64> m0f = aie::broadcast<uint8_t, 64>(0x0F), m30 = aie::broadcast<uint8_t, 64>(0x30);
  for (int gi = 0; gi < 4; gi++) {  // four groups of 16 columns
    const int g = sl * 4 + gi;
    const bf16x16 sc = aie::load_v<16>(scv + g * RB);
    const f32x16 s = aie::mac(aie::mul(d_hi, sc), d_lo, sc).to_vector<float>();
    const bf16x16 sc128 = times(sc, 128.0f), sc32 = times(sc, 32.0f);  // offset 160 s
    const facc16 off = aie::mac(aie::mac(aie::mac(aie::mul(d_hi, sc128), d_lo, sc128), d_hi, sc32), d_lo, sc32);
    const Scale sv = scale(s, off.to_vector<float>());
    const aie::vector<uint8_t, 64> n0 = aie::load_v<64>(wb + g * 8 * RB), n1 = aie::load_v<64>(wb + g * 8 * RB + 64);
    const aie::vector<uint8_t, 64> h = aie::load_v<64>(hbits + g * 64);
    bf16x32 a, b, c, d;
    to_bf16(aie::bit_or(aie::bit_and(n0, m0f), aie::bit_and(aie::upshift(h, 4), m30)), a, c);
    to_bf16(aie::bit_or(aie::downshift(n0, 4), aie::bit_and(aie::upshift(h, 2), m30)), b, d);
    emit(a, b, c, d, sv, out + (gi * 2) * 8 * 32);
    to_bf16(aie::bit_or(aie::bit_and(n1, m0f), aie::bit_and(h, m30)), a, c);
    to_bf16(aie::bit_or(aie::downshift(n1, 4), aie::bit_and(aie::downshift(h, 2), m30)), b, d);
    emit(a, b, c, d, sv, out + (gi * 2 + 1) * 8 * 32);
  }
}

extern "C" {

// b: the B object (two super-blocks); out: B for mm.cc, 64 x 32 bf16 in micro-tiles; rtp[2]: format (4 or 6)
void qm_dequant(const uint8_t *__restrict b, bfloat16 *__restrict out, const int32_t *__restrict rtp) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int fmt = rtp[2];
  for (int j = 0; j < 2; j++) {  // rows 0-15, 16-31: micro-tile columns 4j .. 4j + 3
    if (fmt == 4)
      q4_slice(b + j * SB_STRIDE, slice, out + j * 4 * 32);
    else
      q6_slice(b + j * SB_STRIDE, slice, out + j * 4 * 32);
  }
  slice = (slice + 1) & 3;
}

}  // extern "C"
