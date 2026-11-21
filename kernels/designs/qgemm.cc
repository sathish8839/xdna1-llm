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

#if QM_FN == 1
// sigmoid(x) from a table: 64 segments of 0.25 over [-8, 8), each slope * x + offset with a bf16 slope and an
// fp32 offset fitted to the bf16 inputs of its segment (error <= 1.3e-4 inside, the end segments are 0 and 1:
// <= 4.3e-4). It is read the way mlir-aie's LUT kernels read theirs (aie_kernels/common/activations.h): one
// parallel lookup from two copies in two banks, x * 64 the byte offset of the segment's 16-byte entry. AIE2 has
// no fp32 vector multiplier (fp32 aie::mul is a bf16 emulation), so the products around it are bf16 macs on
// split operands instead.
// Each bank run of 8 uint16 holds two {slope, offset} pairs, and the lookup wants every run stored twice.
#define SIG_ROW(s0, o0, s1, o1, s2, o2, s3, o3) s0, o0, s1, o1, s0, o0, s1, o1, s2, o2, s3, o3, s2, o2, s3, o3
#define SIG_TABLE \
  { \
    SIG_ROW(0.0f, 0.0f, 0.00048828125f, 0.0042128693f, 0.00062942505f, 0.0052709053f, 0.0008049011f, 0.0065421048f), \
    SIG_ROW(0.0010375977f, 0.008169975f, 0.0013275146f, 0.0101250205f, 0.001701355f, 0.012553425f, 0.0021820068f, 0.015555893f), \
    SIG_ROW(0.0028076172f, 0.019307002f, 0.0035858154f, 0.023777468f, 0.0046081543f, 0.029396694f, 0.0058898926f, 0.036119416f), \
    SIG_ROW(0.007537842f, 0.04435314f, 0.009643555f, 0.054347143f, 0.012329102f, 0.06641968f, 0.01574707f, 0.08092587f), \
    SIG_ROW(0.020019531f, 0.097986184f, 0.025390625f, 0.11809529f, 0.031982422f, 0.14113867f, 0.040283203f, 0.16811001f), \
    SIG_ROW(0.05053711f, 0.19886737f, 0.06298828f, 0.23309767f, 0.078125f, 0.2709083f, 0.095214844f, 0.3093025f), \
    SIG_ROW(0.115234375f, 0.34935018f, 0.13769531f, 0.38864267f, 0.16113281f, 0.42374778f, 0.18457031f, 0.4530732f), \
    SIG_ROW(0.20703125f, 0.475709f, 0.2265625f, 0.49051702f, 0.24121094f, 0.4979666f, 0.24902344f, 0.49999896f), \
    SIG_ROW(0.24902344f, 0.50000244f, 0.24121094f, 0.50203943f, 0.2265625f, 0.5094667f, 0.20703125f, 0.5242701f), \
    SIG_ROW(0.18457031f, 0.54688114f, 0.16113281f, 0.5762565f, 0.13769531f, 0.6113801f, 0.115234375f, 0.65063196f), \
    SIG_ROW(0.095214844f, 0.69067264f, 0.078125f, 0.7291117f, 0.06298828f, 0.7668901f, 0.05053711f, 0.8011194f), \
    SIG_ROW(0.040283203f, 0.83187586f, 0.031982422f, 0.8588518f, 0.025390625f, 0.8819111f, 0.020019531f, 0.9020207f), \
    SIG_ROW(0.01574707f, 0.9190812f, 0.012329102f, 0.9335831f, 0.009643555f, 0.94565356f, 0.007537842f, 0.95564675f), \
    SIG_ROW(0.0058898926f, 0.9638805f, 0.0046081543f, 0.97060436f, 0.0035858154f, 0.97622204f, 0.0028076172f, 0.98069394f), \
    SIG_ROW(0.0021820068f, 0.9844439f, 0.001701355f, 0.9874465f, 0.0013275146f, 0.9898751f, 0.0010375977f, 0.99183047f), \
    SIG_ROW(0.0008049011f, 0.9934578f, 0.00062942505f, 0.99472934f, 0.00048828125f, 0.9957871f, 0.0f, 1.0f) \
  }
__attribute__((section(".aie.bank0"))) alignas(aie::vector_decl_align) static float sig_ab[256] = SIG_TABLE;
__attribute__((section(".aie.bank1"))) alignas(aie::vector_decl_align) static float sig_cd[256] = SIG_TABLE;
#undef SIG_TABLE
#undef SIG_ROW

static inline facc16 sigmoid(bf16x16 x) {
  const bf16x16 xc = aie::max(aie::min(x, bfloat16(8.0f - 1.0f / 32)), bfloat16(-8.0f));
  const aie::vector<int32_t, 16> index = aie::add(aie::vector<int32_t, 16>(bfloat16_to_int(xc, 6)), 512);
  v32bfloat16 c0, c1;
  load_lut_2x_float(sig_ab, sig_cd, index, c0, c1);
  // lanes 0-15 of the slopes are their zero low halves, so x goes in twice
  return mac_elem_16_2(::shuffle(c0, c1, T16_16x4_lo), aie::concat(xc, xc), (v16accfloat)::shuffle(c0, c1, T32_16x2_hi));
}

// x = hi + lo, two bf16 (to ~2^-17 relative)
static inline void split(facc16 a, bf16x16 &hi, bf16x16 &lo) {
  hi = a.to_vector<bfloat16>();
  lo = aie::msc(a, hi, aie::broadcast<bfloat16, 16>(1.0f)).to_vector<bfloat16>();
}

// bf16(silu(g) * u) for fp32 g, u: three-term products of split operands, rounded once at the end
static inline bf16x16 glu(f32x16 g, f32x16 u) {
  bf16x16 gh, gl, sh, sl, uh, ul, th, tl;
  split(facc16(g), gh, gl);
  split(sigmoid(gh), sh, sl);
  split(aie::mac(aie::mac(aie::mul(gh, sh), gh, sl), gl, sh), th, tl);  // t = silu(g)
  split(facc16(u), uh, ul);
  return aie::mac(aie::mac(aie::mul(th, uh), th, ul), tl, uh).to_vector<bfloat16>();
}
#endif

// Each entry point is its own object file (qgemm.py builds this source once per function, QM_FN selecting it).
extern "C" {

#if QM_FN == 1
// The output tile c (64 x 32 fp32, mm.cc's 4 x 4 micro-tiles, 16 row blocks of 8) after its last k slice.
// rtp[3] = 1 (gated, for FFN gate + up with the weight rows interleaved in 16-row blocks: columns 0-15 are
// gate, 16-31 up): silu(gate) * up rounded to bf16, 64 x 16, stored in place where the C stream transform
// (row-major, 32 words per row) puts it in words 0-7 of each row: micro-tile 0 of a row block holds columns
// 0-7 of its 4 rows (4 bf16 pairs each), micro-tile 1 columns 8-15. The rest of the tile is left as it was.
void qm_finish(float *__restrict c, const int32_t *__restrict rtp) {
  if (rtp[3] == 0)
    return;
  for (int z = 0; z < 16; z++) {  // row block z: rows 4z .. 4z + 3
    float *__restrict cz = c + z * 128;
    bf16x16 a[4];
    for (int j = 0; j < 4; j++)  // columns 4j .. 4j + 3 of gate (micro-tile j) and up (j + 4)
      a[j] = glu(aie::load_v<16>(cz + j * 16), aie::load_v<16>(cz + (j + 4) * 16));
    const auto [p0, p1] = aie::interleave_zip(a[0], a[1], 4);  // per row: columns 0-3, then 4-7
    const auto [q0, q1] = aie::interleave_zip(a[2], a[3], 4);
    aie::store_v((bfloat16 *)cz, aie::concat(p0, p1));
    aie::store_v((bfloat16 *)(cz + 16), aie::concat(q0, q1));
  }
}
#endif

#if QM_FN == 0

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
#endif

}  // extern "C"
