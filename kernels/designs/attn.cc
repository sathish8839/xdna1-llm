// Prompt attention on an npu1 core: one query head, 32 query rows at a time, flash-attention style over 32-key
// blocks (online softmax), so any number of keys fits in core memory. kernels/designs/attn.py drives it: the core
// for query head h reads kv head h / 4, so the 4 cores of an array column share each K / V block.
//
// Per key block:  S = Q K^T                       (bf16 in, fp32 out, mmul<4, 8, 4>)
//                 mask S (causal, sliding window) only on blocks that need it
//                 m' = max(m, rowmax S)           (bf16), kept only if it grows by more than LAZY
//                 c = exp2(m - m'), P = exp2(S - m')
//                 O = O * c + P [V | 1]           (the ones column makes O's column 128 the row sum l)
// The host scales Q by 1 / sqrt(head dim) and K by log2(e), so the softmax is exp2 of S directly, and computes
// O / l. The row max m only rescales O when it grows by more than LAZY (FlashAttention-3's trick): P = exp2(S - m)
// then stays below 2^LAZY, which bf16 and the fp32 sums hold easily, and most key blocks skip the rescale of O.
//
// Layouts (bf16; mmul<4, 8, 4> tiles: A 4x8, B 8x4, C 4x4, each row-major):
//   Q   A tiles [ti 8][td 16]: Q[4 ti + r][8 td + c]      (the DMA tiles the host's rows)
//   K   B tiles [tj 8][td 16]: K[4 tj + c][8 td + r]      (one 32-key block, 8 KB)
//   V   B tiles [tn 32][tk 4]: V[8 tk + r][4 tn + c]      (8 KB, after its K block)
//   O   C tiles [ti 8][tn 33]: O[4 ti + r][4 tn + c], fp32; tile column 32 holds l in every lane of a row (the
//       DMA untiles it)
// Before its key blocks, each block of query rows gets a header block (int32): [key blocks, position of query
// row 0, position of key 0 of the first block, window (0 = none)]. Positions only matter relative to each other
// (masks), so the host may give them relative to the first key it sends.
//
// AIE2 has no fp32 multiplier: exp2 is a cubic in bf16 MACs (after mlir-aie's aie_kernels/common/exp2_bf16.h,
// within 0.2 %, below bf16's rounding of P), and O * c takes O as two bf16 limbs.
//
// IRON compiles one object per ExternalFunction, so each function is built on its own (ATTN_FN); their state
// lives in buffers the design passes (st: m and c per row, the header, rescale flags).
#include <aie_api/aie.hpp>
#include <stdint.h>

#define LQ 32  // query rows per call of attn_begin
#define LK 32  // keys per block
#define HD 128
#define TI (LQ / 4)      // A / C tile rows
#define TD (HD / 8)      // A tiles per Q row (k = head dim)
#define TJ (LK / 4)      // C tiles per S row
#define TK (LK / 8)      // A tiles per P row (k = keys)
#define TN (HD / 4 + 1)  // C tiles per O row, the last for l
#define LAZY 8.0f        // log2 of how far P may exceed 1 before O is rescaled
// Masked scores and the initial row max: exact in bf16, so a row whose keys so far are all masked gets
// S - m = 0 (finite P, wiped by c = 0 at its first real key) and never 2^127.
#define NEG (-0x1p100f)

// st, in floats: m[TI][16] as bf16 (lanes 4r..4r+3 hold row r), c[TI][16] as bf16, then int32s
#define ST_M 0
#define ST_C (TI * 16 / 2)
#define ST_I (TI * 16)
enum { I_KB, I_QPOS, I_KPOS, I_WIN, I_FLAG };  // key block counter, header, rescale flag per tile row (I_FLAG + ti)

typedef aie::vector<bfloat16, 16> bf16x16;
typedef aie::vector<float, 16> f32x16;
typedef aie::accum<accfloat, 16> facc16;
using MMUL = aie::mmul<4, 8, 4, bfloat16, bfloat16, accfloat>;

static inline bf16x16 to_bf16(f32x16 v) {
  facc16 a;
  a.from_vector(v);
  return a.to_vector<bfloat16>();
}

static inline f32x16 to_f32(bf16x16 v) {
  facc16 a;
  a.from_vector(v);
  return a.to_vector<float>();
}

// 2^x rounded to bf16, for x <= LAZY + 1: x = k + f, k = round(x), |f| <= 1/2; 2^f a cubic in bf16 MACs, 2^k
// added to the exponent; x clamped to [-200, 127], so masked lanes (NEG - m) give +0.
static inline bf16x16 exp2v(f32x16 x) {
  x = aie::max(x, aie::broadcast<float, 16>(-200.0f));
  x = aie::min(x, aie::broadcast<float, 16>(127.0f));
  const f32x16 magic = aie::broadcast<float, 16>(12582912.0f);  // 1.5 * 2^23: adding it rounds to an integer
  const f32x16 xm = aie::add(x, magic);
  const aie::vector<int32_t, 16> k = aie::sub(xm.cast_to<int32_t>(), aie::broadcast<int32_t, 16>(0x4b400000));
  facc16 fa;
  fa.from_vector(aie::sub(x, aie::sub(xm, magic)));
  const bf16x16 f = fa.to_vector<bfloat16>();
  auto horner = [&](bf16x16 c1, float c0) {
    facc16 acc;
    acc.from_vector(aie::broadcast<float, 16>(c0));
    return aie::mac(acc, f, c1);
  };
  facc16 t = horner(aie::broadcast<bfloat16, 16>(0.0555041087f), 0.2402265069f);
  t = horner(t.to_vector<bfloat16>(), 0.6931471805f);
  t = horner(t.to_vector<bfloat16>(), 1.0f);
  facc16 r;
  r.from_vector(
      aie::max(aie::add(t.to_vector<float>().cast_to<int32_t>(), aie::upshift(k, 23)), aie::zeros<int32_t, 16>())
          .cast_to<float>());
  return r.to_vector<bfloat16>();
}

// Lane 4r + c of the result is the max of lanes 4r .. 4r + 3 of v: swap neighbours, then pairs.
static inline bf16x16 rowmax4(bf16x16 v) {
  auto [e, o] = aie::interleave_unzip(v, v, 1);
  bf16x16 s = aie::interleave_zip(o, e, 1).first;
  v = aie::max(v, s);
  auto [e2, o2] = aie::interleave_unzip(v, v, 2);
  s = aie::interleave_zip(o2, e2, 2).first;
  return aie::max(v, s);
}

#if ATTN_FN == 0
// A new block of query rows, from its header block h: O = 0, m = NEG, the key block count into n[0] (the core's
// loop count), the positions into st.
extern "C" void attn_begin(bfloat16 *__restrict h, float *__restrict o, float *__restrict st, int32_t *__restrict n) {
  const int32_t *hdr = (const int32_t *)h;
  int32_t *si = (int32_t *)st + ST_I;
  n[0] = hdr[0];
  si[I_KB] = 0;
  si[I_QPOS] = hdr[1];
  si[I_KPOS] = hdr[2];
  si[I_WIN] = hdr[3];
  for (int i = 0; i < TI * TN; i++)
    aie::store_v(o + 16 * i, aie::zeros<float, 16>());
  bfloat16 *m = (bfloat16 *)(st + ST_M);
  for (int i = 0; i < TI; i++)
    aie::store_v(m + 16 * i, aie::broadcast<bfloat16, 16>(NEG));
}
#endif

#if ATTN_FN == 1
// S = Q K^T for the next key block, then the softmax step: the row maxima (rescale flags and c where they grew),
// and P (bf16, A tiles) in p.
extern "C" void attn_qk(bfloat16 *__restrict q, bfloat16 *__restrict kt, float *__restrict s, bfloat16 *__restrict p,
                        float *__restrict st) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  int32_t *si = (int32_t *)st + ST_I;
  const int32_t kb = si[I_KB], qpos = si[I_QPOS], kpos = si[I_KPOS] + LK * kb, win = si[I_WIN];
  si[I_KB] = kb + 1;

  for (int ti = 0; ti < TI; ti += 2)
    for (int tj = 0; tj < TJ; tj += 2) {
      const bfloat16 *a0 = q + ti * TD * 32, *a1 = a0 + TD * 32;
      const bfloat16 *b0 = kt + tj * TD * 32, *b1 = b0 + TD * 32;
      MMUL c00, c01, c10, c11;
      {
        auto A0 = aie::load_v<32>(a0), A1 = aie::load_v<32>(a1);
        auto B0 = aie::load_v<32>(b0), B1 = aie::load_v<32>(b1);
        c00.mul(A0, B0);
        c01.mul(A0, B1);
        c10.mul(A1, B0);
        c11.mul(A1, B1);
      }
      for (int td = 1; td < TD; td++) {
        auto A0 = aie::load_v<32>(a0 + 32 * td), A1 = aie::load_v<32>(a1 + 32 * td);
        auto B0 = aie::load_v<32>(b0 + 32 * td), B1 = aie::load_v<32>(b1 + 32 * td);
        c00.mac(A0, B0);
        c01.mac(A0, B1);
        c10.mac(A1, B0);
        c11.mac(A1, B1);
      }
      aie::store_v(s + 16 * (ti * TJ + tj), c00.to_vector<float>());
      aie::store_v(s + 16 * (ti * TJ + tj + 1), c01.to_vector<float>());
      aie::store_v(s + 16 * ((ti + 1) * TJ + tj), c10.to_vector<float>());
      aie::store_v(s + 16 * ((ti + 1) * TJ + tj + 1), c11.to_vector<float>());
    }

  // Key minus query position of each lane, before the tile offsets: c - r.
  alignas(aie::vector_decl_align) static const int32_t cmr[16] = {0, 1, 2, 3, -1, 0, 1, 2, -2, -1, 0, 1, -3, -2, -1, 0};
  const aie::vector<int32_t, 16> dcr = aie::load_v<16>(cmr);
  const bool masked = kpos + LK - 1 > qpos || (win > 0 && qpos + LQ - 1 - kpos >= win);
  bfloat16 *m = (bfloat16 *)(st + ST_M), *cc = (bfloat16 *)(st + ST_C);

  for (int ti = 0; ti < TI; ti++) {
    f32x16 S[TJ];
    for (int tj = 0; tj < TJ; tj++)
      S[tj] = aie::load_v<16>(s + 16 * (ti * TJ + tj));
    if (masked)
      for (int tj = 0; tj < TJ; tj++) {
        const aie::vector<int32_t, 16> d = aie::add(dcr, aie::broadcast<int32_t, 16>(kpos - qpos + 4 * (tj - ti)));
        aie::mask<16> drop = aie::gt(d, aie::zeros<int32_t, 16>());  // key after the query
        if (win > 0)
          drop = drop | aie::le(d, aie::broadcast<int32_t, 16>(-win));  // key outside the window
        S[tj] = aie::select(S[tj], aie::broadcast<float, 16>(NEG), drop);
      }
    bf16x16 mx = to_bf16(S[0]);
    for (int tj = 1; tj < TJ; tj++)
      mx = aie::max(mx, to_bf16(S[tj]));
    bf16x16 m_row = aie::load_v<16>(m + 16 * ti);
    const f32x16 m_old = to_f32(m_row);
    const bf16x16 m_blk = aie::max(m_row, rowmax4(mx));
    int32_t flag = 0;
    if (!aie::gt(aie::sub(to_f32(m_blk), m_old), aie::broadcast<float, 16>(LAZY)).empty()) {
      m_row = m_blk;
      aie::store_v(m + 16 * ti, m_row);
      aie::store_v(cc + 16 * ti, exp2v(aie::sub(m_old, to_f32(m_row))));
      flag = kb > 0;  // on the first block O is still 0
    }
    si[I_FLAG + ti] = flag;
    const f32x16 mf = to_f32(m_row);
    for (int tk = 0; tk < TK; tk++) {
      const bf16x16 e0 = exp2v(aie::sub(S[2 * tk], mf)), e1 = exp2v(aie::sub(S[2 * tk + 1], mf));
      const auto [lo, hi] = aie::interleave_zip(e0, e1, 4);  // rows 0-1, rows 2-3 of the 4x8 A tile
      aie::store_v(p + 32 * (ti * TK + tk), aie::concat(lo, hi));
    }
  }
}
#endif

#if ATTN_FN == 2
// An O tile as an accumulator, times the row factors c if R: O = hi + lo in bf16, each times c exactly.
template <bool R>
static inline facc16 o_acc(const float *t, bf16x16 c) {
  facc16 a;
  a.from_vector(aie::load_v<16>(t));
  if constexpr (R) {
    const bf16x16 hi = a.to_vector<bfloat16>();
    const bf16x16 lo = aie::msc(a, hi, aie::broadcast<bfloat16, 16>(1.0f)).to_vector<bfloat16>();
    return aie::mac(aie::mul(hi, c), lo, c);
  }
  return a;
}

// Two tile rows of O (o0, o1) += their P rows (a0, a1) [V | 1], rescaled first where R0 / R1.
template <bool R0, bool R1>
static inline void pv_rows(const bfloat16 *a0, const bfloat16 *a1, const bfloat16 *v, float *o0, float *o1,
                           bf16x16 c0, bf16x16 c1) {
  for (int tn = 0; tn < TN - 1; tn += 2) {
    const bfloat16 *b0 = v + 32 * tn * TK, *b1 = b0 + 32 * TK;
    MMUL c00(o_acc<R0>(o0 + 16 * tn, c0)), c01(o_acc<R0>(o0 + 16 * (tn + 1), c0));
    MMUL c10(o_acc<R1>(o1 + 16 * tn, c1)), c11(o_acc<R1>(o1 + 16 * (tn + 1), c1));
    for (int tk = 0; tk < TK; tk++) {
      auto A0 = aie::load_v<32>(a0 + 32 * tk), A1 = aie::load_v<32>(a1 + 32 * tk);
      auto B0 = aie::load_v<32>(b0 + 32 * tk), B1 = aie::load_v<32>(b1 + 32 * tk);
      c00.mac(A0, B0);
      c01.mac(A0, B1);
      c10.mac(A1, B0);
      c11.mac(A1, B1);
    }
    aie::store_v(o0 + 16 * tn, c00.to_vector<float>());
    aie::store_v(o0 + 16 * (tn + 1), c01.to_vector<float>());
    aie::store_v(o1 + 16 * tn, c10.to_vector<float>());
    aie::store_v(o1 + 16 * (tn + 1), c11.to_vector<float>());
  }
  const aie::vector<bfloat16, 32> ones = aie::broadcast<bfloat16, 32>(1.0f);
  MMUL l0(o_acc<R0>(o0 + 16 * (TN - 1), c0)), l1(o_acc<R1>(o1 + 16 * (TN - 1), c1));
  for (int tk = 0; tk < TK; tk++) {
    l0.mac(aie::load_v<32>(a0 + 32 * tk), ones);
    l1.mac(aie::load_v<32>(a1 + 32 * tk), ones);
  }
  aie::store_v(o0 + 16 * (TN - 1), l0.to_vector<float>());
  aie::store_v(o1 + 16 * (TN - 1), l1.to_vector<float>());
}

// O = O * c + P [V | 1] for the key block's V (O * c only on the tile rows attn_qk flagged).
extern "C" void attn_pv(bfloat16 *__restrict p, bfloat16 *__restrict v, float *__restrict o, float *__restrict st) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *cc = (const bfloat16 *)(st + ST_C);
  const int32_t *flag = (const int32_t *)st + ST_I + I_FLAG;
  for (int ti = 0; ti < TI; ti += 2) {
    const bf16x16 c0 = aie::load_v<16>(cc + 16 * ti), c1 = aie::load_v<16>(cc + 16 * (ti + 1));
    const bfloat16 *a0 = p + 32 * ti * TK, *a1 = a0 + 32 * TK;
    float *o0 = o + 16 * ti * TN, *o1 = o0 + 16 * TN;
    switch (2 * flag[ti] + flag[ti + 1]) {
    case 0:
      pv_rows<false, false>(a0, a1, v, o0, o1, c0, c1);
      break;
    case 1:
      pv_rows<false, true>(a0, a1, v, o0, o1, c0, c1);
      break;
    case 2:
      pv_rows<true, false>(a0, a1, v, o0, o1, c0, c1);
      break;
    default:
      pv_rows<true, true>(a0, a1, v, o0, o1, c0, c1);
    }
  }
}
#endif

