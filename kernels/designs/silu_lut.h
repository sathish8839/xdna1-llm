// SwiGLU's silu(g) * u on an AIE2 core, shared by the prompt GEMM (qgemm.cc) and the decode GEMV (qgemv.cc).
// Include it in one object file per kernel: the tables are static and live in data memory banks 0 and 1.
#pragma once
#include <aie_api/aie.hpp>

typedef aie::vector<bfloat16, 16> bf16x16;
typedef aie::vector<float, 16> f32x16;
typedef aie::accum<accfloat, 16> facc16;

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
