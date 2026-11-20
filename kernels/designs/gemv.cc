// bf16 GEMV for one decode token on an npu1 core: y[rows] = W[rows, GEMV_K] . x[GEMV_K], fp32 output.
// Compile-time: GEMV_K (row length), GEMV_KS (row stride in a tile), GEMV_R (rows per weight tile).
//
// IRON compiles one object per ExternalFunction, so each function is built on its own (GEMV_FN) and they
// share the output cursor through a global.
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {

#if GEMV_FN == 0
int32_t gemv_pos;  // next output row; reset per run by gemv_load_x

// The first tile of a run carries x, followed by the number of weight tiles to expect (int32 at element GEMV_K).
void gemv_load_x(const bfloat16 *__restrict src, bfloat16 *__restrict x, int32_t *__restrict tiles) {
  for (int i = 0; i < GEMV_K; i += 32)
    aie::store_v(x + i, aie::load_v<32>(src + i));
  tiles[0] = *(const int32_t *)(src + GEMV_K);
  gemv_pos = 0;
}
#else
extern int32_t gemv_pos;

void gemv_rows(const bfloat16 *__restrict w, const bfloat16 *__restrict x, float *__restrict y) {
  for (int r = 0; r < GEMV_R; r++) {
    const bfloat16 *__restrict wr = w + r * GEMV_KS;
    // four independent accumulators hide the MAC latency
    aie::accum<accfloat, 16> a0 = aie::zeros<accfloat, 16>(), a1 = a0, a2 = a0, a3 = a0;
    for (int i = 0; i < GEMV_K; i += 64) {
      a0 = aie::mac(a0, aie::load_v<16>(wr + i), aie::load_v<16>(x + i));
      a1 = aie::mac(a1, aie::load_v<16>(wr + i + 16), aie::load_v<16>(x + i + 16));
      a2 = aie::mac(a2, aie::load_v<16>(wr + i + 32), aie::load_v<16>(x + i + 32));
      a3 = aie::mac(a3, aie::load_v<16>(wr + i + 48), aie::load_v<16>(x + i + 48));
    }
    aie::vector<float, 16> s = aie::add(aie::add(a0.to_vector<float>(), a1.to_vector<float>()),
                                        aie::add(a2.to_vector<float>(), a3.to_vector<float>()));
    y[gemv_pos + r] = aie::reduce_add(s);
  }
  gemv_pos += GEMV_R;
}
#endif

}  // extern "C"
