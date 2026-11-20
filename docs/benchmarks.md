# Measurements

Everything below was measured on a Ryzen 7 250 (XDNA1, "NPU Phoenix"), Windows 11, NPU driver 32.0.203.297
(XRT 2.19), firmware 1.5.5.391, unless noted. Medians of 20+ runs.

## 1. XDNA1 works on Windows with self-compiled kernels

* Kernels compiled with mlir-aie (`devicename=npu`) in WSL load and run through `pyxrt` from the XRT 2.21.75 SDK,
  even though the installed driver runtime is XRT 2.19.
* `vector_scalar_mul`: correct, **0.16 ms** per call. `xrt-smi validate`: latency 86.6 µs, throughput 34.4k ops/s.
* This contradicts reports that npu1 aborts on Windows ([amd/Triton-XDNA#88](https://github.com/amd/Triton-XDNA/issues/88),
  driver 32.0.20101.3760). It may depend on the driver version.

## 2. Driver limits that shape the design

| Limit | Measured |
|---|---|
| Concurrent hardware contexts | **5** (the 6th fails with `0xc01e0009`) |
| Alternating between two contexts | **+1.45 ms per call** (0.54 → 1.99 ms) |
| Swapping weight buffers within one context | free (0.538 ms both ways) |
| pyxrt teardown at interpreter exit | can access-violate → exit with `os._exit()` |
| Single GEMM with N = 128000 (LM head) | fails to compile (DMA dimension limit) → split into 10 × 12800 |
| fp32 output with 64×64×64 tiles | does not fit in a tile's 64 KB → use n = 32 |

## 3. GEMM shapes of LFM3-1.7B (bf16 in/out, whole_array 4×4)

| Projection (K→N) | Prefill M=256 | GFLOP/s | Decode stand-in M=64 | Weight stream |
|---|---|---|---|---|
| q / o / conv-out 2048→2048 | 1.36 ms | 1583 | 0.90 ms | 9.3 GB/s |
| k+v 2048→1024 | 0.55 ms | **1944** | 0.40 ms | 10.5 GB/s |
| conv in_proj 2048→6144 | 3.95 ms | 1630 | 2.60 ms | 9.7 GB/s |
| FFN gate+up 2048→15360 | 12.6 ms | 1278 | 7.67 ms | 8.2 GB/s |
| FFN down 7680→2048 | 4.41 ms | 1827 | 3.46 ms | 9.1 GB/s |
| LM head chunk 2048→12800 | 9.19 ms | 1460 | 6.19 ms | 8.5 GB/s |
| conv in_proj, int8 | — | — | 2.26 ms | 5.6 GB/s |

* Prefill GEMMs reach 1.3–1.9 TFLOP/s, close to the best published XDNA1 numbers.
* Decode is limited to ~9 GB/s of weight streaming with this kernel. int8 barely helps (2.26 vs 2.60 ms), so the
  limit is the DMA/tiling pattern, not bytes. A dedicated M=1 GEMV design is needed (see [roadmap](roadmap.md)).
* bf16 output on K = 7680 shows ~3 % max relative error; the engine uses fp32 output.

## 4. LFM3-1.7B prefill with linears on the NPU

Engine: two kernels (256×2048×1024 and 256×7680×1024, bf16 → fp32, n = 32), 567 calls per forward pass,
~2.9 GB of weights resident in device buffers (registration ~15 s).

**Correctness** (256-token prompt): top-5 next tokens identical to llama.cpp, same order; logits cosine 0.999998
vs a CPU fp32 reference; per-layer hidden-state cosine ≥ 0.99995.

**Time per 256-token forward:**

| Part | Time |
|---|---|
| NPU kernels (519 × k2048 at 0.91 ms + 48 × k7680 at 2.86 ms) | ~610 ms |
| Host side of NPU calls (bf16 conversion, output sync, ~48 context switches) | ~320 ms |
| numpy ops between layers (norms, conv, RoPE, attention, SwiGLU, LM head) | ~1.0 s |
| **Total** | 2.28 s cold → 1.72 s sustained (**112–149 tok/s**) |

Profiling notes: `np.einsum` for attention cost 2 s (no BLAS) → batched `np.matmul`; bf16 conversion via
`ml_dtypes` instead of bit tricks.

LFM3 details confirmed by the GGUF: `rope_freqs` is 1.0 for the first 32 pairs and 1e30 for the rest (full-attention
layers rotate 32 of 64 pairs); `attn_gate` is a per-head sigmoid gate applied before `attn_output`; sandwich norms.

## 5. Power (AMD RAPL via Windows "Energy Meter" counters, 30 s per phase, on AC)

| Workload | tok/s | SoC package | Cores | CPU % | dGPU | J/token over idle |
|---|---|---|---|---|---|---|
| Idle | — | 7.7–9.4 W | 3.5 W | 6 % | 9.0 W | — |
| NPU GEMMs only | 254 | 21.7 W | 10.5 W | 11 % | 9.0 W | 0.055 |
| NPU engine, full prefill | 149 | 26.0 W | 15.6 W | 14 % | 9.1 W | 0.123 |
| llama.cpp CPU only (BF16, 8 threads) | 95 | 38.5 W | 32.3 W | 60 % | 9.0 W | 0.305 |
| llama.cpp CUDA build at `-ngl 0` (GPU offload) | 478 | 40.1 W | 31.2 W | 62 % | 20.5 W | ~0.09 incl. dGPU |

* Against CPU-only llama.cpp: **1.6× faster, ~2.5× less energy per token, a quarter of the CPU use.**
* The NPU itself adds only ~7 W over idle (package minus cores).
* Pitfall: CUDA builds of llama.cpp offload prompt batches to the GPU even with `-ngl 0`.

## Related work

* [amd/IRON](https://github.com/amd/IRON): operator library and Llama 3.2 1B on NPU1 (Linux).
* [hnws/xdna1-npu-lab](https://github.com/hnws/xdna1-npu-lab): the 6-context limit on Linux, runtime-parameter GEMM.
* [Tobynau/npu-ai](https://github.com/Tobynau/npu-ai), [c8dhjp4tyv-bit/hawkpoint-npu-llm](https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm),
  [qwq12738qwq/llama.cpp-hybrid-XDNA1](https://github.com/qwq12738qwq/llama.cpp-hybrid-XDNA1),
  [Scottcjn/open-xdna](https://github.com/Scottcjn/open-xdna): XDNA1 LLM experiments on Linux.
