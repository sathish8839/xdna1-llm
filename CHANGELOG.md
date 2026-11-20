# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- NPU runtime for XDNA1 (npu1) on Windows through pyxrt from the XRT Windows SDK.
- Chunked bf16 GEMM engine: every LFM3-1.7B linear layer runs on the NPU (prefill up to 256 tokens).
- LFM3 model (short conv, GQA attention with sliding window, SwiGLU) loading BF16 GGUF weights; output matches
  llama.cpp (logits cosine 0.999998).
- `xdna1-llm` CLI with a `--compare` mode against llama.cpp.
- Kernel build script for mlir-aie in WSL, plus shape, context-switch and power benchmarks.
- CPU unit tests and on-device tests (`npu` marker), ruff and GitHub Actions CI.
- Decode GEMV kernel (8 lanes, bf16, fp32 out): 14–28 GB/s weight streaming on LFM3 shapes, plus a
  streaming-bandwidth probe and benchmarks.
- One GEMV xclbin per K now serves every N: the tile count travels with the activation, and
  `Kernel.add_program` switches instruction streams inside one context.
- Faster quantized GEMV: 64-byte bit masks, bytes to bf16 by one shuffle, x read from memory, independent
  accumulators and a twice-unrolled group loop (Q6_K GEMVs 15–25 % faster, Q4_K ~10 %). All-Q6_K models use
  a padding-free `q6only` tile geometry. Decode: Q6_K ~11 → ~12–13 tok/s, Q4_K_M ~14 → ~14–15 tok/s.
- Q6_K "quality" model: an all-Q6_K GGUF decodes on the same NPU GEMV at ~11 tok/s with KLD 0.006 against
  BF16 (Q4_K_M: ~14 tok/s, KLD 0.056). The CLI warns when a running llama-server holds a different model.
- Accuracy harness (`benchmarks/eval_quality.py`, `xdna1_llm.quality`): perplexity, KL divergence and top-1
  agreement against BF16. Q4_K_M costs +6 % perplexity on LFM3-1.7B, Q6_K ~0; the NPU engine adds KLD 0.0001.
  The CPU reference also reads Q8_0.
- Decode: q, k and v run as one weight, calls with several weight groups are queued together, and the host
  polls for completion instead of sleeping (11–12 → ~14 tok/s).
- One NPU program for both quantized formats and every matrix shape: the format and sizes travel in a header
  with the activation, so decode runs in a single hardware context (8.5 → 11–12 tok/s).
- Decode loop for Q4_K_M: `LFM3.step` with KV cache and short-conv state, `NpuQLinear` running every linear
  layer and the LM head on the quantized GEMVs, and `xdna1-llm --generate N [--compare]`. 8.5 tok/s, logits
  within cosine 0.99993 of an fp32 reference.
- `GGUFWeights.tensor()` / `f32()` read Q4_K and Q6_K tensors; `CpuLinear` takes them too (dequantizing per call).
- 6-bit GEMV on GGUF Q6_K weights (LM head, part of FFN down, attn_v), lossless repack in `xdna1_llm/q6k.py`;
  the real LM head runs in 9.5 ms. Q4_K and Q6_K share one design script (`qgemv.py`).
- 4-bit GEMV on GGUF Q4_K weights, repacked losslessly at load (`xdna1_llm/q4k.py`): long rows split across
  tiles, outputs streamed per 16 rows, 15–44 G weights/s on LFM3 shapes, ~1e-4 from the GGUF dequantization.
