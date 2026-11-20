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
