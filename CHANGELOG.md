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
- 4-bit GEMV (Q4_K-style scales and mins): up to 35 G weights/s for K = 2048, ~2e-5 from an fp32 reference.
