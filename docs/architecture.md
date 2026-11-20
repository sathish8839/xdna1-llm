# Architecture

This describes what exists today. Planned parts (tokenizer, server, Q4_K_M, the app) are in the
[roadmap](roadmap.md).

```
engine/src/xdna1_llm/
  npu/device.py      pyxrt wrapper: device, kernels (hw contexts), buffers, zero-copy numpy views
  npu/linear.py      NpuLinear: every projection as 1024-wide chunks on two fixed GEMM kernels
  cpu_linear.py      CpuLinear: the same interface in numpy, as a reference and fallback
  models/lfm3.py     LFM3 forward pass (short conv, GQA attention, SwiGLU) over a linear backend
  gguf_weights.py    memory-mapped GGUF tensors (BF16, F32)
  bf16.py            float32 <-> bf16 bit conversion
  reference.py       llama-server client: tokenizer and ground truth until the native tokenizer exists
  cli.py             run a prompt on the NPU and/or CPU and compare against llama.cpp
kernels/build_kernels.sh   builds the xclbins from mlir-aie's whole_array GEMM example
```

## Constraints that shape the design

All of these were measured on the Windows driver; see [benchmarks.md](benchmarks.md).

1. **At most 5 hardware contexts.** Opening a 6th fails. The engine uses two.
2. **Switching between contexts costs ~1.45 ms. Swapping buffers within a context is free.** So there is
   one kernel per K dimension, reused for every layer, with each weight chunk in its own resident buffer.
3. **Each dispatch costs ~0.1–0.2 ms.** All chunks of a projection are queued first and then awaited.
4. **64 KB of data memory per compute tile.** An fp32 64×64 output tile plus double-buffered inputs does
   not fit, so the kernels use 64×64×32 tiles.
5. **Decode is bandwidth-bound.** The prefill GEMM streams weights at ~9 GB/s; the dedicated 8-lane GEMV
   reaches 17–28 GB/s, and the Q4_K / Q6_K GEMVs 15–44 G weights/s.

## Why the model code is plain numpy

The numpy path is the reference that the NPU path is checked against, layer by layer (`--compare`). Ops move
onto the NPU one at a time, and each must match the numpy result before it replaces it.
