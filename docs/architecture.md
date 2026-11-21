# Architecture

This describes what exists today. Planned parts (the installer, fused kernels) are in the
[roadmap](roadmap.md).

```
engine/src/xdna1_llm/
  npu/device.py      pyxrt wrapper: device, kernels (hw contexts), buffers, zero-copy numpy views
  npu/qlinear.py     NpuQLinear: decode, Q4_K / Q6_K weights repacked once into device buffers, on the quantized GEMV
                     (the FFN as one chain: gate + up writes silu(gate) * up into down's x on the device)
  npu/qgemm.py       NpuQGemm: prompts of up to 256 tokens on the W4A16 GEMM, reading NpuQLinear's buffers
                     (the FFN fused: silu(gate) * up on the cores, left on the device for down)
  npu/linear.py      NpuLinear: the older bf16 prompt path, 1024-wide chunks on two fixed GEMM kernels
  cpu_linear.py      CpuLinear: the same interface in numpy, as a reference and fallback
  models/lfm3.py     LFM3: batched prefill and one-token decode over a shared state, on one or two linear backends
  gguf_weights.py    memory-mapped GGUF tensors (BF16, F32, Q4_K, Q6_K, ...)
  q4k.py, q6k.py     the GEMV / GEMM tile layout of Q4_K and Q6_K, lossless
  weight_cache.py    the NPU backends' converted weights on disk, so a later load skips converting them
  bf16.py            float32 <-> bf16 bit conversion
  tokenizer.py       byte-level BPE from the GGUF (Llama 3 pre-tokenizer), identical to llama.cpp
  chat.py            LFM3's chat template (thinking, tools) in Python
  sampling.py        next-token choice: greedy, or temperature / top-k / top-p / min-p / repetition penalty
  reference.py       llama-server client: the optional ground truth for tests and benchmarks
  engine.py          one chat turn: template, prompt, sampling; splits reasoning, content and tool calls;
                     resumes from the last turn's state, so a follow-up only runs the tokens that are new
  server.py          OpenAI-compatible HTTP server (standard library), one turn at a time on the NPU
  cli.py             run a prompt or chat message on the NPU and/or CPU, optionally against llama.cpp
app/                 WinUI 3 chat app (C#, unpackaged, Windows App SDK copied next to the exe)
  EngineHost.cs      starts `python -m xdna1_llm.server` on a free localhost port, in a kill-on-close job object
  ChatClient.cs      streams /v1/chat/completions
  WebSearch.cs       the web_search tool: DuckDuckGo's HTML results, run by the app, never by the engine
  MarkdownView.cs    markdown (Markdig) to native WinUI text, code blocks and tables; keeps finished blocks
  Themes.cs          the five looks: shared brushes recoloured in place, WinUI control colours, composition animations
  Hotkey.cs          Spotlight's system-wide Alt+Space
  MainWindow.xaml    status bar, chat, composer with the thinking and web search toggles
kernels/designs/     IRON designs and their core C++: qgemv (decode), qgemm (prompt), gemv, stream_bw
kernels/build_kernels.sh   builds the xclbins (the designs, and mlir-aie's whole_array GEMM example)
```

## Constraints that shape the design

All of these were measured on the Windows driver; see [benchmarks.md](benchmarks.md).

1. **At most 5 hardware contexts.** Opening a 6th fails. The engine uses two.
2. **Switching between contexts costs ~1.45 ms. Swapping buffers within a context is free.** So each kernel
   takes its sizes at run time: one context runs every decode GEMV, one every prompt GEMM, with each weight in
   its own resident buffer that both read.
3. **Each dispatch costs ~0.1–0.2 ms.** All chunks of a projection are queued first and then awaited.
4. **64 KB of data memory per compute tile.** An fp32 64×64 output tile plus double-buffered inputs does
   not fit, so the kernels use 64×64×32 tiles.
5. **Decode is bandwidth-bound.** The prefill GEMM streams weights at ~9 GB/s; the dedicated 8-lane GEMV
   reaches 17–28 GB/s, and the Q4_K / Q6_K GEMVs 15–44 G weights/s.

## Why the model code is plain numpy

The numpy path is the reference that the NPU path is checked against, layer by layer (`--compare`). Ops move
onto the NPU one at a time, and each must match the numpy result before it replaces it.
