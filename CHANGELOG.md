# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Release workflow (`.github/workflows/release.yml`, on `v*` tags or by hand): builds the `engine` and `qgemv`
  kernels on Ubuntu with mlir-aie, runs their xclbinutil step on Windows with the XRT SDK (XRT has no Linux
  binary: `kernels/xclbinutil-defer` records the calls, `kernels/package_xclbins.py` runs them), builds the
  installer and attaches it and the kernels to a draft release. Checked locally: kernels recorded in WSL and
  packaged on Windows pass the NPU tests and give the same greedy output.
- Installer (`installer/build-installer.ps1`, Inno Setup): the portable build as a 62 MB per-user setup, with a
  Start menu entry, a warning when the NPU driver is missing, and an uninstaller that offers to delete models,
  cache and settings. The portable build now carries `THIRD-PARTY-NOTICES.txt` and `licenses/`, generated from
  the packages it ships.
- First-run model download in the app: LFM3-1.7B Q4_K_M or Q6_K from Hugging Face (pinned revision, SHA-256
  checked, resumable, cancellable) into `%LOCALAPPDATA%\xdna1-llm\models`; the Model… button became a menu to
  switch or download models, or pick a file.
- NPU errors in plain words: no NPU or driver ("No AMD NPU found") and a driver that rejects the kernels (with
  its version and the tested one) raise `NpuError`; the server prints it as one `error:` line, which the app
  shows. An untested driver version is logged. docs/setup.md has an "NPU driver" section.
- Copy button on code blocks in the app.
- The engine reuses the last turn: a follow-up resumes from its state and runs only the tokens that differ
  (a 368-token follow-up: first token 2.97 → 0.99 s). The conv layers' state is snapshotted at the end of each
  prompt chunk and every 32 reply tokens, so a prompt that departs mid-reply rewinds to the nearest snapshot.
  Resumed vs fresh: KLD ≤ 0.0005, same top token. Usage reports `prompt_tokens_details.cached_tokens`.
- Weight cache (`xdna1_llm.weight_cache`): the NPU backends keep their converted weights (the GEMV's repacked
  tiles, the GEMM's bf16 chunks) on disk, so later loads skip the conversion: Q4_K_M 38.0 → 18.5 s (the first
  load writes 3.9 GB and takes 57.7 s). Keyed by the GGUF and the layout code; `--no-weight-cache`,
  `XDNA1_LLM_CACHE`. The GEMM backend now dequantizes to bf16 itself.
- Portable build (`installer/build-portable.ps1`): the app, the embeddable CPython 3.13 with the engine and
  pyxrt, and the kernels in one folder (241 MB) that needs only the NPU driver. The app now references
  `Microsoft.WindowsAppSDK.WinUI` instead of the metapackage (no AI/ML runtimes: 204 → 158 MB) and builds its
  `.pri` on publish, without which the published app crashed at start.
- Windows chat app (`app/`, WinUI 3, .NET 8): streams replies as native markdown, a thinking toggle (reasoning in
  a collapsed section), a web search toggle (the model calls `web_search`; the app queries DuckDuckGo and returns
  the results), Stop, NPU status and tokens per second. It starts the engine on a free localhost port and stops it
  with the window. `/health` now reports the model, device and context length.
- OpenAI-compatible server (`xdna1-llm-server`, standard library only): `/v1/chat/completions` with SSE streaming,
  tool calls in the OpenAI shape, `reasoning_content`, stop strings and sampling options; `/v1/models` and
  `/health`. Binds to localhost and runs one turn at a time. Checked with the OpenAI Python SDK (Q4_K_M: first
  token 1.2 s, 14.3 tok/s). The turn logic is in `xdna1_llm.engine`.
- Batched prompt: `LFM3.forward` prefills in chunks into the decode state, and `--generate` runs prompt chunks of
  32–256 tokens on the bf16 GEMM with a bf16 copy of the weights (689 tokens: 38.6 → 5.1 s; +2.9 GB RAM, +25 s
  load; KLD 0.00005 against the CPU). `--no-gemm-prompt` keeps the old token-by-token path.
- Sampling (`xdna1_llm.sampling`): repetition penalty, top-k, top-p, min-p and temperature, in llama.cpp's order,
  defaulting to the model's recommended values from the GGUF. CLI: `--sample`, `--temperature`, `--top-k`,
  `--top-p`, `--min-p`, `--repeat-penalty`, `--seed`; greedy otherwise.
- Native tokenizer (`xdna1_llm.tokenizer`): byte-level BPE read from the GGUF with the Llama 3 pre-tokenizer,
  identical to llama.cpp token for token (`benchmarks/check_tokenizer.py`). Chat template in Python
  (`xdna1_llm.chat`: thinking, tools, tool results), identical to the GGUF's Jinja template. `xdna1-llm --chat`
  streams text, stops at end of turn and needs no llama-server.
- Fixed: `eval_quality.py` started every chunk with the text's first token instead of BOS (llama.cpp adds no
  BOS for LFM3); accuracy numbers re-measured.

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
- Q6_K "quality" model: an all-Q6_K GGUF decodes on the same NPU GEMV at ~11 tok/s with KLD 0.003 against
  BF16 (Q4_K_M: ~14 tok/s, KLD 0.035). The CLI warns when a running llama-server holds a different model.
- Accuracy harness (`benchmarks/eval_quality.py`, `xdna1_llm.quality`): perplexity, KL divergence and top-1
  agreement against BF16. Q4_K_M costs +3 % perplexity on LFM3-1.7B, Q6_K ~0; the NPU engine adds KLD 0.00002.
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
