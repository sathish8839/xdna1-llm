# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.0] - 2026-11-21

Long prompts take about a third less time, with the same accuracy. Prompt attention runs on the NPU, and the
CPU work of a prompt overlaps the NPU's. For Q4_K_M, a 4K-token prompt goes from 14.1 to 9.2 s (291 → 447
tok/s) and a 1K-token prompt from 2.9 to 2.3 s; KLD against the CPU 0.00006 (was 0.00005). Generation is
unchanged. The v0.3.0 kernels still work, but without the new `attn` set (the installer brings it) prompt
attention stays on the CPU.

### Added

- Prompt attention on the NPU (`kernels/designs/attn.py`, `xdna1_llm.npu.attn`): flash attention on all 16
  cores, one kv head per array column, one context for every key length (`attn_256` … `attn_4352` and
  `attn_swa`, the new `attn` set in `build_kernels.sh`). Each conversation keeps a bf16 copy of its KV cache on
  the device. A 4K-token Q4_K_M prompt takes 11.1 s instead of 14.1 s (291 → 370 tok/s), 1K tokens 2.74 s
  instead of 2.93 s; KLD against the CPU 0.00006 (was 0.00005). Chunks no build covers fall back to the CPU.

### Changed

- Prompts run layer by layer (every chunk through a layer before the next layer), so the NPU switches between
  the GEMM and attention contexts twice per attention layer instead of twice per chunk.
- The CPU work of a prompt overlaps the NPU: the prompt GEMM runs while the host works (`submit` / `collect`),
  two chunks are in flight, and the CPU ops, the attention output and the q/k/v GEMM result go into reused
  arrays instead of fresh ones (on Windows a fresh multi-MB array costs ~0.3 ms in page faults). Q4_K_M
  prompts: 4K tokens 11.1 → 9.2 s (370 → 447 tok/s), 1K tokens 2.76 → 2.30 s; the output is identical.

## [0.3.0] - 2026-11-21

Faster on both ends, with the same output. The FFN's SwiGLU now runs on the NPU cores for prompts and for
generation, and the CPU work between the NPU calls is rewritten. For Q4_K_M, generation goes from ~12.7 to
~15–17 tok/s, an 820-token prompt from ~3.3 s to ~2.7 s, and generation after a 4K-token prompt from ~10 to
~13–14 tok/s; KLD against the CPU is unchanged. Kernels built for v0.2.0 do not work with this version; the
installer brings new ones, and the weight cache rebuilds itself on the first load.

### Added

- Fused FFN for prompts (`NpuQGemm.ffn`): the gate + up GEMM computes silu(gate) · up on the NPU cores and
  leaves it in bf16 on the device, where the down projection reads it. The 15 MB gate + up result, the CPU's
  SiLU and the bf16 conversion are gone: an 814-token Q4_K_M prompt takes 4.29 s instead of 5.39 s (~20 %
  less, same machine and session), KLD 0.00005 against the CPU. SiLU uses a 64-segment sigmoid table, since AIE2
  has no fp32 vector multiplier. New builds `qm4_15360x2048_glu` and `qm6only_15360x2048_glu` in the `qgemm` set.
- Fused FFN for decode (`NpuQLinear.ffn`): the gate + up GEMV's cores write silu(gate) · up as bf16 straight
  into the down GEMV's input, and the down GEMV is queued right behind it, so an FFN costs one round trip
  instead of two and no host work in between. Together with reusing each GEMV's run (below), Q4_K_M generation
  goes from 12.7 to 15.0 tok/s (+18 %, same machine, alternating runs), KLD 0.00003 against the CPU. New builds
  `q4_15360x2048_glu` and `q6only_15360x2048_glu` in the `qgemv` set; the SiLU table is shared with the GEMM
  (`kernels/designs/silu_lut.h`).

### Changed

- FFN gate and up weights are stored with their rows interleaved in blocks of 16 (`xdna1_llm.npu.qlinear`), so
  the weight cache rebuilds on the first load. The `qgemm` and `qgemv` cores changed, so every build of both
  sets must be rebuilt.
- Each decode GEMV keeps one prebuilt XRT run and starts it again (`Kernel.prepare`) instead of building a run
  per call: ~0.025 ms less per call, ~10 % faster generation on its own.
- Faster CPU ops between a prompt's GEMM calls (`xdna1_llm.models.lfm3`): attention runs per kv head on 4
  threads without repeating K and V, RMSNorm takes one pass for the sum of squares, and the short conv
  accumulates in place. CPU time of an 820-token Q4_K_M prompt 1.45 → 0.79 s; the prompt 3.33 → 2.65 s
  (247 → 309 tok/s, alternating runs). Same KLD; decode unchanged.
- The KV cache is stored per kv head (kv head, position, head dim), so attention reads each head's keys as one
  contiguous matrix: decode attention uses BLAS matmuls instead of `einsum`, and prompt attention runs one
  query head per task on 8 threads. After a 4K-token prompt, generation 9.2–10.8 → 12.5–14.4 tok/s (CPU ops
  per token 39–52 → 15–21 ms); prompt attention at 4K tokens 13–23 % faster. Same KLD.

## [0.2.0] - 2026-11-21

Prompts now read the 4-bit weights directly on the NPU instead of a bf16 copy of them. For Q4_K_M that means
2.8 GB less RAM, loading in ~8 s instead of ~31 s, and prompts at ~200 tok/s instead of ~138, with the same
output. Decode is unchanged. Kernels built for v0.1.0 do not work with this version; the installer brings new
ones, and the weight cache rebuilds itself on the first load.

### Added

- W4A16 prompt GEMM (`kernels/designs/qgemm.py`, `xdna1_llm.npu.qgemm`): prompts read the Q4_K / Q6_K weights
  the decode GEMV already holds on the device, so the bf16 weight copy is gone. Q4_K_M: 2.8 GB less RAM, load
  31 → 8 s, prompt 138 → ~209 tok/s (Q6_K 141 → 192 tok/s), same output (KLD 0.00004 against the CPU). New
  kernel set `qgemm` in `kernels/build_kernels.sh`; releases ship `qgemv` + `qgemm`. The bf16 GEMM stays as the
  fallback when `qgemm` is not built. The prompt GEMM converts activations to bf16 on 4 threads and returns
  outputs without copying them (two output buffer sets in turn).

### Changed

- The GEMV tile layout keeps each super-block's bytes together (`xdna1_llm.q4k.sb_tiles`), so the GEMM can read
  it a super-block at a time. Decode speed and output are unchanged; kernels built for v0.1.0 do not run with
  this layout (rebuild `qgemv`), and the weight cache starts afresh.

## [0.1.0] - 2026-11-21

The first release: LiquidAI LFM3-1.7B on the XDNA1 NPU of Ryzen 7040 / 8040 / Ryzen 200 laptops, on Windows.
Prompt processing and generation run on the NPU (Q4_K_M ~15 tok/s, Q6_K ~12–13 tok/s) and match a CPU
reference. It comes as a Windows chat app with an installer that downloads the model on first run, and as an
OpenAI-compatible local server. Tested on one laptop (Ryzen 7 250) with NPU driver 32.0.203.297.

### Added

- App themes (Theme menu, saved): Windows (native, the default), Silicon (an NPU panel with a tile per streamed
  token and the live speed), Paper (serif reading column), Console (monospace, streaming caret) and Spotlight (a
  compact always-on-top window toggled with Alt+Space). Switching recolours the open chat in place. New
  paragraphs animate in, and finished markdown blocks are no longer rebuilt on every streamed token.
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

### Fixed

- `eval_quality.py` started every chunk with the text's first token instead of BOS (llama.cpp adds no BOS for
  LFM3); accuracy numbers re-measured.

[Unreleased]: https://github.com/sathish8839/xdna1-llm/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/sathish8839/xdna1-llm/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/sathish8839/xdna1-llm/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/sathish8839/xdna1-llm/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sathish8839/xdna1-llm/releases/tag/v0.1.0
