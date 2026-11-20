# Roadmap

**Goal:** a Windows app and an OpenAI-compatible local server that run **LFM3-1.7B Q4_K_M entirely on the
XDNA1 NPU** of Ryzen 7040 / 8040 / 200 laptops, the NPUs that today's LLM stacks leave out. The CPU only
tokenizes, schedules and samples.

## Milestones

| | Milestone | State |
|---|---|---|
| M0 | Toolchain: npu1 kernels compiled in WSL run on Windows via pyxrt; driver limits measured | ✅ |
| M1 | Prefill with every linear layer on the NPU (bf16), verified against llama.cpp | ✅ |
| M2 | Native tokenizer, chat template, sampling and a generation loop (no llama-server needed) | ✅ |
| M3 | Decode on the NPU: M=1 GEMV kernel, KV cache, short-conv state | ✅ |
| M4 | **Q4_K_M** weights: Q4_K / Q6_K repacked at load and dequantized on the NPU | ✅ decode; W4A16 prefill open |
| M5 | OpenAI-compatible server: `/v1/chat/completions` (SSE streaming, tool calls), `/v1/models` | ✅ |
| M6 | **v0.1 release:** WinUI 3 app (chat, thinking toggle, DuckDuckGo web search) and an installer with prebuilt kernels | |
| M7 | Every op on the NPU: fused per-layer designs (norms, conv, RoPE, attention, SwiGLU) | |
| M8 | More context (chunked prefill beyond 256 tokens, up to 8K), more small models | |

M7 is listed after the release on purpose. Linear layers are more than 95% of the FLOPs and nearly all of the
bytes, so a release with linears on the NPU (M1–M4) is already most of the work and energy on the NPU. Fusing
the remaining ops is how we get to "entirely".

## Hard constraints

These come from measurements in [benchmarks.md](benchmarks.md) (the last one is a rule we hold ourselves to):

1. **5 hardware contexts**, and **~1.45 ms** to switch between them. Every design keeps a forward pass inside
   1–3 contexts.
2. **~0.1–0.2 ms per dispatch**, so ops must be batched and later fused (one call per layer, not per op).
3. **64 KB data memory per compute tile**, 16 compute tiles (4×4), 4 memory tiles (512 KB each).
4. **No native transcendental functions in AIE2 vectors**: softmax, SiLU and sigmoid need LUT or polynomial
   kernels (mlir-aie and IRON ship AIE2 versions).
5. **Decode weight streaming:** ~9 GB/s with the GEMM design, 14–28 GB/s with the 8-lane bf16 GEMV, and up to
   35 G weights/s with the 4-bit GEMV ([benchmarks](benchmarks.md#3b-decode-gemv-m--1)), which puts 4-bit decode
   at an estimated ~16 tok/s before fusing.
6. **No silent accuracy loss.** The NPU path must match the CPU run of the same GGUF (KLD ≲ 1e-3 on
   `benchmarks/eval_quality.py`) before a speed change lands; what a quantization format costs is measured
   against BF16 and published ([benchmarks §6](benchmarks.md#6-accuracy-what-quantization-costs)).

## M2: tokenizer and generation

* Byte-level BPE from the GGUF (`tokenizer.ggml.tokens`, `merges`, pre-tokenizer `lfm2`): done
  (`xdna1_llm/tokenizer.py`), identical to llama.cpp token for token on the eval text and edge cases
  (`benchmarks/check_tokenizer.py`).
* Chat template: done (`xdna1_llm/chat.py`, a Python port of the GGUF's Jinja template with thinking, tools and
  tool results), identical to the Jinja render. `xdna1-llm --chat` runs without a llama-server.
* Sampling: done (`xdna1_llm/sampling.py`: repetition penalty, top-k, top-p, min-p, temperature, in llama.cpp's
  order; defaults from the GGUF's `general.sampling.*`, for LFM3 temperature 0.1, top-k 50, penalty 1.1). Under 2 ms
  per token.
* Generation loop: done (`LFM3.generate`, stops at `<|im_end|>`); the prompt runs as a batch (M3).

## M3: decode on the NPU

* **GEMV design** for M=1: done (`kernels/designs/gemv.py`), 8 lanes, 14–28 GB/s, exact against numpy. One xclbin
  per K serves every N through per-N instruction streams, so all projections of a K share one context.
* **Quantized GEMVs** (`kernels/designs/qgemv.py`): read Q4_K and Q6_K repacked at load (`xdna1_llm/q4k.py`,
  `q6k.py`), every LFM3 Q4_K_M tensor shape including FFN down (K = 7680) and the 128000-row LM head,
  ~1e-4 from the GGUF dequantization. Remaining: fewer dispatches, a faster 6-bit unpack.
* **Decode loop**: done for Q4_K_M (`LFM3.step` / `generate`, `xdna1-llm --generate`), ~14 tok/s, logits
  within cosine 0.99993 of an fp32 reference. KV cache and short-conv state live on the CPU for now.
* One context for every quantized GEMV (format and sizes travel with the data): done, 8.5 → 11–12 tok/s.
* q/k/v as one call, completion polling instead of a sleeping wait: done, 11–12 → ~14 tok/s.
* Faster 6-bit kernel: done (Q6_K GEMVs 15–25 % faster, Q4_K ~10 %; all-Q6_K models stream no padding).
* Batched prompt: done. Prompt chunks of 32–256 tokens run on the bf16 prefill GEMM with a bf16 copy of the
  weights made at load (+2.9 GB RAM, +25 s load), then decode continues from the same state: 689 tokens in 5.1 s
  instead of 38.6 s, KLD 0.00005 against the CPU ([benchmarks 3d](benchmarks.md#3d-batched-prompt)).
* Next: the CPU ops between calls on the NPU (M7) so a layer needs fewer round trips.
  The GEMV loop still waits on latency more than on the MAC unit (broadcasting x is the largest remaining cost);
  32-row blocks would share each broadcast between two MACs.

## M4: Q4_K_M on the NPU

GGUF Q4_K_M mixes **Q4_K** (most tensors) and **Q6_K** (for LFM3-1.7B: `token_embd`, which is also the LM head,
11 of 24 `ffn_down` and 5 `attn_v`).

| Format | Super-block | Layout | bits/weight |
|---|---|---|---|
| Q4_K | 256 weights = 8 sub-blocks × 32 | fp16 `d`, fp16 `dmin`, 12 B of 6-bit scales/mins, 128 B of 4-bit quants | 4.5 |
| Q6_K | 256 weights = 16 sub-blocks × 16 | 128 B low 4 bits, 64 B high 2 bits, 16 int8 scales, fp16 `d` | 6.56 |

1. **Repack at load** (one time): reorder super-blocks into the tile order the kernels consume and unpack the
   6-bit scales/mins (`w = d·sc·q − dmin·m`). Done for Q4_K and Q6_K (`xdna1_llm/q4k.py`, `q6k.py`),
   lossless; tested against `gguf.quants`.
2. **W4A16 kernel:** unpack nibbles in the vector unit, apply the group scale and min, then bf16 `mmul`. It
   would replace the bf16 weight copy the batched prompt uses today (2.9 GB, ~25 s of load).
3. **Optional W4A8:** AIE2's int4 × int8 MAC modes with per-token int8 activations. Faster, slightly different
   numerics; benchmark against (2).
4. Q6_K: same structure with a 6-bit unpack. Done for decode (`kernels/designs/qgemv.cc`).
5. **Quality option:** done. Q4_K_M costs +3 % perplexity and KLD 0.035 against BF16 on LFM3-1.7B, Q6_K
   KLD 0.003 ([benchmarks §6](benchmarks.md#6-accuracy-what-quantization-costs)). An all-Q6_K GGUF runs on the
   same GEMV ("quality", now ~12–13 tok/s with the faster 6-bit kernel and padding-free `q6only` tiles) next to
   Q4_K_M ("fast", ~14–15 tok/s); the app should offer both.

## M5–M6: server, app, installer

* **Server (done, `xdna1-llm-server`):** OpenAI `/v1/chat/completions` with SSE streaming and `tools`, `/v1/models`, and a health
  endpoint. It binds to localhost by default. It is usable from Open WebUI, Unsloth Studio's external
  provider, IDE plugins and the OpenAI SDKs.
* **App (WinUI 3, `app/`, first version done):** a chat window with streaming markdown, a **thinking** toggle (off by default) and a
  **web search** toggle. With search on, the app offers a `web_search` tool to the model, runs DuckDuckGo
  queries itself and returns the results as tool messages. The app starts and stops the engine and shows
  NPU status and speed.
* **Installer (`installer/`):** an embeddable Python 3.13 with the engine and pyxrt, prebuilt kernels, and the
  app. On first run it downloads the model from Hugging Face. There is no WSL, no Python setup and no driver
  fiddling beyond the stock AMD NPU driver.
* **CI-built kernels:** a GitHub Actions job builds the xclbins with mlir-aie on Linux and attaches them to
  releases.

## M7: every op on the NPU

Per-layer fused designs, one xclbin each (≤ 4 contexts total):

| Design | Ops |
|---|---|
| `conv_layer` | RMSNorm → in_proj → B·x → causal depthwise conv (k=7) → C·y → out_proj → RMSNorm → residual |
| `attn_layer` | RMSNorm → q/kv → q/k RMSNorm → partial RoPE → GQA attention → sigmoid gate → o_proj → RMSNorm → residual |
| `ffn` | RMSNorm → gate+up → SiLU·mul → down → RMSNorm → residual |
| `lm_head` | final RMSNorm → tied head (128K vocab, chunked) → top-k |

The hidden state stays in one device buffer across designs. Each design is validated op by op against the numpy
reference in `xdna1_llm.models.lfm3`, which already matches llama.cpp.

## Open questions

* Do newer Windows NPU drivers (≥ 32.0.20101) still accept these npu1 kernels? ([Triton-XDNA#88](https://github.com/amd/Triton-XDNA/issues/88) says no.)
* Can several runtime sequences share one xclbin on npu1, to merge per-layer designs into fewer contexts?
* Real shim DMA bandwidth on npu1 with an optimal GEMV access pattern (published results: 13–20 GB/s).
