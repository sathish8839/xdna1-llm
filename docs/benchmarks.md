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
* The GEMM design streams decode weights at only ~9 GB/s, and int8 barely helps (2.26 vs 2.60 ms), so the limit is
  its DMA/tiling pattern, not bytes. The dedicated GEMV in section 3b fixes this.
* bf16 output on K = 7680 shows ~3 % max relative error; the engine uses fp32 output.

## 3b. Decode GEMV (M = 1)

`kernels/designs/stream_bw.py` streams host memory into cores with no compute. `kernels/designs/gemv.py` (bf16) and
`kernels/designs/qgemv.py` (GGUF Q4_K and Q6_K) are the decode GEMVs: 8 lanes (two shim channels per column,
cores in rows 2–3), each lane owning N/8 rows, with the activation and a small header sent on the same lane ahead
of the weights. Because the cores read the sizes from the data, one bf16 xclbin per K serves every N, and one
quantized xclbin serves every format, K and N; only the instruction stream changes. Medians of 50 runs,
including dispatch.

| Streaming only (16 MB) | 1 lane | 2 lanes | 4 lanes | 8 lanes |
|---|---|---|---|---|
| GB/s | 6.4 | 11.5 | 19.3 | **23.5** |

| bf16 shape (N×K) | time | stream |
|---|---|---|
| k+v 1024×2048 | 0.30 ms | 14.0 GB/s |
| q / o / conv-out 2048×2048 | 0.44 ms | 19.3 GB/s |
| conv in_proj 6144×2048 | 1.05 ms | 24.1 GB/s |
| FFN gate+up 15360×2048 | 2.23 ms | 28.2 GB/s |
| FFN down 2048×7680 | 1.16 ms | 27.1 GB/s |

Quantized GEMVs, `benchmarks/bench_qgemv.py`, on real tensors from `LFM3-1.7B-Q4_K_M.gguf` where the shape
exists in the model:

| Q4_K shape (N×K) | bytes | time | weights/s | max rel err vs GGUF |
|---|---|---|---|---|
| attn k 512×2048 | 0.7 MB | 0.20 ms | 5.2 G/s | 5e-5 |
| q / o / conv-out 2048×2048 | 2.6 MB | 0.27 ms | 15.4 G/s | 6e-5 – 1.2e-4 |
| conv in_proj 6144×2048 | 7.9 MB | 0.47 ms | 26.9 G/s | 5e-5 |
| FFN gate / up 7680×2048 | 9.8 MB | 0.54 ms | 29.0 G/s | 6e-5 |
| FFN gate+up fused 15360×2048 (random) | 19.7 MB | 0.88 ms | 35.8 G/s | 3e-5 |
| FFN down 2048×7680 | 9.8 MB | 0.57 ms | 27.7 G/s | 1.5e-4 |
| LM-head size 128000×2048 (random) | 164 MB | **5.9 ms** | **44.4 G/s** | 4e-5 |

| Q6_K shape (N×K) | bytes | time | weights/s | max rel err vs GGUF |
|---|---|---|---|---|
| attn v 512×2048 | 0.9 MB | 0.21 ms | 5.0 G/s | 2e-5 |
| FFN down 2048×7680 | 13.4 MB | 0.81–0.86 ms | 18.8 G/s | 1e-5 |
| LM head (`token_embd`) 128000×2048 | 223 MB | **9.5 ms** | **27.5 G/s** | 2e-5 |

Faster 6-bit kernel (current; real tensors from an all-Q6_K GGUF; the machine's speed varies by ~10 % between
runs, so ranges are given):

| Q6_K shape (N×K) | before | unified tiles | q6only tiles (no padding) |
|---|---|---|---|
| conv out 2048×2048 | 0.31 ms | 0.28 ms | 0.28–0.31 ms |
| conv in_proj 6144×2048 | 0.65 ms | 0.53–0.55 ms | **0.49 ms** |
| FFN down 2048×7680 | 0.79 ms | **0.63–0.65 ms** | same (K = 7680 has no padding either way) |
| FFN gate+up 15360×2048 (random) | 1.32 ms | 1.09–1.14 ms | **1.02–1.12 ms** |
| LM head 128000×2048 | 9.9 ms | 8.1–8.9 ms | **7.5–7.6 ms** (35 G weights/s) |
| Q4_K FFN gate+up 15360×2048 | 0.85 ms | 0.77 ms | — |
| Q4_K FFN down 2048×7680 | 0.54 ms | 0.50 ms | — |

* The old loop took 111 instruction bundles per 256 weights but issued only ~170 operations: it waited on
  latency, not on the MAC unit. 8-bit shifts compile to accumulator round trips (`vups` / `vsrs`), each x value
  was extracted from a vector register before broadcasting, and a group's 8 MACs formed one dependent chain.
  Now: masks on full 64-byte vectors, bytes to bf16 with one shuffle (zip with `0x43`), x read as scalars from
  memory, 4 independent accumulators per group, and the group loop unrolled twice (99 bundles, two groups
  overlapping). Q4_K shares the helpers and gains ~10 %.
* Once compute caught up, the 12.5 % padding of K = 2048 Q6_K rows (768-column tiles) showed up again in the big
  shapes. All-Q6_K models therefore use a second build, `q6only`: 512-column, 7296-byte tiles, which divide
  K = 2048 and 7680 exactly. Q4_K_M models keep the unified build, so they still run in one context.
* Results are unchanged: max relative error vs the GGUF dequantization as before, and the accuracy gate passes
  (KLD 0.00005 for Q6_K, 0.0001 for Q4_K_M against their CPU runs).

* bf16 matches numpy to ~2e-7. The quantized kernels are checked against an fp64 product with the GGUF
  dequantization (`gguf.quants`); the remaining ~1e-4 is fp32 rounding.
* Switching instruction streams inside one context costs nothing: alternating 2048- and 15360-row calls takes
  exactly the average of the two.
* 4-bit layout: Q4_K's own numbers, reordered (5.25 bits per weight; the GGUF file is 4.5). The 6-bit scales and
  mins are unpacked to bf16 (small integers, exact) and each fp16 super-block scale is stored as two bf16 halves
  that add up to it exactly, so `w = d·sc·q − dmin·mn` is computed with nothing rounded away. Pre-multiplying
  `d·sc` into one bf16 instead costs ~1 % error on real weights. Nibbles become bf16 exactly via the exponent
  trick (`0x4300 | q` = 128 + q). AIE2 emulates fp32 multiplies, so every scaling step uses native bf16 MACs on
  high/low bf16 parts: that change alone took the kernel from 17.5 to 35 G weights/s.
* 6-bit layout: Q6_K's low 4 bits are stored like the Q4_K nibbles and its high 2 bits packed four per byte in
  the lane order of a group's two 64-byte nibble vectors, so merging them is a few byte-wise shifts and masks;
  the int8 group scales and the fp16 scale are stored as in the 4-bit layout (7.1 bits per weight; the GGUF
  file is 6.56).
* Rows are split into fixed-size tiles (10.9 KB: 1024 Q4_K or 768 Q6_K columns), so one core program handles
  both formats and every K; the last tile of a row may be partly used (Q4_K pads under 2 %, K = 2048 Q6_K rows
  12.5 %; all-Q6_K models use 7.3 KB, 512-column tiles with no padding). The core keeps the partial sums between a row's tiles, and each 16-row block's outputs stream back as
  soon as they are done, so N is not limited by core memory. The tables above are from the earlier per-format
  builds; the merged kernel is within ~5 % (LM head 10.0 ms).
* Each call carries ~0.15–0.2 ms of fixed cost, which is why small shapes reach less. Fusing calls per layer
  removes most of it.
* Estimated LFM3-1.7B 4-bit decode at these rates, one call per projection: ~10 ms conv layers, ~9 ms attention
  projections, ~38 ms FFN (11 of 24 `ffn_down` are Q6_K), ~9.5 ms LM head (Q6_K, tied to `token_embd`), so
  ~67 ms or **~15 tok/s** before fusing. bf16 with the same scheme is ~7 tok/s. Measured end to end: section 3c.

## 3c. LFM3-1.7B Q4_K_M decode end to end

`xdna1-llm --gguf LFM3-1.7B-Q4_K_M.gguf --generate 24 --compare`: every linear layer and the LM head run on the
quantized GEMVs; norms, RoPE, attention over the KV cache and the short conv run on the CPU in numpy. The prompt
is fed one token at a time for now.

| | |
|---|---|
| | four contexts (one per format and K) | one context | **+ q/k/v fused, polled completion** |
|---|---|---|---|
| Generation | 8.5 tok/s | 11.0–11.9 tok/s | **13.8–13.9 tok/s** |
| Per token: linear layers | 107 ms, 113 calls, 59 context switches | 77–81 ms, 113 calls | 67 ms, 102 calls |
| Per token: CPU ops | 8.4 ms | 7.7–9.5 ms | 4.7–5.1 ms |
| Logits vs. fp32 CPU reference | cosine ≥ 0.99993 | cosine ≥ 0.99989 | cosine ≥ 0.99989, same top-1 every step |

* With four contexts the layers alternated between Q4_K/Q6_K and K = 2048/7680 GEMVs, costing ~45 ms per token
  in context switches; the merged kernel removes them.
* Most of a small call's fixed cost is the host round trip, not NPU work: one 512-row GEMV takes 0.245 ms with
  `run.wait()`, 0.193 ms when the host polls the run's state instead, and 0.086 ms each when ten are queued
  back to back. So the engine polls for completion, stacks q, k and v into one weight (one call, or a Q4_K and
  a Q6_K call queued together where `attn_v` is Q6_K), and gives every weight its own x buffer so queued calls
  do not overwrite each other.
* What is left is mostly the kernels themselves (~60 ms per token, a third of it in Q6_K: LM head and half of
  FFN down). The other calls depend on CPU work between them (norms, short conv, attention, SwiGLU), so fusing
  further means moving that work onto the NPU.
* llama.cpp's greedy output for the same prompt diverges after a few tokens where two candidates are close.
  At that step both the fp32 reference and llama.cpp's own prompt evaluation rank our token first; llama.cpp's
  CPU decode path rounds activations to 8 bits, so its token-by-token output differs slightly from its own
  batched evaluation.

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

## 6. Accuracy: what quantization costs

`benchmarks/eval_quality.py` makes every variant predict each next token of a text
([`benchmarks/data/eval.txt`](../benchmarks/data/eval.txt): prose, Python, a worked word problem, a tool call,
instructions, dialogue; 1,441 tokens in chunks of 256, each starting from BOS with an empty context) and compares it
with the original BF16 weights run in fp32 on the CPU. These are the measures llama.cpp uses to judge
quantizations:

* **perplexity**: how surprised the model is by the real text (lower is better);
* **KL divergence (KLD)**: how far the variant's probabilities over the whole vocabulary are from BF16's
  (0 = identical), mean and 99th percentile;
* **same top-1**: how often both rank the same token first.

| Variant | file | perplexity | mean KLD vs BF16 | KLD p99 | same top-1 as BF16 |
|---|---|---|---|---|---|
| BF16 (reference) | 3.47 GB | 6.129 | — | — | — |
| Q8_0 | 1.85 GB | 6.125 | 0.0003 | 0.002 | 99.4 % |
| Q6_K | 1.43 GB | 6.098 | 0.0031 | 0.019 | 97.4 % |
| **Q4_K_M** | 1.10 GB | 6.321 (+3.1 %) | 0.035 | 0.21 | 92.0 % |
| Q4_K_M on our NPU engine | | 6.324 | 0.035 | 0.21 | 92.0 % |
| Q6_K on our NPU engine | | 6.099 | 0.0031 | 0.020 | 97.4 % |
| *NPU engine vs the same Q4_K_M on the CPU* | | | *0.00002* | *0.0001* | *99.7 %* |
| *NPU engine vs the same Q6_K on the CPU* | | | *0.00002* | *0.0002* | *99.7 %* |

* **The engine adds nothing measurable.** Against the same GGUF weights in fp32 the NPU path's KLD is 0.00002,
  over 100× below what either format costs; the 0.3 % of tokens where the top-1 differs are near-ties.
* **Q4_K_M has a real cost on this 1.7B model:** perplexity +3 %, and about one token in twelve gets a
  different first choice than BF16 (often a near-synonym, but it shows up first in arithmetic, exact facts and
  formats). Small models lose more to 4-bit than 7B+ ones.
* **Q6_K is close to lossless** (KLD 11× lower, perplexity within noise of BF16) for 30 % more bytes, and runs
  on the same decode GEMV and context (four more instruction streams: 2048, 3072, 6144 and 15360 × 2048).

| Decode on the NPU (32–64 tokens) | file | generation | linears per token | logits vs BF16 (KLD) |
|---|---|---|---|---|
| Q4_K_M ("fast"), first 6-bit kernel | 1.10 GB | 13.5–13.7 tok/s | 68 ms, 102 calls | 0.035 |
| Q6_K ("quality"), first 6-bit kernel | 1.43 GB | 10.6–10.9 tok/s | 86–88 ms, 97 calls | 0.003 |
| Q4_K_M, faster kernel | | **14.1–14.9 tok/s** | 61.5–64 ms | 0.035 |
| Q6_K, faster kernel, q6only tiles | | **12.1–13.2 tok/s** | 72–75.5 ms | 0.003 |

  The laptop also has a slower state (CPU ops double from ~5.5 to ~10 ms, and the NPU calls slow down too) in
  which the same builds give 12.4 (Q4_K_M) and 10.2–10.9 tok/s (Q6_K); the table compares runs in the normal
  state.
* 1,441 tokens is small: differences under ~1 % perplexity are noise. (An earlier version of this table started
  each chunk with the text's first token instead of BOS, which inflated every number; the ranking was the same.) Q6_K / Q8_0 were made locally from the
  BF16 GGUF with `llama-quantize` (no imatrix).

## Related work

* [amd/IRON](https://github.com/amd/IRON): operator library and Llama 3.2 1B on NPU1 (Linux).
* [hnws/xdna1-npu-lab](https://github.com/hnws/xdna1-npu-lab): the 6-context limit on Linux, runtime-parameter GEMM.
* [Tobynau/npu-ai](https://github.com/Tobynau/npu-ai), [c8dhjp4tyv-bit/hawkpoint-npu-llm](https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm),
  [qwq12738qwq/llama.cpp-hybrid-XDNA1](https://github.com/qwq12738qwq/llama.cpp-hybrid-XDNA1),
  [Scottcjn/open-xdna](https://github.com/Scottcjn/open-xdna): XDNA1 LLM experiments on Linux.
