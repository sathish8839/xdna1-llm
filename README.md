<div align="center">

# xdna1-llm

**Local LLMs on the NPU of Ryzen 7040 / 8040 / Ryzen 200 laptops**

[![CI](https://github.com/sathish8839/xdna1-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/sathish8839/xdna1-llm/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Windows%2011-0078d4)
![NPU](https://img.shields.io/badge/NPU-AMD%20XDNA1%20(npu1)-ed1c24)
![Python](https://img.shields.io/badge/python-3.13-3776ab)

</div>

Millions of laptops have a first-generation AMD XDNA NPU (Phoenix / Hawk Point, 16 TOPS) that sits idle,
because today's NPU LLM stacks (FastFlowLM, Lemonade, Ryzen AI OGA and Unsloth's NPU backend) require XDNA 2.
**xdna1-llm runs LLMs on that NPU instead.** It starts with [LiquidAI LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B)
and aims for the Q4_K_M quant running entirely on the NPU, behind:

- an **OpenAI-compatible local server**, so you can use it from Open WebUI, Unsloth Studio, IDE plugins or the
  OpenAI SDKs, and
- a small **native Windows chat app** with a thinking toggle and DuckDuckGo web search.

> [!IMPORTANT]
> **Pre-alpha.** Prefill and token generation (Q4_K_M or Q6_K) run on the NPU and match a CPU reference. The
> tokenizer, the server and the app are still being built. See the [roadmap](docs/roadmap.md).

## Results so far

LFM3-1.7B BF16, 256-token prompt, Ryzen 7 250, Windows 11, on AC power:

| | tokens/s | SoC power | CPU load | energy per token |
|---|---:|---:|---:|---:|
| **xdna1-llm** (all linear layers on the NPU) | **149** | **26 W** | **14 %** | **0.12 J** |
| llama.cpp, CPU only (8 threads) | 95 | 38.5 W | 60 % | 0.30 J |

That is **1.6× faster and 2.5× less energy than the CPU, with the CPU nearly free.** The output matches llama.cpp
(top-5 tokens identical, logits cosine similarity 0.999998). Method, driver limits and per-shape numbers are in
[docs/benchmarks.md](docs/benchmarks.md).

## Status

| Feature | |
|---|---|
| NPU kernels compiled with mlir-aie, running on the **Windows** NPU driver | ✅ |
| Prefill with every linear layer on the NPU (bf16) | ✅ |
| Native tokenizer, chat template, generation loop | ⏳ next |
| Decode on the NPU (KV cache): **Q4_K_M** ~14–15 tok/s, or **Q6_K** ~12–13 tok/s at near-BF16 accuracy | ✅ |
| Accuracy harness: perplexity and KL divergence vs BF16; NPU engine adds ~0 on top of the GGUF | ✅ |
| OpenAI-compatible server (streaming, tool calls) | ⏳ |
| Windows app (chat, thinking toggle, web search) and installer | ⏳ |
| Norms, conv, attention and SwiGLU fused on the NPU | ⏳ |

## Supported hardware

| | |
|---|---|
| Processors | Ryzen 7040 "Phoenix", 8040 "Hawk Point" and Ryzen 200 series with a Ryzen AI (XDNA1) NPU |
| OS | Windows 11 with the AMD NPU driver (tested: 32.0.203.297) |
| Memory | 16 GB RAM recommended |

Only a Ryzen 7 250 has been tested so far. **Have one of these laptops? A [test report](CONTRIBUTING.md)
is the most valuable contribution right now**, especially with newer NPU drivers.

## Try it (developers)

There is no installer yet. For now you need the XRT SDK, plus WSL to build the kernels. The full guide is in
[docs/setup.md](docs/setup.md).

```powershell
cd engine
uv venv --python 3.13 .venv; .venv\Scripts\activate
uv pip install -e ".[dev]"
copy <xrt_sdk>\xrt\python\pyxrt.pyd .venv\Lib\site-packages\
python -m pytest -rs                       # CPU tests, plus NPU tests when kernels are built
xdna1-llm --gguf LFM3-1.7B-BF16.gguf --llama-server <llama-server.exe> --compare               # prefill
xdna1-llm --gguf LFM3-1.7B-Q4_K_M.gguf --llama-server <llama-server.exe> --generate 24 --compare  # decode (or Q6_K)
```

## How it works

```
 app (WinUI 3) ─┐
 any OpenAI ────┼─▶ engine (Python): server · tokenizer · LFM3 · Q4_K repack · pyxrt runtime
 client       ──┘                                                    │
                                                  XDNA1 NPU ◀────────┘  prebuilt npu1 kernels
```

The Windows XDNA1 driver allows **5 hardware contexts**, and switching between them costs **~1.45 ms**.
Swapping weight buffers inside one context is free. So the engine keeps a few long-lived kernels, uploads all
weights once, and streams every projection through them in 1024-wide chunks. Details are in
[docs/architecture.md](docs/architecture.md).

## Repository layout

```
engine/       Python engine: xdna1_llm package (npu runtime, models, GGUF, CLI) and tests
kernels/      scripts that build the npu1 kernels with mlir-aie (in WSL or Linux)
benchmarks/   GEMM shape, context-switch and power benchmarks
docs/         setup, architecture, benchmarks, roadmap
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

Kernels are built from [mlir-aie / IRON](https://github.com/Xilinx/mlir-aie) programming examples (Apache-2.0 WITH
LLVM-exception). The runtime is [XRT](https://github.com/Xilinx/XRT). The model is LiquidAI LFM3 (LFM Open
License); its weights are not redistributed here. [llama.cpp](https://github.com/ggml-org/llama.cpp) is the
reference implementation. Related XDNA1 community work is listed in
[docs/benchmarks.md](docs/benchmarks.md#related-work).

Much of the code is written with Claude (Anthropic) as a coding assistant.

xdna1-llm is an independent project, not affiliated with AMD, Liquid AI or Unsloth.

## License

[Apache-2.0](LICENSE)
