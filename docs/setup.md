# Developer setup

This guide is for building xdna1-llm from source: the Python engine on Windows, and the NPU kernels in WSL.
(End users will get an installer with prebuilt kernels; see the [roadmap](roadmap.md).)

Tested configuration (November 2026):

| Component | Version |
|---|---|
| Laptop | Ryzen 7 250 (Hawk Point refresh, XDNA1 "NPU Phoenix", 4×4 AIE2 tiles) |
| OS | Windows 11 build 26300, WSL2 Ubuntu 24.04 |
| NPU driver | 32.0.203.297 (XRT 2.19.0), NPU firmware 1.5.5.391 |
| XRT Windows SDK | 2.21.75 (`xrt_windows_sdk.zip`, for `pyxrt.pyd` and `xclbinutil.exe`) |
| mlir-aie | 1.4.6.dev63 (commit 0f76700), llvm-aie (Peano) 22.0.0.2026102001 |
| Python | 3.13 on Windows (pyxrt is built for 3.13), 3.12 in WSL |

> **Newer NPU drivers:** an npu1 abort on Windows was reported with driver 32.0.20101.3760
> ([amd/Triton-XDNA#88](https://github.com/amd/Triton-XDNA/issues/88)). On the driver above, our kernels run
> correctly. Please open a hardware report with your driver version and results.

## 1. Windows: NPU runtime and engine

1. Check the NPU: `C:\Windows\System32\AMD\xrt-smi.exe examine` should list **NPU Phoenix**, and
   `xrt-smi validate` should pass.
2. Download the XRT Windows SDK
   (https://github.com/Xilinx/XRT/releases/download/2.21.75/xrt_windows_sdk.zip, 67.6 MB,
   sha256 `ccc244c2c423588972ade76142cdc01049477aaa39a35be97e782b97eb7c5295`) and extract it, for example to
   `C:\Users\<you>\xrt_sdk`.
3. Create the engine environment:
   ```powershell
   cd engine
   uv venv .venv --python 3.13
   .venv\Scripts\activate
   uv pip install -e ".[dev]"
   copy C:\Users\<you>\xrt_sdk\xrt_sdk\xrt\python\pyxrt.pyd .venv\Lib\site-packages\
   python -c "import pyxrt; print(pyxrt.device(0).get_info(pyxrt.xrt_info_device.name))"
   ```

## 2. WSL: kernel toolchain

The NPU is not visible inside WSL. WSL is only used to compile; the kernels then run on Windows.

```bash
sudo apt-get install -y build-essential clang clang-14 lld lld-14 cmake ninja-build python3-venv python3-pip uuid-dev jq
git clone https://github.com/Xilinx/mlir-aie.git ~/mlir-aie && cd ~/mlir-aie
source utils/env_install.sh        # creates ironenv with the mlir_aie + llvm-aie wheels
```

Ubuntu 24.04 has no `xclbinutil`, so `kernels/build_kernels.sh` uses `kernels/xclbinutil-wsl`. This shim calls the
Windows SDK's `xclbinutil.exe` and rewrites Linux paths, including those inside the AIE partition JSON, to
`\\wsl.localhost\...` paths. Point it at the SDK:

```bash
cd /mnt/d/path/to/xdna1-llm
MLIR_AIE=~/mlir-aie XRT_SDK_WIN=/mnt/c/Users/<you>/xrt_sdk/xrt_sdk/xrt ./kernels/build_kernels.sh kernels/build all
```

Kernel sets: `engine` (the two prefill GEMMs), `qgemv` (the Q4_K / Q6_K decode GEMV, every shape of Q4_K_M and Q6_K
models), `gemv` (bf16 decode GEMV), `shapes` (benchmark kernels), `bw`, `smoke`, `all`.

## 3. Model

From [LiquidAI/LFM3-1.7B-GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF):

* **Q4_K_M** for decode (`--generate`): fastest (~14–15 tok/s), +6 % perplexity against BF16.
* **Q6_K** for decode at near-BF16 accuracy (~12–13 tok/s; uses the `q6only_*` kernel builds). If the repository has no Q6_K file, make one from the
  BF16 GGUF: `llama-quantize LFM3-1.7B-BF16.gguf LFM3-1.7B-Q6_K.gguf Q6_K`.
* **BF16** for the bf16 prefill path (`--compare` without `--generate`) and as the reference of
  `benchmarks/eval_quality.py` ([benchmarks §6](benchmarks.md#6-accuracy-what-quantization-costs)).
* Until the native tokenizer lands, a llama.cpp `llama-server` with LFM3 support provides `/tokenize` and the
  reference output. Pass `--llama-server <path>` and the CLI starts it on the CPU.

## 4. Run

```powershell
cd engine; python -m pytest -rs; cd ..
xdna1-llm --gguf <gguf> --llama-server <llama-server.exe> --compare
python benchmarks\bench_shapes.py
python benchmarks\power_bench.py --gguf <gguf> --llama-server <llama-server.exe>
```

`bench_shapes.py` needs the `shapes` kernel set.

Environment variables:

* `XDNA1_LLM_KERNELS`: the kernel directory. By default the engine uses `kernels/build` in a source checkout,
  else `%LOCALAPPDATA%\xdna1-llm\kernels`.
* `XDNA1_LLM_GGUF`: the default model path.
