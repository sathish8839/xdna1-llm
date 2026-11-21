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

## NPU driver

The engine needs AMD's NPU driver: Device Manager lists it under **Neural processors**. It is tested on
32.0.203.297; with any other version it logs `note: NPU driver ... is untested` and carries on. An npu1 abort on
Windows was reported with driver 32.0.20101.3760
([amd/Triton-XDNA#88](https://github.com/amd/Triton-XDNA/issues/88)).

* **"No AMD NPU found"**: the processor has no XDNA1 NPU, or the driver is not installed. Install it from your
  PC maker's support page.
* **"The NPU driver did not accept the kernel"**: the driver refused our npu1 kernels, which newer drivers may
  do. Installing 32.0.203.297 works; please also open a hardware report with your driver version.

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

Kernel sets: `qgemv` (the Q4_K / Q6_K decode GEMV, every shape of Q4_K_M and Q6_K models), `qgemm` (the W4A16
prompt GEMM, which reads the same weights), `engine` (the bf16 prompt GEMMs, used only when `qgemm` is not built),
`gemv` (bf16 decode GEMV), `shapes` (benchmark kernels), `bw`, `smoke`, `all`.

**In CI** (`.github/workflows/release.yml`, on a `v*` tag or by hand) the kernels build on an Ubuntu runner from
mlir-aie at the pinned commit. XRT ships no Linux `xclbinutil`, so with `XCLBINUTIL_DEFER=<dir>` the build
records each call (`kernels/xclbinutil-defer`) and writes placeholder `.xclbin` files; a Windows job then runs
the calls with the SDK's `xclbinutil.exe` (`kernels/package_xclbins.py`), builds the installer, and attaches both
to a draft release. The same split works locally: record in WSL, then package on Windows.

## 3. Model

From [LiquidAI/LFM3-1.7B-GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF):

* **Q4_K_M** for decode (`--generate`): fastest (~14–15 tok/s), +3 % perplexity against BF16.
* **Q6_K** for decode at near-BF16 accuracy (~12–13 tok/s; uses the `q6only_*` kernel builds). If the repository has no Q6_K file, make one from the
  BF16 GGUF: `llama-quantize LFM3-1.7B-BF16.gguf LFM3-1.7B-Q6_K.gguf Q6_K`.
* **BF16** for the bf16 prefill path (`--compare` without `--generate`) and as the reference of
  `benchmarks/eval_quality.py` ([benchmarks §6](benchmarks.md#6-accuracy-what-quantization-costs)).
* Optional: a llama.cpp `llama-server` with LFM3 support, as a reference. Pass `--llama-server <path>` and the
  CLI starts it on the CPU and prints its output next to ours; `benchmarks/check_tokenizer.py` uses it to check
  the tokenizer.

## 4. Run

```powershell
cd engine; python -m pytest -rs; cd ..
xdna1-llm --gguf <gguf> --chat --prompt "Hello" --generate 100 --sample   # greedy without --sample
xdna1-llm --gguf <gguf> --llama-server <llama-server.exe> --compare
python benchmarks\bench_shapes.py
python benchmarks\power_bench.py --gguf <gguf> --llama-server <llama-server.exe>
```

`bench_shapes.py` needs the `shapes` kernel set.

### The chat app

Needs the .NET 8 SDK (no Visual Studio).

```powershell
cd app
dotnet build -c Release
$env:XDNA1_LLM_GGUF = "<path>\LFM3-1.7B-Q4_K_M.gguf"   # or pick the file with Model… in the app
.\bin\Release\net8.0-windows10.0.19041.0\win-x64\xdna1-chat.exe
```

The app finds the engine's Python in `engine\.venv` above it (or `$env:XDNA1_LLM_PYTHON`), starts the server on a
free localhost port and stops it when the window closes. Its settings live in `%LOCALAPPDATA%\xdna1-llm\app.json`
and the engine's output in `engine.log` next to it. Web search sends the model's queries to DuckDuckGo; nothing
else leaves the machine.

**Theme** switches the look, and the choice is saved:

* **Windows**: the native look (Mica, system accent, light or dark with the system).
* **Silicon**: dark, with an NPU panel whose 16 tiles light up as tokens stream, and the live speed.
* **Paper**: light and editorial, answers in a serif reading font in one column.
* **Console**: monospace, amber on black, a caret while text streams.
* **Spotlight**: a small acrylic window near the top of the screen, always on top. Alt+Space shows or hides it
  from anywhere, Esc hides it. (PowerToys Run also uses Alt+Space; if it holds the key, the app says so.)

Animations use the compositor and follow Windows' "Animation effects" setting.

### Portable build

```powershell
.\installer\build-portable.ps1   # -> dist\xdna1-llm
```

This assembles a folder that runs on any XDNA1 PC with the stock NPU driver, with no Python, .NET or WSL. It
contains the app (published self-contained), the embeddable CPython 3.13 with the engine, its dependencies and
`pyxrt.pyd` in `python\`, and the kernels in `kernels\`. It needs the .NET 8 SDK, uv, built kernels and pyxrt in
`engine\.venv` (or pass `-Kernels` and `-Pyxrt`). The app uses `python\` and `kernels\` when they sit beside it.
The folder also gets `LICENSE`, `NOTICE`, `THIRD-PARTY-NOTICES.txt` and `licenses\` (written by
`installer\notices.py` from what the build ships).

### Installer

```powershell
winget install JRSoftware.InnoSetup   # once
.\installer\build-installer.ps1       # -> dist\xdna1-llm-setup-<version>.exe
```

Wraps the portable build with Inno Setup (`installer\xdna1-llm.iss`, about 60 MB). It installs for the current user
into `%LOCALAPPDATA%\Programs\xdna1-llm` without an admin prompt, warns when the NPU driver is missing, and adds a
Start menu entry. On first run the app offers to download LFM3-1.7B Q4_K_M or Q6_K from Hugging Face (a pinned
revision, checked by SHA-256) into `%LOCALAPPDATA%\xdna1-llm\models`. Uninstalling asks whether to delete the
models, weight cache and settings too.

### The OpenAI-compatible server

```powershell
xdna1-llm-server --gguf <gguf> [--port 8080] [--thinking] [--ctx 4096]
```

Point any OpenAI client at `http://127.0.0.1:8080/v1` (any API key works). It serves `POST /v1/chat/completions`
(streaming, `tools`, `stop`, `max_tokens`, `temperature`, `top_p`, `seed`, plus `top_k`, `min_p` and
`repeat_penalty`), `GET /v1/models` and `GET /health`. Reasoning comes back in `reasoning_content`; thinking is off
unless `--thinking` is given or a request sends `chat_template_kwargs: {"enable_thinking": true}` or a
`reasoning_effort` other than `none`. The NPU runs one turn at a time, so concurrent requests queue. The server
binds to localhost; pass `--host 0.0.0.0` only on a network you trust (there is no authentication).

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(model="LFM3-1.7B-Q4_K_M", messages=[{"role": "user", "content": "Hi"}])
```

Environment variables:

* `XDNA1_LLM_KERNELS`: the kernel directory. By default the engine uses `kernels/build` in a source checkout,
  else `%LOCALAPPDATA%\xdna1-llm\kernels`.
* `XDNA1_LLM_GGUF`: the default model path.
* `XDNA1_LLM_CACHE`: where the converted NPU weights are kept (default `%LOCALAPPDATA%\xdna1-llm\cache`, about
  4 GB per model, rebuilt when the model or the layout code changes). `--no-weight-cache` turns it off.
