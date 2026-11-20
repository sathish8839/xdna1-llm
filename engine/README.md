# xdna1-llm engine

The Python inference engine behind [xdna1-llm](../README.md): the NPU runtime (pyxrt), the LFM3 model,
GGUF loading and the command-line tool. See the [top-level README](../README.md) for what the project is
and [docs/setup.md](../docs/setup.md) for installing XRT and building kernels.

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
.venv\Scripts\python.exe -m pytest
```

Tests marked `npu` run only when `pyxrt` imports and kernels are built; everything else runs anywhere.
