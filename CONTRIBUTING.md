# Contributing

Issues and pull requests are welcome.

**Test reports are the most useful contribution right now.** If you have an XDNA1 laptop (Ryzen 7040, 8040 or
Ryzen 200 series), install the app and fill in a
[hardware report](https://github.com/sathish8839/xdna1-llm/issues/new?template=hardware-report.yml): your
processor, NPU driver version, whether it worked and how fast. From a source checkout, add the output of
`pytest -rs`.

## Development

Setup is described in [docs/setup.md](docs/setup.md). Then:

```powershell
cd engine
python -m pytest -rs        # npu-marked tests skip themselves without pyxrt or built kernels
ruff check .. ; ruff format ..
```

## Pull requests

- Keep each PR to one change, and explain why in the description.
- For changes to numerics or kernels, include accuracy against llama.cpp, speed before and after, and the
  hardware you measured on.
- Note user-visible changes in [CHANGELOG.md](CHANGELOG.md).
- If you used an AI assistant, say so in the PR. That's fine here, but you are responsible for the code and
  should be able to explain it.

Contributions are licensed under [Apache-2.0](LICENSE).
