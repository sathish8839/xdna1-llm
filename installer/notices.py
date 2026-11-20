"""Writes THIRD-PARTY-NOTICES.txt for a portable build and copies the licence files it refers to into licenses/.

    python notices.py <bundle dir> <app obj/project.assets.json> <NuGet package root>

Run by build-portable.ps1 with the bundle's own Python. The NuGet and Python package lists come from what the
build actually ships, so they cannot drift from it.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from importlib import metadata
from pathlib import Path

HEADER = """xdna1-llm third-party notices

xdna1-llm (the engine, the app and the NPU kernels) is licensed under the Apache License 2.0: see LICENSE and
NOTICE. It ships with the components below, each under its own licence. Licence texts are in licenses/, in each
Python package's *.dist-info folder (python/Lib/site-packages), and in python/LICENSE.txt.

Not included: the model. On first run the app downloads LiquidAI's LFM3-1.7B GGUF from Hugging Face, which is
licensed by Liquid AI under the LFM Open License v1.0:
https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/blob/main/LICENSE

Runtime
  CPython {python} (python/) - Python Software Foundation License Version 2, python/LICENSE.txt
  pyxrt (python/Lib/site-packages/pyxrt.pyd), from the XRT Windows SDK 2.21.75 - Apache License 2.0,
    https://github.com/Xilinx/XRT (it loads xrt_coreutil.dll, which AMD's NPU driver installs)
  .NET 8 runtime (published with the app) - MIT, https://github.com/dotnet/runtime
  NPU kernels (kernels/) - built from xdna1-llm's designs with mlir-aie (Apache License 2.0 with LLVM
    exceptions, https://github.com/Xilinx/mlir-aie) and llvm-aie (same licence, https://github.com/Xilinx/llvm-aie)
"""


def nuget_packages(assets: Path, root: Path) -> list[tuple[str, str, str, list[Path]]]:
    """(id, version, licence, licence files) of the packages whose files the app ships."""
    data = json.loads(assets.read_text(encoding="utf-8"))
    out = []
    for target, libs in data["targets"].items():
        if "/" not in target:  # the RID-specific target is what publish copies
            continue
        for key, lib in sorted(libs.items()):
            if lib.get("type") != "package" or not any(k in lib for k in ("runtime", "native", "runtimeTargets")):
                continue
            name, version = key.split("/")
            folder = root / name.lower() / version.lower()
            files = [f for f in folder.iterdir() if re.match(r"(?i)(licen[cs]e|notice)", f.name) and f.is_file()]
            out.append((name, version, _nuspec_license(name, folder / f"{name.lower()}.nuspec", files), files))
    return out


def _nuspec_license(name: str, path: Path, files: list[Path]) -> str:
    try:
        meta = next(c for c in ET.parse(path).getroot() if c.tag.endswith("metadata"))
    except (OSError, ET.ParseError, StopIteration):
        return "see the package"
    tags = {c.tag.split("}")[-1]: c for c in meta}
    lic, project = tags.get("license"), tags.get("projectUrl")
    if lic is not None and lic.get("type") == "file":
        return f"licenses/{name}/{lic.text.strip()}"
    text = lic.text.strip() if lic is not None and lic.text else "see the package"
    if not files and project is not None and project.text:
        text += f", {project.text.strip()}"  # no licence file in the package: point to the project
    return text


def python_packages(site: Path) -> list[tuple[str, str, str]]:
    out = []
    for dist in metadata.distributions(path=[str(site)]):
        m = dist.metadata
        lic = m.get("License-Expression") or ""
        if not lic:
            classifiers = [c.split(" :: ")[-1] for c in m.get_all("Classifier") or [] if c.startswith("License ::")]
            lic = ", ".join(classifiers)
        if not lic and m.get("License"):
            lic = m["License"].strip().splitlines()[0]
        out.append((m["Name"], dist.version, lic or "see its dist-info"))
    return sorted(out, key=lambda p: p[0].lower())


def main(bundle: str, assets: str, nuget_root: str):
    out = Path(bundle)
    lic_dir = out / "licenses"
    lic_dir.mkdir(exist_ok=True)
    lines = [HEADER.format(python=sys.version.split()[0])]

    lines.append("App (.NET packages; licence texts in licenses/<package>/)")
    for name, version, lic, files in nuget_packages(Path(assets), Path(nuget_root)):
        lines.append(f"  {name} {version} - {lic}")
        if files:
            (lic_dir / name).mkdir(exist_ok=True)
            for f in files:
                shutil.copy2(f, lic_dir / name / f.name)

    lines.append("\nEngine (Python packages in python/Lib/site-packages; licence texts in their *.dist-info)")
    for name, version, lic in python_packages(out / "python" / "Lib" / "site-packages"):
        lines.append(f"  {name} {version} - {lic}")
    (out / "THIRD-PARTY-NOTICES.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:4])
