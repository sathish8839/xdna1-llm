"""Turns the placeholder xclbins of a deferred Linux build into real ones, with the Windows XRT SDK's xclbinutil.

    python kernels/package_xclbins.py <calls dir> <kernels dir> <path to xclbinutil.exe>

kernels/xclbinutil-defer recorded each xclbinutil call under <calls dir>; every .xclbin in <kernels dir> that is
still a placeholder is rebuilt from its call. Fails if a placeholder has no call, or a call fails.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PLACEHOLDER = b"xdna1-llm deferred xclbin "


def resolve(o, call: Path):
    if isinstance(o, dict):
        return {k: resolve(v, call) for k, v in o.items()}
    if isinstance(o, list):
        return [resolve(v, call) for v in o]
    if isinstance(o, str) and o.startswith("@@"):
        return str(call / o[2:])
    return o


def package(calls: Path, kernels: Path, xclbinutil: str) -> int:
    # Absolute paths throughout: xclbinutil runs inside each call folder.
    calls, kernels = calls.resolve(), kernels.resolve()
    if Path(xclbinutil).exists():
        xclbinutil = str(Path(xclbinutil).resolve())
    done = 0
    for xclbin in sorted(kernels.glob("*.xclbin")):
        head = xclbin.read_bytes()[:64]
        if not head.startswith(PLACEHOLDER):
            continue
        call = calls / head[len(PLACEHOLDER) :].decode().strip()
        spec = json.loads((call / "args.json").read_text(encoding="utf-8"))
        for js in call.glob("*.json"):  # point the PDI paths inside the JSON sections at the kept copies
            if js.name != "args.json":
                js.write_text(json.dumps(resolve(json.loads(js.read_text(encoding="utf-8")), call)), encoding="utf-8")
        out = xclbin.with_suffix(".xclbin.tmp")
        args = []
        for a in spec["args"]:
            if a == "@@OUTPUT":
                a = str(out)
            elif "@@" in a:
                a = a.replace("@@", "", 1)  # relative to cwd (the call folder): no drive colon in SECTION:FORMAT:PATH
            args.append(a)
        r = subprocess.run([xclbinutil, *args], cwd=call, capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            sys.exit(f"{xclbin.name}: xclbinutil failed ({r.returncode})\n{r.stdout}\n{r.stderr}")
        os.replace(out, xclbin)
        done += 1
        print(f"packaged {xclbin.name}")
    return done


if __name__ == "__main__":
    n = package(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
    print(f"{n} xclbins packaged")
