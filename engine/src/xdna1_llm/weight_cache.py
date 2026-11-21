"""Device-ready weight layouts on disk, so loading a model again skips converting its weights.

The NPU backends turn GGUF weights into the exact bytes their device buffers hold: the GEMV repacks Q4_K / Q6_K
into its tile layout (which the W4A16 prompt GEMM reads too); the bf16 prompt GEMM, when used, dequantizes to bf16
and transposes. That takes most of a load (~7 s for LFM3-1.7B, ~35 s with the bf16 GEMM). The cache keeps each
result as an .npy file and maps it back on the next load:

    %LOCALAPPDATA%/xdna1-llm/cache/<model>-<key>/<backend>/<weight>.npy

The key covers the GGUF (its size and the bytes at both ends, so copies of a file share one cache) and the code
that defines the layouts, so a changed model or layout never reads stale bytes; older entries under the same
file name are deleted.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import numpy as np

# The modules whose code decides the cached bytes: any change to them starts a new cache.
_LAYOUT_MODULES = ("bf16.py", "gguf_weights.py", "q4k.py", "q6k.py", "npu/linear.py", "npu/qlinear.py")
_ENDS = 4 << 20  # bytes hashed at each end of the GGUF to tell models apart


def default_dir() -> Path:
    """$XDNA1_LLM_CACHE, else %LOCALAPPDATA%/xdna1-llm/cache."""
    if os.environ.get("XDNA1_LLM_CACHE"):
        return Path(os.environ["XDNA1_LLM_CACHE"])
    return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "xdna1-llm" / "cache"


def _layout_hash() -> str:
    h = hashlib.sha256()
    pkg = Path(__file__).parent
    for m in _LAYOUT_MODULES:
        h.update((pkg / m).read_bytes())
    return h.hexdigest()


class WeightCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    @classmethod
    def for_model(cls, gguf: str, base: str | Path | None = None) -> WeightCache:
        h = hashlib.sha256(_layout_hash().encode())
        size = os.path.getsize(gguf)
        h.update(str(size).encode())
        with open(gguf, "rb") as f:  # the file's ends (header, metadata, last tensors): copies share a cache
            h.update(f.read(_ENDS))
            f.seek(max(0, size - _ENDS))
            h.update(f.read(_ENDS))
        stem = Path(gguf).stem
        base = Path(base) if base else default_dir()
        root = base / f"{stem}-{h.hexdigest()[:12]}"
        if base.is_dir():  # drop what earlier versions of this model or of the layouts left
            for old in base.glob(f"{stem}-*"):
                if old != root and old.is_dir() and len(old.name) == len(root.name):
                    shutil.rmtree(old, ignore_errors=True)
        return cls(root)

    def get(self, key: str, make) -> np.ndarray:
        """The array cached under key (memory-mapped, read-only), else make() saved under key and returned."""
        path = self.root / (key + ".npy")
        try:
            return np.load(path, mmap_mode="r")
        except (OSError, ValueError):
            pass
        arr = make()
        tmp = path.with_suffix(".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                np.save(f, arr)
            os.replace(tmp, path)  # never a half-written entry, even if the load is cut short
        except OSError:  # a full or read-only disk only costs speed
            tmp.unlink(missing_ok=True)
        return arr
