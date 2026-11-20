"""Reference linear backend: fp32 matmul on the same bf16 weights the NPU sees (no pyxrt needed)."""

from __future__ import annotations

import numpy as np

from .bf16 import from_bf16


class CpuLinear:
    def __init__(self):
        self.mats: dict[str, np.ndarray] = {}
        self.npu_ms, self.calls = 0.0, 0

    def register(self, name: str, wbits: np.ndarray):
        self.mats[name] = wbits  # keep bf16 bits; decode per call to save RAM

    def __call__(self, name: str, x: np.ndarray) -> np.ndarray:
        return x.astype(np.float32) @ from_bf16(self.mats[name]).T
