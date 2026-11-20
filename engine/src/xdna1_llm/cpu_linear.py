"""Reference linear backend: fp32 matmul on the same weights the NPU sees (no pyxrt needed).

Quantized weights are dequantized on every call to keep RAM low, so this is slow for Q4_K / Q6_K models: it is
meant for checking results, not for running them.
"""

from __future__ import annotations

import numpy as np

from .gguf_weights import as_parts, dequantize


class CpuLinear:
    wants_head = True  # the model may route the LM head through this backend

    def __init__(self):
        self.mats: dict[str, list] = {}
        self.npu_ms, self.calls = 0.0, 0

    def register(self, name: str, w):
        self.mats[name] = as_parts(w)

    def __call__(self, name: str, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        return np.concatenate([x @ dequantize(kind, data).T for kind, data in self.mats[name]], axis=-1)
