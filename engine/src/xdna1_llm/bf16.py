"""bfloat16 <-> float32 conversion on uint16 bit patterns (the NPU kernels take bf16 as raw uint16)."""

from __future__ import annotations

import numpy as np

try:
    import ml_dtypes

    def to_bf16(x: np.ndarray) -> np.ndarray:
        """float32 -> bf16 bits (uint16), round-to-nearest-even, implemented in C by ml_dtypes."""
        return np.asarray(x, dtype=np.float32).astype(ml_dtypes.bfloat16).view(np.uint16)

except ImportError:  # pragma: no cover - fallback without ml_dtypes

    def to_bf16(x: np.ndarray) -> np.ndarray:
        u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
        return ((u + (0x7FFF + ((u >> 16) & 1))) >> 16).astype(np.uint16)


def from_bf16(b: np.ndarray) -> np.ndarray:
    """bf16 bits (uint16) -> float32."""
    return (np.asarray(b).astype(np.uint32) << 16).view(np.float32)
