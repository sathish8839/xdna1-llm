"""Memory-mapped access to GGUF tensors and metadata (via the `gguf` package)."""

from __future__ import annotations

import numpy as np
from gguf import GGUFReader

from . import q4k, q6k
from .bf16 import from_bf16


class GGUFWeights:
    """Linear weights come back as bf16 bits with numpy shape (out, in); norms/conv as float32."""

    def __init__(self, path: str):
        self.path = path
        self.reader = GGUFReader(path)
        self.tensors = {t.name: t for t in self.reader.tensors}
        self.fields = self.reader.fields
        self.arch = self.string("general.architecture")

    # ---- metadata
    def string(self, key: str) -> str:
        f = self.fields[key]
        return bytes(f.parts[f.data[0]]).decode()

    def scalar(self, key: str):
        f = self.fields[key]
        return f.parts[f.data[0]][0]

    def array(self, key: str) -> list:
        f = self.fields[key]
        return [f.parts[i][0] for i in f.data]

    def tokens(self) -> list[str]:
        f = self.fields["tokenizer.ggml.tokens"]
        return [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]

    # ---- tensors
    def types(self) -> set[str]:
        """Tensor types in the file, for example {"F32", "Q4_K", "Q6_K"} for a Q4_K_M model."""
        return {t.tensor_type.name for t in self.tensors.values()}

    def has(self, name: str) -> bool:
        return name in self.tensors

    def raw(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        shape = tuple(int(s) for s in reversed(t.shape))  # ggml [ne0, ne1] -> numpy (ne1, ne0)
        kind = t.tensor_type.name
        if kind == "BF16":
            return np.asarray(t.data).view(np.uint16).reshape(shape)
        if kind == "F32":
            return np.asarray(t.data, dtype=np.float32).reshape(shape)
        raise NotImplementedError(f"{name}: {kind} is quantized; use tensor() or f32()")

    def quant(self, name: str) -> tuple[str, np.ndarray]:
        """A quantized tensor as (type name, uint8 (rows, bytes per row)), still in GGUF block format."""
        t = self.tensors[name]
        return t.tensor_type.name, np.asarray(t.data).view(np.uint8).reshape(int(t.shape[-1]), -1)

    def tensor(self, name: str) -> tuple[str, np.ndarray]:
        """A 2D weight as (type, data): BF16 as bf16 bits (out, in), Q4_K / Q6_K as GGUF blocks (out, bytes)."""
        kind = self.tensors[name].tensor_type.name
        return (kind, self.raw(name)) if kind in ("BF16", "F32") else self.quant(name)

    def f32(self, name: str) -> np.ndarray:
        """Any supported tensor as float32 (quantized ones dequantized), numpy shape (ne1, ne0)."""
        return dequantize(*self.tensor(name))


def dequantize(kind: str, data: np.ndarray) -> np.ndarray:
    """(type, data) from GGUFWeights.tensor -> float32 (out, in)."""
    if kind == "BF16":
        return from_bf16(data)
    if kind == "F32":
        return data
    if kind == "Q4_K":
        return q4k.dequantize(data)
    if kind == "Q6_K":
        return q6k.dequantize(data)
    if kind == "Q8_0":  # blocks of 32: fp16 scale, 32 int8 (CPU reference only, no NPU kernel)
        b = data.reshape(data.shape[0], -1, 34)
        d = b[..., :2].copy().view(np.float16).astype(np.float32)
        return (d * b[..., 2:].view(np.int8)).reshape(data.shape[0], -1)
    raise NotImplementedError(f"tensor type {kind} not supported (BF16, F32, Q4_K, Q6_K, Q8_0)")


def as_parts(w) -> list[tuple[str, np.ndarray]]:
    """A weight given to a linear backend: bf16 bits, one (type, data) tensor, or a list of tensors whose rows
    are stacked (for example k and v)."""
    if isinstance(w, np.ndarray):
        return [("BF16", w)]
    if isinstance(w, tuple):
        return [w]
    return list(w)
