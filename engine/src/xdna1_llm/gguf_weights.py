"""Memory-mapped access to GGUF tensors and metadata (via the `gguf` package)."""

from __future__ import annotations

import numpy as np
from gguf import GGUFReader

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
        raise NotImplementedError(
            f"{name}: tensor type {kind} not supported yet (BF16/F32 only; Q4_K support is on the roadmap)"
        )

    def quant(self, name: str) -> tuple[str, np.ndarray]:
        """A quantized tensor as (type name, uint8 (rows, bytes per row)), still in GGUF block format."""
        t = self.tensors[name]
        return t.tensor_type.name, np.asarray(t.data).view(np.uint8).reshape(int(t.shape[-1]), -1)

    def f32(self, name: str) -> np.ndarray:
        a = self.raw(name)
        return from_bf16(a) if a.dtype == np.uint16 else a
