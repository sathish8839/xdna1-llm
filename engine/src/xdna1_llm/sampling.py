"""Picking the next token from the logits: greedy, or sampled with temperature, top-k, top-p and min-p.

The steps run in llama.cpp's order, each one narrowing the candidates the next one sees:
  1. repetition penalty: tokens among the last `penalty_last_n` become less likely (a positive logit is divided by
     the penalty, a negative one multiplied);
  2. top-k: keep the k highest logits;
  3. top-p: keep the smallest set whose probabilities add up to p;
  4. min-p: drop tokens less likely than min_p × the top token's probability;
  5. temperature: divide the logits (below 1 sharpens, above 1 flattens), then draw one token.
Temperature 0 means greedy (the highest logit after the penalty). LFM3 GGUFs carry recommended values
(`general.sampling.*`: temperature 0.1, top-k 50, repetition penalty 1.1), which `Sampler.from_gguf` uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Sampler:
    temperature: float = 0.0
    top_k: int = 0  # 0 = off
    top_p: float = 1.0  # 1 = off
    min_p: float = 0.0  # 0 = off
    repetition_penalty: float = 1.0  # 1 = off
    penalty_last_n: int = 64
    seed: int | None = None
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    @classmethod
    def from_gguf(cls, w, **overrides) -> Sampler:
        """The model's recommended settings, with any keyword overridden (None values are ignored)."""
        keys = {"temperature": "temp", "top_k": "top_k", "top_p": "top_p", "min_p": "min_p",
                "repetition_penalty": "penalty_repeat", "penalty_last_n": "penalty_last_n"}  # fmt: skip
        kw = {}
        for name, key in keys.items():
            if f"general.sampling.{key}" in w.fields:
                v = w.scalar(f"general.sampling.{key}")
                kw[name] = int(v) if name in ("top_k", "penalty_last_n") else round(float(v), 6)  # stored as f32
        kw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kw)

    def __call__(self, logits: np.ndarray, history=()) -> int:
        """The next token for these logits; `history` is the token ids so far (prompt and generated)."""
        logits = np.asarray(logits, dtype=np.float32)
        if self.repetition_penalty != 1.0 and self.penalty_last_n and len(history):
            recent = np.unique(np.asarray(history[-self.penalty_last_n :], dtype=np.int64))
            logits = logits.copy()
            v = logits[recent]
            logits[recent] = np.where(v > 0, v / self.repetition_penalty, v * self.repetition_penalty)
        if self.temperature <= 0:
            return int(np.argmax(logits))

        ids = np.arange(logits.size)
        if 0 < self.top_k < logits.size:
            ids = np.argpartition(-logits, self.top_k - 1)[: self.top_k]
        x = logits[ids].astype(np.float64)
        p = np.exp(x - x.max())
        p /= p.sum()
        if self.top_p < 1.0:
            order = _best_first(p, min(1024, p.size))  # sorting all 128K logits costs ~15 ms; the head is enough
            c = np.cumsum(p[order])
            if c[-1] < self.top_p:
                order = _best_first(p, p.size)
                c = np.cumsum(p[order])
            keep = min(int(np.searchsorted(c, self.top_p)) + 1, order.size)  # smallest set reaching top_p
            ids, x, p = ids[order[:keep]], x[order[:keep]], p[order[:keep]]
        if self.min_p > 0:
            keep = p >= self.min_p * p.max()
            ids, x = ids[keep], x[keep]
        x = x / self.temperature
        q = np.exp(x - x.max())
        return int(ids[self.rng.choice(ids.size, p=q / q.sum())])


def _best_first(p: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest values of p, largest first."""
    part = np.argpartition(-p, k - 1)[:k] if k < p.size else np.arange(p.size)
    return part[np.argsort(-p[part], kind="stable")]
