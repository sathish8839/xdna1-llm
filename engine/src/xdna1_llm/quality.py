"""Accuracy metrics: how far one model's next-token predictions are from a reference model's.

For every position of a text both models predict the next token. Per comparison:
  * perplexity of the text under the model (exp of the mean negative log-likelihood; lower is better);
  * KL divergence of the model's distribution from the reference's, mean and 99th percentile (0 = identical);
  * top-1 agreement: how often both models rank the same token first.
These are the measures llama.cpp uses to judge quantizations (`llama-perplexity --kl-divergence`).
benchmarks/eval_quality.py runs them on real GGUFs.
"""

from __future__ import annotations

import numpy as np


def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


class Score:
    """Accumulates the metrics of one model against a reference, chunk by chunk."""

    def __init__(self):
        self.nll, self.kld, self.agree = [], [], []

    def add(self, ref_lp: np.ndarray, lp: np.ndarray, targets) -> None:
        """ref_lp, lp: (T, vocab) log-probabilities; targets: the T tokens that actually came next."""
        targets = np.asarray(targets)
        self.nll.append(-lp[np.arange(len(targets)), targets])
        self.kld.append(np.sum(np.exp(ref_lp) * (ref_lp - lp), axis=-1))
        self.agree.append(ref_lp.argmax(-1) == lp.argmax(-1))

    def summary(self) -> dict:
        nll, kld, agree = (np.concatenate(a) for a in (self.nll, self.kld, self.agree))
        return {
            "tokens": len(nll),
            "ppl": float(np.exp(nll.mean())),
            "kld": float(kld.mean()),
            "kld_p99": float(np.percentile(kld, 99)),
            "top1": float(agree.mean()),
        }


def chunks(body: list[int], bos: int, size: int, max_tokens: int | None = None) -> list[list[int]]:
    """Cut a tokenized text (without BOS) into chunks of `size` tokens that each start with BOS, so every chunk is
    scored from an empty context. A trailing piece shorter than half a chunk is dropped."""
    if max_tokens:
        body = body[:max_tokens]
    out = [[bos, *body[i : i + size - 1]] for i in range(0, len(body), size - 1)]
    return [c for c in out if len(c) > size // 2 or len(out) == 1]
