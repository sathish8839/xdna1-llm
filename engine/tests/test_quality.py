"""Accuracy metrics: known values on small distributions, and all-position prefill logits."""

import numpy as np
from test_decode import TinyLfm3Weights

from xdna1_llm.cpu_linear import CpuLinear
from xdna1_llm.models.lfm3 import LFM3
from xdna1_llm.quality import Score, chunks, log_softmax


def test_identical_models_score_zero_divergence():
    lp = log_softmax(np.random.default_rng(0).standard_normal((6, 10)))
    s = Score()
    s.add(lp, lp, [1, 2, 3, 4, 5, 6])
    r = s.summary()
    assert r["tokens"] == 6 and r["top1"] == 1.0
    assert abs(r["kld"]) < 1e-12


def test_known_values():
    ref = np.log(np.array([[0.5, 0.5]]))
    lp = np.log(np.array([[0.25, 0.75]]))
    s = Score()
    s.add(ref, lp, [0])
    r = s.summary()
    assert np.isclose(r["ppl"], 4.0)  # -log(0.25) -> exp = 4
    assert np.isclose(r["kld"], 0.5 * np.log(0.5 / 0.25) + 0.5 * np.log(0.5 / 0.75))


def test_chunks_start_with_bos():
    ids = [99, *range(10)]
    c = chunks(ids, 4)
    assert c == [[99, 0, 1, 2], [99, 3, 4, 5], [99, 6, 7, 8]]  # [99, 9] is shorter than half a chunk
    assert chunks(ids, 4, max_tokens=5) == [[99, 0, 1, 2], [99, 3, 4]]


def test_prefill_all_logits_match_decode():
    m = LFM3(TinyLfm3Weights(), CpuLinear())
    ids = [3, 1, 4, 1, 5, 9]
    logits, _ = m.forward(ids, all_logits=True)
    st = m.new_state(len(ids))
    steps = np.stack([m.step(t, st) for t in ids])
    assert logits.shape == steps.shape
    np.testing.assert_allclose(logits, steps, rtol=1e-4, atol=1e-4)
