"""Sampler: each filter on small hand-made logits, plus the draw's statistics."""

import numpy as np
import pytest

from xdna1_llm.sampling import Sampler

LOGITS = np.log(np.array([0.5, 0.3, 0.15, 0.05]))  # probabilities 0.5 / 0.3 / 0.15 / 0.05


def draws(s: Sampler, logits=LOGITS, n=4000, history=()):
    return np.bincount([s(logits, history) for _ in range(n)], minlength=len(logits)) / n


def test_temperature_zero_is_greedy():
    assert Sampler()(LOGITS) == 0
    assert Sampler(temperature=0, repetition_penalty=10.0)(np.array([2.0, 1.9, 0.0]), history=[0]) == 1


def test_temperature_one_follows_the_probabilities():
    np.testing.assert_allclose(draws(Sampler(temperature=1.0, seed=0)), [0.5, 0.3, 0.15, 0.05], atol=0.03)


def test_top_k_top_p_min_p_cut_the_tail():
    assert set(np.nonzero(draws(Sampler(temperature=1.0, top_k=2, seed=0)))[0]) == {0, 1}
    assert set(np.nonzero(draws(Sampler(temperature=1.0, top_p=0.8, seed=0)))[0]) == {0, 1}  # 0.5 + 0.3 reaches 0.8
    assert set(np.nonzero(draws(Sampler(temperature=1.0, top_p=0.81, seed=0)))[0]) == {0, 1, 2}
    assert set(np.nonzero(draws(Sampler(temperature=1.0, min_p=0.5, seed=0)))[0]) == {0, 1}  # >= 0.25


def test_low_temperature_sharpens():
    assert draws(Sampler(temperature=0.1, seed=0))[0] > 0.99


def test_repetition_penalty_divides_positive_and_multiplies_negative_logits():
    s = Sampler(temperature=0, repetition_penalty=2.0, penalty_last_n=2)
    assert s(np.array([3.0, 2.0, -1.0]), history=[0]) == 1  # 3 / 2 = 1.5 < 2
    assert s(np.array([3.0, 2.0, -1.0]), history=[0, 0, 2, 1]) == 0  # only the last 2 ids count
    assert s(np.array([-1.0, -1.5]), history=[0]) == 1  # -1 * 2 = -2 < -1.5


def test_seed_makes_runs_repeatable():
    s1, s2 = Sampler(temperature=1.0, seed=7), Sampler(temperature=1.0, seed=7)
    assert [s1(LOGITS) for _ in range(50)] == [s2(LOGITS) for _ in range(50)]


class FakeGGUF:
    def __init__(self, values):
        self.fields = {f"general.sampling.{k}": v for k, v in values.items()}

    def scalar(self, key):
        return self.fields[key]


@pytest.mark.parametrize("override", [{}, {"temperature": 0.7}])
def test_from_gguf_reads_recommended_values(override):
    s = Sampler.from_gguf(FakeGGUF({"top_k": 50, "temp": 0.1, "penalty_repeat": 1.1}), top_p=None, **override)
    assert (s.top_k, s.repetition_penalty, s.top_p) == (50, pytest.approx(1.1), 1.0)
    assert s.temperature == pytest.approx(override.get("temperature", 0.1))
