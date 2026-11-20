"""Decode (one token at a time, with state) must give the same logits as prefill over the same prefix."""

import numpy as np

from xdna1_llm.bf16 import to_bf16
from xdna1_llm.cpu_linear import CpuLinear
from xdna1_llm.models.lfm3 import LCONV, LFM3


class TinyLfm3Weights:
    """A random LFM3 with every layer type: conv, sliding-window attention, conv, full attention."""

    arch = "lfm3"

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        d, ff, nh, hd, vocab = 32, 64, 4, 8, 50
        kv = [0, 2, 0, 2]
        self.meta = {
            "block_count": len(kv),
            "embedding_length": d,
            "feed_forward_length": ff,
            "attention.head_count": nh,
            "attention.key_length": hd,
            "attention.head_count_kv": kv,
            "attention.sliding_window_pattern": [False, True, False, False],
            "rope.freq_base": 1e6,
            "rope.freq_base_swa": 1e4,
        }
        lin = lambda o, i: to_bf16(rng.standard_normal((o, i)).astype(np.float32) / np.sqrt(i))  # noqa: E731
        vec = lambda n: (1 + 0.1 * rng.standard_normal(n)).astype(np.float32)  # noqa: E731
        t = {"rope_freqs.weight": np.array([1, 1, 1e30, 1e30], np.float32), "output_norm.weight": vec(d)}
        t["token_embd.weight"] = lin(vocab, d)
        for i, nkv in enumerate(kv):
            b = f"blk.{i}."
            for n in ("attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm"):
                t[b + n + ".weight"] = vec(d)
            if nkv == 0:
                t[b + "shortconv.in_proj.weight"] = lin(3 * d, d)
                t[b + "shortconv.out_proj.weight"] = lin(d, d)
                t[b + "shortconv.conv.weight"] = rng.standard_normal((d, LCONV)).astype(np.float32) * 0.3
            else:
                t[b + "attn_q.weight"] = lin(nh * hd, d)
                t[b + "attn_k.weight"] = lin(nkv * hd, d)
                t[b + "attn_v.weight"] = lin(nkv * hd, d)
                t[b + "attn_output.weight"] = lin(d, nh * hd)
                t[b + "attn_gate.weight"] = rng.standard_normal((nh, d)).astype(np.float32) * 0.2
                t[b + "attn_q_norm.weight"] = vec(hd)
                t[b + "attn_k_norm.weight"] = vec(hd)
            t[b + "ffn_gate.weight"] = lin(ff, d)
            t[b + "ffn_up.weight"] = lin(ff, d)
            t[b + "ffn_down.weight"] = lin(d, ff)
        self.t = t

    def scalar(self, key):
        return self.meta[key.removeprefix("lfm3.")]

    array = scalar

    def f32(self, name):
        return self.t[name]

    def tensor(self, name):
        return ("BF16", self.t[name])


def test_decode_matches_prefill_at_every_position():
    m = LFM3(TinyLfm3Weights(), CpuLinear())
    m.window = 3  # small, so the sliding-window layer actually drops old keys
    ids = np.random.default_rng(1).integers(0, 50, size=9)
    st = m.new_state(max_len=16)
    for t, tok in enumerate(ids):
        step_logits = m.step(int(tok), st)
        prefill_logits, _ = m.forward(ids[: t + 1])
        np.testing.assert_allclose(step_logits, prefill_logits, rtol=1e-4, atol=1e-5)
    assert st.pos == len(ids)


def test_generate_is_greedy_over_step():
    m = LFM3(TinyLfm3Weights(seed=2), CpuLinear())
    out, st, _, _ = m.generate([3, 7, 11], 4)
    st2 = m.new_state()
    logits = [m.step(t, st2) for t in (3, 7, 11)][-1]
    expect = []
    for _ in range(4):
        expect.append(int(np.argmax(logits)))
        logits = m.step(expect[-1], st2)
    assert out == expect and st.pos == 3 + 3
