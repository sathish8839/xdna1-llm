"""LiquidAI LFM3 (GGUF arch "lfm3"), e.g. LFM3-1.7B: 24 layers = 13 gated short-conv + 6 full + 5 sliding-window GQA.

Forward-pass math (checked against llama.cpp output, see docs/benchmarks.md):
  block:      h += RMS(Op(RMS(h, attn_norm)), post_attention_norm)
              h += RMS(down(silu(gate(u)) * up(u)), post_ffw_norm),  u = RMS(h, ffn_norm)
  short conv: [B, C, x] = in_proj(u); y = causal_depthwise_conv7(B * x); out_proj(C * y)
  attention:  per-head RMSNorm on q, k (before RoPE); NeoX RoPE; full layers rotate 32 of 64 pairs (base 1e6),
              sliding layers rotate all 64 (base 1e4, window 256); GQA 16q/4kv; per-head sigmoid gate on the
              attention output before o_proj; scale 1/sqrt(128)
  head:       RMS(h, output_norm) @ token_embd^T (tied)
Linear layers go through the backend; everything else runs here in numpy (fp32).

`forward` is prefill (all prompt tokens at once). `step` is decode: one token, with the attention keys/values
and the last 6 short-conv inputs of each layer carried in a `State`.
"""

from __future__ import annotations

import time
from functools import partial

import numpy as np

from ..gguf_weights import dequantize

EPS, LCONV, SWA_WINDOW = 1e-5, 7, 256


def rms(x, w):
    x = x.astype(np.float32, copy=False)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope_tables(T, head_dim, base, rot_pairs, start=0):
    """NeoX (rotate_half) tables for positions start..start+T-1; pairs i >= rot_pairs get angle 0 (partial
    rotary)."""
    inv = base ** (-np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    inv[rot_pairs:] = 0.0
    ang = np.arange(start, start + T, dtype=np.float64)[:, None] * inv[None, :]
    ang = np.concatenate([ang, ang], axis=1)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def apply_rope(x, cos, sin):  # x (T, H, D)
    half = x.shape[-1] // 2
    rot = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


class State:
    """Decode state: the position, per attention layer the keys/values so far, per conv layer the last 6 inputs."""

    def __init__(self, model: LFM3, max_len: int):
        self.pos, self.max_len = 0, max_len
        self.k, self.v, self.conv = {}, {}, {}
        for i in range(model.n_layers):
            if model.kind(i) == "conv":
                self.conv[i] = np.zeros((LCONV - 1, model.d), np.float32)
            else:
                self.k[i] = np.zeros((max_len, model.kv_heads[i], model.hd), np.float32)
                self.v[i] = np.zeros((max_len, model.kv_heads[i], model.hd), np.float32)


class LFM3:
    def __init__(self, w, lin):
        if w.arch != "lfm3":
            raise ValueError(f"expected an lfm3 GGUF, got {w.arch!r}")
        self.w, self.lin = w, lin
        a = "lfm3."
        self.n_layers = int(w.scalar(a + "block_count"))
        self.d = int(w.scalar(a + "embedding_length"))
        self.ff = int(w.scalar(a + "feed_forward_length"))
        self.nh = int(w.scalar(a + "attention.head_count"))
        self.hd = int(w.scalar(a + "attention.key_length"))
        self.kv_heads = [int(x) for x in w.array(a + "attention.head_count_kv")]
        self.swa = [bool(x) for x in w.array(a + "attention.sliding_window_pattern")]
        self.base_full = float(w.scalar(a + "rope.freq_base"))
        self.base_swa = float(w.scalar(a + "rope.freq_base_swa"))
        rf = w.f32("rope_freqs.weight")  # 1.0 = rotate, 1e30 = frozen pair
        self.rot_pairs_full = int(np.sum(rf < 1e10))
        self.window = SWA_WINDOW

        self.p: dict[str, np.ndarray] = {}
        t0 = time.perf_counter()
        for i in range(self.n_layers):
            b = f"blk.{i}."
            for n in ("attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm"):
                self.p[b + n] = w.f32(b + n + ".weight")
            if self.kind(i) == "conv":
                lin.register(b + "in_proj", w.tensor(b + "shortconv.in_proj.weight"))
                lin.register(b + "out_proj", w.tensor(b + "shortconv.out_proj.weight"))
                self.p[b + "conv"] = w.f32(b + "shortconv.conv.weight")  # (D, 7), k=6 is the newest token
            else:
                qkv = [w.tensor(b + f"attn_{n}.weight") for n in "qkv"]
                lin.register(b + "qkv", qkv)  # one weight with stacked rows: q (nh*hd), k, v (nkv*hd each)
                lin.register(b + "o", w.tensor(b + "attn_output.weight"))
                self.p[b + "gate"] = w.f32(b + "attn_gate.weight")  # (H, D)
                self.p[b + "q_norm"] = w.f32(b + "attn_q_norm.weight")
                self.p[b + "k_norm"] = w.f32(b + "attn_k_norm.weight")
            lin.register(b + "gateup", [w.tensor(b + "ffn_gate.weight"), w.tensor(b + "ffn_up.weight")])
            lin.register(b + "down", w.tensor(b + "ffn_down.weight"))
        self.out_norm = w.f32("output_norm.weight")
        self.embd = w.tensor("token_embd.weight")  # (type, data); rows are tokens
        if lin.wants_head:
            lin.register("lm_head", self.embd)
        else:
            self.lm_head = dequantize(*self.embd)  # fp32 copy of the tied head
        self.load_s = time.perf_counter() - t0
        self.cpu_ms = 0.0

    def kind(self, i: int) -> str:
        if self.kv_heads[i] == 0:
            return "conv"
        return "swa" if self.swa[i] else "full"

    def embed(self, ids) -> np.ndarray:
        kind, data = self.embd
        return dequantize(kind, data[np.asarray(ids)])

    def head(self, h) -> np.ndarray:
        """Final norm and tied LM head for the rows of h."""
        t0 = time.perf_counter()
        u = rms(h, self.out_norm)
        self.cpu_ms += (time.perf_counter() - t0) * 1e3
        if self.lin.wants_head:
            return self.lin("lm_head", u)
        t0 = time.perf_counter()
        logits = u @ self.lm_head.T
        self.cpu_ms += (time.perf_counter() - t0) * 1e3
        return logits

    def rope(self, kind, T, start=0):
        if kind == "full":
            return rope_tables(T, self.hd, self.base_full, self.rot_pairs_full, start)
        return rope_tables(T, self.hd, self.base_swa, self.hd // 2, start)

    # ---- prefill
    def _conv(self, b, u):
        P = self.lin(b + "in_proj", u)
        t0 = time.perf_counter()
        d = self.d
        Bg, Cg, X = P[:, :d], P[:, d : 2 * d], P[:, 2 * d :]
        z = Bg * X
        wc = self.p[b + "conv"]
        zp = np.concatenate([np.zeros((LCONV - 1, d), np.float32), z], 0)
        y = np.zeros_like(z)
        for k in range(LCONV):
            y += wc[:, k][None, :] * zp[k : k + z.shape[0]]
        g = Cg * y
        self.cpu_ms += (time.perf_counter() - t0) * 1e3
        return self.lin(b + "out_proj", g)

    def _attn(self, b, u, kind):
        T, nh, hd = u.shape[0], self.nh, self.hd
        nkv = self.kv_heads[int(b.split(".")[1])]
        qkv = self.lin(b + "qkv", u)
        t0 = time.perf_counter()
        q = rms(qkv[:, : nh * hd].reshape(T, nh, hd), self.p[b + "q_norm"])
        k = rms(qkv[:, nh * hd : (nh + nkv) * hd].reshape(T, nkv, hd), self.p[b + "k_norm"])
        v = qkv[:, (nh + nkv) * hd :].reshape(T, nkv, hd)
        cos, sin = self.rope(kind, T)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = nh // nkv
        k, v = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
        s = np.matmul(q.transpose(1, 0, 2), k.transpose(1, 2, 0)) * np.float32(1.0 / np.sqrt(hd))
        i, j = np.arange(T)[:, None], np.arange(T)[None, :]
        mask = j > i
        if kind == "swa":
            mask |= (i - j) >= self.window
        s = np.where(mask[None], -np.inf, s)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = np.matmul(s, v.transpose(1, 0, 2)).transpose(1, 0, 2)
        gate = 1.0 / (1.0 + np.exp(-(u @ self.p[b + "gate"].T)))
        o = (o * gate[:, :, None]).reshape(T, nh * hd)
        self.cpu_ms += (time.perf_counter() - t0) * 1e3
        return self.lin(b + "o", o)

    def _block(self, b, h, mixer):
        """One layer around a token mixer (conv or attention): norms, residuals and the SwiGLU FFN."""
        t0 = time.perf_counter()
        u = rms(h, self.p[b + "attn_norm"])
        self.cpu_ms += (time.perf_counter() - t0) * 1e3
        o = mixer(u)
        t0 = time.perf_counter()
        h = h + rms(o, self.p[b + "post_attention_norm"])
        u = rms(h, self.p[b + "ffn_norm"])
        self.cpu_ms += (time.perf_counter() - t0) * 1e3
        gu = self.lin(b + "gateup", u)
        t0 = time.perf_counter()
        act = silu(gu[:, : self.ff]) * gu[:, self.ff :]
        self.cpu_ms += (time.perf_counter() - t0) * 1e3
        f = self.lin(b + "down", act)
        t0 = time.perf_counter()
        h = h + rms(f, self.p[b + "post_ffw_norm"])
        self.cpu_ms += (time.perf_counter() - t0) * 1e3
        return h

    def forward(self, ids, keep_hidden: bool = False, all_logits: bool = False):
        """Prefill: returns (logits of the last position, or of every position if all_logits, and the list of
        per-layer hidden states if keep_hidden)."""
        h = self.embed(ids)
        hidden = []
        for i in range(self.n_layers):
            b, kind = f"blk.{i}.", self.kind(i)
            if kind == "conv":
                h = self._block(b, h, partial(self._conv, b))
            else:
                h = self._block(b, h, partial(self._attn, b, kind=kind))
            if keep_hidden:
                hidden.append(h.copy())
        return (self.head(h) if all_logits else self.head(h[-1:])[0]), hidden

    # ---- decode
    def new_state(self, max_len: int = 4096) -> State:
        return State(self, max_len)

    def _conv_step(self, b, i, u, st):
        P = self.lin(b + "in_proj", u)[0]
        t0 = time.perf_counter()
        d = self.d
        zp = np.concatenate([st.conv[i], (P[:d] * P[2 * d :])[None]], 0)  # (7, D), newest last = tap 6
        y = np.einsum("kd,dk->d", zp, self.p[b + "conv"])
        st.conv[i] = zp[1:]
        g = P[d : 2 * d] * y
        self.cpu_ms += (time.perf_counter() - t0) * 1e3
        return self.lin(b + "out_proj", g[None])

    def _attn_step(self, b, i, u, kind, st):
        nh, hd, nkv, pos = self.nh, self.hd, self.kv_heads[i], st.pos
        qkv = self.lin(b + "qkv", u)[0]
        t0 = time.perf_counter()
        q = rms(qkv[: nh * hd].reshape(1, nh, hd), self.p[b + "q_norm"])
        k = rms(qkv[nh * hd : (nh + nkv) * hd].reshape(1, nkv, hd), self.p[b + "k_norm"])
        cos, sin = self.rope(kind, 1, pos)
        st.k[i][pos] = apply_rope(k, cos, sin)[0]
        st.v[i][pos] = qkv[(nh + nkv) * hd :].reshape(nkv, hd)
        q = apply_rope(q, cos, sin)[0].reshape(nkv, nh // nkv, hd)  # GQA: query heads grouped by kv head
        lo = max(0, pos - self.window + 1) if kind == "swa" else 0
        K, V = st.k[i][lo : pos + 1], st.v[i][lo : pos + 1]  # (L, nkv, hd)
        s = np.einsum("grd,lgd->grl", q, K) * np.float32(1.0 / np.sqrt(hd))
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = np.einsum("grl,lgd->grd", s, V).reshape(nh, hd)
        gate = 1.0 / (1.0 + np.exp(-(u[0] @ self.p[b + "gate"].T)))
        o = (o * gate[:, None]).reshape(1, nh * hd)
        self.cpu_ms += (time.perf_counter() - t0) * 1e3
        return self.lin(b + "o", o)

    def step(self, token: int, st: State) -> np.ndarray:
        """Decode one token at position st.pos; returns the next-token logits."""
        if st.pos >= st.max_len:
            raise ValueError(f"context full ({st.max_len} tokens)")
        h = self.embed([token])
        for i in range(self.n_layers):
            b, kind = f"blk.{i}.", self.kind(i)
            if kind == "conv":
                h = self._block(b, h, partial(self._conv_step, b, i, st=st))
            else:
                h = self._block(b, h, partial(self._attn_step, b, i, kind=kind, st=st))
        st.pos += 1
        return self.head(h)[0]

    def generate(self, ids, n: int, st: State | None = None, on_token=None, logits_out: list | None = None, stop=()):
        """Greedy decoding: feed the prompt one token at a time, then generate up to n tokens, ending early after a
        token in `stop` (end-of-generation ids). Returns (new token ids, state, prompt seconds, generation seconds)."""
        st = st or self.new_state()
        t0 = time.perf_counter()
        for t in ids:
            logits = self.step(int(t), st)
        t1 = time.perf_counter()
        out = []
        while True:
            if logits_out is not None:
                logits_out.append(logits)
            out.append(int(np.argmax(logits)))
            if on_token:
                on_token(out[-1])
            if len(out) == n or out[-1] in stop:
                break
            logits = self.step(out[-1], st)
        return out, st, t1 - t0, time.perf_counter() - t1
