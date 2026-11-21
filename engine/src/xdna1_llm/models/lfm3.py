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

`forward` is prefill: many tokens per call, in chunks of up to 256. `step` is decode: one token. Both read and
extend a `State` (the attention keys/values and the last 6 short-conv inputs of each layer), so a prompt can be
prefilled in one batch and then continued token by token.

Two linear backends: `lin` serves decode (and prefill when there is no other), `prefill_lin` (optional) serves
prompt chunks of at least PREFILL_MIN tokens. For a Q4_K_M / Q6_K model on the NPU, `lin` is the quantized GEMV
(one token per call) and `prefill_lin` the W4A16 GEMM, which reads the GEMV's weights (or, if it is not built, the
bf16 GEMM, which makes a bf16 copy of every linear weight at load).
"""

from __future__ import annotations

import time
from functools import partial

import numpy as np

from ..gguf_weights import dequantize

EPS, LCONV, SWA_WINDOW = 1e-5, 7, 256
CHUNK = 256  # prefill rows per pass (the bf16 GEMM's tile)
PREFILL_MIN = 32  # shorter prompt chunks run on `lin`: the GEMM always computes 256 rows


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
    """Decode state: the position, per attention layer the keys/values so far, per conv layer the last 6 inputs.

    A state can go back to an earlier position, to reuse a shared prefix (the next chat turn starts with this
    one). Attention needs nothing for that: keys/values past the position are overwritten. The conv window only
    holds the latest inputs, so `snapshot` keeps a copy of it at chosen positions and `rewind` returns to one."""

    def __init__(self, model: LFM3, max_len: int):
        self.pos, self.max_len = 0, max_len
        self.k, self.v, self.conv = {}, {}, {}
        for i in range(model.n_layers):
            if model.kind(i) == "conv":
                self.conv[i] = np.zeros((LCONV - 1, model.d), np.float32)
            else:
                self.k[i] = np.zeros((max_len, model.kv_heads[i], model.hd), np.float32)
                self.v[i] = np.zeros((max_len, model.kv_heads[i], model.hd), np.float32)
        self.snaps = {0: {i: c.copy() for i, c in self.conv.items()}}  # position -> conv windows there

    def snapshot(self):
        """Remember the conv windows at the current position (about 0.6 MB for LFM3-1.7B)."""
        self.snaps[self.pos] = {i: c.copy() for i, c in self.conv.items()}

    def rewind(self, pos: int) -> int:
        """Go back to the latest position <= pos the state can return to; returns that position."""
        if pos >= self.pos:
            return self.pos
        p = max(q for q in self.snaps if q <= pos)
        self.conv = {i: c.copy() for i, c in self.snaps[p].items()}
        self.snaps = {q: s for q, s in self.snaps.items() if q <= p}
        self.pos = p
        return p


class LFM3:
    def __init__(self, w, lin, prefill_lin=None):
        if w.arch != "lfm3":
            raise ValueError(f"expected an lfm3 GGUF, got {w.arch!r}")
        self.w, self.lin, self.prefill_lin = w, lin, prefill_lin
        self.cur = lin  # the backend the running pass uses
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

        def register(name, wt, **kw):
            lin.register(name, wt, **kw)
            if prefill_lin is not None:
                prefill_lin.register(name, wt, **kw)

        for i in range(self.n_layers):
            b = f"blk.{i}."
            for n in ("attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm"):
                self.p[b + n] = w.f32(b + n + ".weight")
            if self.kind(i) == "conv":
                register(b + "in_proj", w.tensor(b + "shortconv.in_proj.weight"))
                register(b + "out_proj", w.tensor(b + "shortconv.out_proj.weight"))
                self.p[b + "conv"] = w.f32(b + "shortconv.conv.weight")  # (D, 7), k=6 is the newest token
            else:
                qkv = [w.tensor(b + f"attn_{n}.weight") for n in "qkv"]
                register(b + "qkv", qkv)  # one weight with stacked rows: q (nh*hd), k, v (nkv*hd each)
                register(b + "o", w.tensor(b + "attn_output.weight"))
                self.p[b + "gate"] = w.f32(b + "attn_gate.weight")  # (H, D)
                self.p[b + "q_norm"] = w.f32(b + "attn_q_norm.weight")
                self.p[b + "k_norm"] = w.f32(b + "attn_k_norm.weight")
            register(b + "gateup", [w.tensor(b + "ffn_gate.weight"), w.tensor(b + "ffn_up.weight")], glu=True)
            register(b + "down", w.tensor(b + "ffn_down.weight"))
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
    def _conv(self, b, i, u, st):
        P = self.cur(b + "in_proj", u)
        t0 = time.perf_counter()
        d = self.d
        Bg, Cg, X = P[:, :d], P[:, d : 2 * d], P[:, 2 * d :]
        z = Bg * X
        wc = self.p[b + "conv"]
        zp = np.concatenate([st.conv[i], z], 0)  # the previous 6 inputs, then this chunk's
        y = np.zeros_like(z)
        for k in range(LCONV):
            y += wc[:, k][None, :] * zp[k : k + z.shape[0]]
        st.conv[i] = zp[-(LCONV - 1) :]
        g = Cg * y
        self.cpu_ms += (time.perf_counter() - t0) * 1e3
        return self.cur(b + "out_proj", g)

    def _attn(self, b, i, u, kind, st):
        T, nh, hd, nkv, p0 = u.shape[0], self.nh, self.hd, self.kv_heads[i], st.pos
        qkv = self.cur(b + "qkv", u)
        t0 = time.perf_counter()
        q = rms(qkv[:, : nh * hd].reshape(T, nh, hd), self.p[b + "q_norm"])
        k = rms(qkv[:, nh * hd : (nh + nkv) * hd].reshape(T, nkv, hd), self.p[b + "k_norm"])
        cos, sin = self.rope(kind, T, p0)
        q = apply_rope(q, cos, sin)
        st.k[i][p0 : p0 + T] = apply_rope(k, cos, sin)
        st.v[i][p0 : p0 + T] = qkv[:, (nh + nkv) * hd :].reshape(T, nkv, hd)
        lo = max(0, p0 - self.window + 1) if kind == "swa" else 0  # earlier keys are outside every query's window
        rep = nh // nkv
        k = np.repeat(st.k[i][lo : p0 + T], rep, axis=1)  # (L, nh, hd)
        v = np.repeat(st.v[i][lo : p0 + T], rep, axis=1)
        s = np.matmul(q.transpose(1, 0, 2), k.transpose(1, 2, 0)) * np.float32(1.0 / np.sqrt(hd))
        i_pos, j_pos = p0 + np.arange(T)[:, None], lo + np.arange(k.shape[0])[None, :]
        mask = j_pos > i_pos
        if kind == "swa":
            mask |= (i_pos - j_pos) >= self.window
        s = np.where(mask[None], -np.inf, s)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = np.matmul(s, v.transpose(1, 0, 2)).transpose(1, 0, 2)
        gate = 1.0 / (1.0 + np.exp(-(u @ self.p[b + "gate"].T)))
        o = (o * gate[:, :, None]).reshape(T, nh * hd)
        self.cpu_ms += (time.perf_counter() - t0) * 1e3
        return self.cur(b + "o", o)

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
        ffn = getattr(self.cur, "ffn", None)  # the prompt GEMM's fused FFN: act stays on the NPU
        f = ffn(b + "gateup", b + "down", u) if ffn else None
        if f is None:
            gu = self.cur(b + "gateup", u)
            t0 = time.perf_counter()
            act = silu(gu[:, : self.ff]) * gu[:, self.ff :]
            self.cpu_ms += (time.perf_counter() - t0) * 1e3
            f = self.cur(b + "down", act)
        t0 = time.perf_counter()
        h = h + rms(f, self.p[b + "post_ffw_norm"])
        self.cpu_ms += (time.perf_counter() - t0) * 1e3
        return h

    def forward(self, ids, st: State | None = None, keep_hidden: bool = False, all_logits: bool = False):
        """Prefill `ids` at st.pos (a fresh state if None), in chunks of up to 256 tokens, extending the state.
        Returns (logits of the last position, or of every position if all_logits, and the list of per-layer
        hidden states if keep_hidden)."""
        ids = np.asarray(ids)
        st = st or self.new_state(len(ids))
        if st.pos + len(ids) > st.max_len:
            raise ValueError(f"context full ({st.pos} + {len(ids)} tokens > {st.max_len})")
        logits, hidden = [], [[] for _ in range(self.n_layers)] if keep_hidden else []
        for c in range(0, len(ids), CHUNK):
            chunk = ids[c : c + CHUNK]
            last = c + CHUNK >= len(ids)
            self.cur = self.prefill_lin if self.prefill_lin is not None and len(chunk) >= PREFILL_MIN else self.lin
            try:
                h = self.embed(chunk)
                for i in range(self.n_layers):
                    b, kind = f"blk.{i}.", self.kind(i)
                    mixer = self._conv if kind == "conv" else partial(self._attn, kind=kind)
                    h = self._block(b, h, partial(mixer, b, i, st=st))
                    if keep_hidden:
                        hidden[i].append(h.copy())
            finally:
                self.cur = self.lin
            st.pos += len(chunk)
            st.snapshot()
            if all_logits:
                logits.append(self.head(h))
            elif last:
                logits.append(self.head(h[-1:])[0])
        hidden = [np.concatenate(parts) for parts in hidden] if keep_hidden else []
        return (np.concatenate(logits) if all_logits else logits[-1]), hidden

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

    def generate(
        self, ids, n: int, st: State | None = None, on_token=None, logits_out: list | None = None, stop=(), sample=None
    ):
        """Prefill the prompt (`forward`), then generate up to n tokens, ending early after a token in `stop`
        (end-of-generation ids). `sample(logits, history)` picks each token (a xdna1_llm.sampling.Sampler); the
        default is greedy. Returns (new token ids, state, prompt seconds, generation seconds)."""
        st = st or self.new_state()
        t0 = time.perf_counter()
        logits, _ = self.forward(ids, st)
        t1 = time.perf_counter()
        out, history = [], [int(t) for t in ids]
        while True:
            if logits_out is not None:
                logits_out.append(logits)
            out.append(int(np.argmax(logits)) if sample is None else sample(logits, history))
            history.append(out[-1])
            if on_token:
                on_token(out[-1])
            if len(out) == n or out[-1] in stop:
                break
            logits = self.step(out[-1], st)
        return out, st, t1 - t0, time.perf_counter() - t1
