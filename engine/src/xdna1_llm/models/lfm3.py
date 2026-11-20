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
"""

from __future__ import annotations

import time

import numpy as np

from ..bf16 import from_bf16

EPS, LCONV, SWA_WINDOW = 1e-5, 7, 256


def rms(x, w):
    x = x.astype(np.float32, copy=False)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope_tables(T, head_dim, base, rot_pairs):
    """NeoX (rotate_half) tables; pairs i >= rot_pairs get angle 0 (partial rotary)."""
    inv = base ** (-np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    inv[rot_pairs:] = 0.0
    ang = np.arange(T, dtype=np.float64)[:, None] * inv[None, :]
    ang = np.concatenate([ang, ang], axis=1)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def apply_rope(x, cos, sin):  # x (T, H, D)
    half = x.shape[-1] // 2
    rot = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


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

        self.p: dict[str, np.ndarray] = {}
        t0 = time.perf_counter()
        for i in range(self.n_layers):
            b = f"blk.{i}."
            for n in ("attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm"):
                self.p[b + n] = w.f32(b + n + ".weight")
            if self.kind(i) == "conv":
                lin.register(b + "in_proj", w.raw(b + "shortconv.in_proj.weight"))
                lin.register(b + "out_proj", w.raw(b + "shortconv.out_proj.weight"))
                self.p[b + "conv"] = w.f32(b + "shortconv.conv.weight")  # (D, 7), k=6 is the newest token
            else:
                lin.register(b + "q", w.raw(b + "attn_q.weight"))
                lin.register(b + "kv", np.concatenate([w.raw(b + "attn_k.weight"), w.raw(b + "attn_v.weight")], 0))
                lin.register(b + "o", w.raw(b + "attn_output.weight"))
                self.p[b + "gate"] = w.f32(b + "attn_gate.weight")  # (H, D)
                self.p[b + "q_norm"] = w.f32(b + "attn_q_norm.weight")
                self.p[b + "k_norm"] = w.f32(b + "attn_k_norm.weight")
            lin.register(b + "gateup", np.concatenate([w.raw(b + "ffn_gate.weight"), w.raw(b + "ffn_up.weight")], 0))
            lin.register(b + "down", w.raw(b + "ffn_down.weight"))
        self.out_norm = w.f32("output_norm.weight")
        self.embd = w.raw("token_embd.weight")
        self.lm_head = from_bf16(self.embd)  # fp32 copy of the tied head
        self.load_s = time.perf_counter() - t0
        self.cpu_ms = 0.0

    def kind(self, i: int) -> str:
        if self.kv_heads[i] == 0:
            return "conv"
        return "swa" if self.swa[i] else "full"

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
        q = self.lin(b + "q", u).reshape(T, nh, hd)
        kv = self.lin(b + "kv", u)
        t0 = time.perf_counter()
        k = rms(kv[:, : nkv * hd].reshape(T, nkv, hd), self.p[b + "k_norm"])
        v = kv[:, nkv * hd :].reshape(T, nkv, hd)
        q = rms(q, self.p[b + "q_norm"])
        if kind == "full":
            cos, sin = rope_tables(T, hd, self.base_full, self.rot_pairs_full)
        else:
            cos, sin = rope_tables(T, hd, self.base_swa, hd // 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = nh // nkv
        k, v = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
        s = np.matmul(q.transpose(1, 0, 2), k.transpose(1, 2, 0)) * np.float32(1.0 / np.sqrt(hd))
        i, j = np.arange(T)[:, None], np.arange(T)[None, :]
        mask = j > i
        if kind == "swa":
            mask |= (i - j) >= SWA_WINDOW
        s = np.where(mask[None], -np.inf, s)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = np.matmul(s, v.transpose(1, 0, 2)).transpose(1, 0, 2)
        gate = 1.0 / (1.0 + np.exp(-(u @ self.p[b + "gate"].T)))
        o = (o * gate[:, :, None]).reshape(T, nh * hd)
        self.cpu_ms += (time.perf_counter() - t0) * 1e3
        return self.lin(b + "o", o)

    def forward(self, ids, keep_hidden: bool = False):
        """Prefill: returns (logits of the last position, list of per-layer hidden states if keep_hidden)."""
        h = from_bf16(self.embd[np.asarray(ids)])
        hidden = []
        for i in range(self.n_layers):
            b, kind = f"blk.{i}.", self.kind(i)
            t0 = time.perf_counter()
            u = rms(h, self.p[b + "attn_norm"])
            self.cpu_ms += (time.perf_counter() - t0) * 1e3
            o = self._conv(b, u) if kind == "conv" else self._attn(b, u, kind)
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
            if keep_hidden:
                hidden.append(h.copy())
        t0 = time.perf_counter()
        logits = (rms(h[-1:], self.out_norm) @ self.lm_head.T)[0]
        self.cpu_ms += (time.perf_counter() - t0) * 1e3
        return logits, hidden
