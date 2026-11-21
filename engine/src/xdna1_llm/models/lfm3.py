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

The layer code is written as generators: each yields the linear-layer calls it needs, ("lin", name, x) or ("ffn",
gate + up, down, x), and gets their results back. Decode runs one straight through (`_run`). A prompt runs one per
chunk (`_drive`) and keeps two chunks' calls in flight, so while the NPU computes one chunk's GEMM the host does the
other chunk's CPU ops; a backend that cannot overlap (no `submit`) is just called. The CPU ops of a prompt chunk
write into scratch arrays reused from layer to layer (`Work`): on Windows a fresh multi-MB numpy array costs
~0.3 ms in page faults, more than most of the ops on it.
"""

from __future__ import annotations

import collections
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np

from ..gguf_weights import dequantize

EPS, LCONV, SWA_WINDOW = 1e-5, 7, 256
CHUNK = 256  # prefill rows per pass (the bf16 GEMM's tile)
PREFILL_MIN = 32  # shorter prompt chunks run on `lin`: the GEMM always computes 256 rows
THREADS = 8  # prompt attention: one task per query head (numpy releases the GIL); 8 beat 4 and 16 on 8 cores
DEPTH = 2  # prompt chunks with a linear-layer call in flight
_pool = None


def _threads():
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(THREADS, thread_name_prefix="lfm3")
    return _pool


def rms(x, w, out=None):
    x = x.astype(np.float32, copy=False)
    inv = 1.0 / np.sqrt(np.einsum("...i,...i->...", x, x) / np.float32(x.shape[-1]) + np.float32(EPS))
    out = np.multiply(x, inv[..., None].astype(np.float32), out=out)
    out *= w
    return out


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


def apply_rope(x, cos, sin, out=None, tmp=None):
    """x (T, H, D) rotated: x * cos + rotate_half(x) * sin, into out; tmp: (T, H, D / 2) scratch."""
    half = x.shape[-1] // 2
    out = np.multiply(x, cos[:, None, :], out=out)
    tmp = np.multiply(x[..., half:], sin[:, None, :half], out=tmp)
    np.subtract(out[..., :half], tmp, out=out[..., :half])
    np.multiply(x[..., :half], sin[:, None, half:], out=tmp)
    np.add(out[..., half:], tmp, out=out[..., half:])
    return out


class Work:
    """Scratch arrays of one prompt chunk in flight, by name, reused from call to call (grown when too small)."""

    def __init__(self):
        self.bufs = {}

    def __call__(self, name: str, *shape: int) -> np.ndarray:
        n = int(np.prod(shape))
        a = self.bufs.get(name)
        if a is None or a.size < n:
            a = self.bufs[name] = np.empty(n, np.float32)
        return a[:n].reshape(shape)


class State:
    """Decode state: the position, per attention layer the keys/values so far (kv head, position, head dim: each
    head's keys are one contiguous matrix), per conv layer the last 6 inputs.

    A state can go back to an earlier position, to reuse a shared prefix (the next chat turn starts with this
    one). Attention needs nothing for that: keys/values past the position are overwritten. The conv window only
    holds the latest inputs, so `snapshot` keeps a copy of it at chosen positions and `rewind` returns to one.
    NPU attention keeps its own copy of the keys/values on the device (npu_kv); `rewind` marks what it must
    copy again."""

    def __init__(self, model: LFM3, max_len: int):
        self.pos, self.max_len = 0, max_len
        self.k, self.v, self.conv = {}, {}, {}
        for i in range(model.n_layers):
            if model.kind(i) == "conv":
                self.conv[i] = np.zeros((LCONV - 1, model.d), np.float32)
            else:
                self.k[i] = np.zeros((model.kv_heads[i], max_len, model.hd), np.float32)
                self.v[i] = np.zeros((model.kv_heads[i], max_len, model.hd), np.float32)
        self.snaps = {0: {i: c.copy() for i, c in self.conv.items()}}  # position -> conv windows there
        self.npu_kv = {}  # layer -> xdna1_llm.npu.attn.DeviceKV

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
        for kv in self.npu_kv.values():
            kv.valid = min(kv.valid, p)
        self.pos = p
        return p


class LFM3:
    def __init__(self, w, lin, prefill_lin=None, attn=None):
        if w.arch != "lfm3":
            raise ValueError(f"expected an lfm3 GGUF, got {w.arch!r}")
        self.w, self.lin, self.prefill_lin, self.attn = w, lin, prefill_lin, attn
        self.work = [Work() for _ in range(DEPTH + 1)]  # DEPTH for prompt chunks, the last for decode
        self.ropes = {}  # (kind, T, start) -> rope tables, for the running prompt
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
        key = (kind, T, start)
        if key not in self.ropes:  # every attention layer of a kind uses the same tables
            self.ropes[key] = self._rope(kind, T, start)
        return self.ropes[key]

    def _rope(self, kind, T, start):
        if kind == "full":
            return rope_tables(T, self.hd, self.base_full, self.rot_pairs_full, start)
        return rope_tables(T, self.hd, self.base_swa, self.hd // 2, start)

    # ---- linear-layer calls
    @staticmethod
    def _submit(lin, req):
        """Start the call req on backend lin; returns a function that waits for its result."""
        if req[0] == "ffn":  # the fused FFN, if the backend has one; else its result is None
            _, gateup, down, x = req
            if hasattr(lin, "submit_ffn"):
                p = lin.submit_ffn(gateup, down, x)
                return (lambda: None) if p is None else partial(lin.collect, p)
            ffn = getattr(lin, "ffn", None)
            return partial(ffn, gateup, down, x) if ffn else (lambda: None)
        _, name, x = req
        if hasattr(lin, "submit"):
            return partial(lin.collect, lin.submit(name, x))
        return partial(lin, name, x)  # called when its result is wanted

    def _run(self, gen, lin):
        """Run one layer generator to the end on backend lin; returns its return value."""
        v = None
        while True:
            try:
                req = gen.send(v)
            except StopIteration as e:
                return e.value
            v = self._submit(lin, req)()

    def _drive(self, jobs):
        """Run jobs [(make, backend)], in order, DEPTH at a time: make(work=...) returns a layer generator, work being
        a scratch set (Work) it has to itself until it ends. The call a generator yields is started, and its result
        waited for only after the others have run on to their next calls; calls finish in the order they started.
        Jobs on different backends do not overlap (the device would switch between their contexts on every call).
        Returns the generators' return values."""
        out = [None] * len(jobs)
        flight = collections.deque()  # (job, generator, its scratch, result), oldest first
        free = self.work[:DEPTH]

        def advance(j, gen, work, v):
            try:
                req = gen.send(v)
            except StopIteration as e:
                out[j] = e.value
                free.append(work)
                return
            flight.append((j, gen, work, self._submit(jobs[j][1], req)))

        k = 0
        while True:
            while k < len(jobs) and len(flight) < DEPTH and all(jobs[f[0]][1] is jobs[k][1] for f in flight):
                work = free.pop()
                k += 1
                advance(k - 1, jobs[k - 1][0](work=work), work, None)
            if not flight:
                return out
            j, gen, work, result = flight.popleft()
            advance(j, gen, work, result())

    # ---- layer generators: h, a chunk's rows of the residual stream, is updated in place; work, its scratch
    def _mix_in(self, b, h, work):
        t0 = time.perf_counter()
        u = rms(h, self.p[b + "attn_norm"], out=work("u", *h.shape))
        self.cpu_ms += (time.perf_counter() - t0) * 1e3
        return u

    def _mix_out(self, b, h, o, work):
        """The rest of a layer after its token mixer's output o: norm and residual, then the SwiGLU FFN."""
        t0 = time.perf_counter()
        t = work("t", *h.shape)
        h += rms(o, self.p[b + "post_attention_norm"], out=t)
        u = rms(h, self.p[b + "ffn_norm"], out=work("u", *h.shape))  # free once submitted (copied to the device)
        self.cpu_ms += (time.perf_counter() - t0) * 1e3
        f = yield ("ffn", b + "gateup", b + "down", u)  # fused (prompt GEMM, decode GEMV): act stays on the NPU
        if f is None:
            gu = yield ("lin", b + "gateup", u)
            t0 = time.perf_counter()
            act = silu(gu[:, : self.ff]) * gu[:, self.ff :]
            self.cpu_ms += (time.perf_counter() - t0) * 1e3
            f = yield ("lin", b + "down", act)
        t0 = time.perf_counter()
        h += rms(f, self.p[b + "post_ffw_norm"], out=t)
        self.cpu_ms += (time.perf_counter() - t0) * 1e3
        return h

    def _short_conv(self, b, P, tail, work=None):
        """The short conv between in_proj (P: (T, 3 d)) and out_proj, after the conv inputs tail (6, d) of the
        tokens before; returns (out_proj's input, the last 6 conv inputs)."""
        t0 = time.perf_counter()
        work = work or Work()
        d, T = self.d, P.shape[0]
        Bg, Cg, X = P[:, :d], P[:, d : 2 * d], P[:, 2 * d :]
        zp = work("zp", LCONV - 1 + T, d)  # the previous 6 inputs, then this chunk's
        zp[: LCONV - 1] = tail
        np.multiply(Bg, X, out=zp[LCONV - 1 :])
        wc = self.p[b + "conv"]
        y, tmp = np.multiply(zp[:T], wc[:, 0], out=work("y", T, d)), work("t", T, d)
        for k in range(1, LCONV):
            np.multiply(zp[k : k + T], wc[:, k], out=tmp)
            y += tmp
        g = np.multiply(Cg, y, out=work("u", T, d))
        self.cpu_ms += (time.perf_counter() - t0) * 1e3
        return g, zp[-(LCONV - 1) :].copy()

    def _conv_layer(self, b, h, tails, j, work):
        """A conv layer on chunk j's rows h: its conv reads the inputs tails[j] before it and sets tails[j + 1]
        (chunk j - 1 has: _drive returns results in the order the calls started)."""
        P = yield ("lin", b + "in_proj", self._mix_in(b, h, work))
        g, tails[j + 1] = self._short_conv(b, P, tails[j], work)
        o = yield ("lin", b + "out_proj", g)
        return (yield from self._mix_out(b, h, o, work))

    def _qkv(self, b, i, h, kind, st, p0, q_out, work):
        """q/k/v of a chunk at p0 (rows h): writes its keys and values into st and q (scaled by 1 / sqrt(hd)) into
        q_out (T, nh, hd); returns the per-head output gate."""
        u = self._mix_in(b, h, work)
        qkv = yield ("lin", b + "qkv", u)
        t0 = time.perf_counter()
        T, nh, hd, nkv = u.shape[0], self.nh, self.hd, self.kv_heads[i]
        cos, sin = self.rope(kind, T, p0)
        tmp = work("rope", T, nh, hd // 2)
        q = rms(qkv[:, : nh * hd].reshape(T, nh, hd), self.p[b + "q_norm"], out=work("t", T, nh, hd))
        apply_rope(q, cos, sin, q_out, tmp)
        q_out *= np.float32(1.0 / np.sqrt(hd))
        k = rms(qkv[:, nh * hd : (nh + nkv) * hd].reshape(T, nkv, hd), self.p[b + "k_norm"], out=work("k", T, nkv, hd))
        k = apply_rope(k, cos, sin, work("t", T, nkv, hd), tmp[:, :nkv])
        st.k[i][:, p0 : p0 + T] = k.transpose(1, 0, 2)
        st.v[i][:, p0 : p0 + T] = qkv[:, (nh + nkv) * hd :].reshape(T, nkv, hd).transpose(1, 0, 2)
        gate = 1.0 / (1.0 + np.exp(-(u @ self.p[b + "gate"].T)))
        self.cpu_ms += (time.perf_counter() - t0) * 1e3
        return gate

    def _attn_out(self, b, h, o, gate, work):
        """The rest of an attention layer: o_proj of the attention output o (T, nh, hd) times its per-head gate."""
        t0 = time.perf_counter()
        T = o.shape[0]
        g = np.multiply(o, gate[:, :, None], out=work("t", T, self.nh, self.hd)).reshape(T, self.nh * self.hd)
        self.cpu_ms += (time.perf_counter() - t0) * 1e3
        o = yield ("lin", b + "o", g)
        return (yield from self._mix_out(b, h, o, work))

    # ---- prefill
    def _attend(self, i, kind, st, chunks):
        """Causal (for swa layers also windowed) attention of a layer's chunks [(q, p0, npu)] over the keys in
        st: on the NPU, all in one go, where npu and a build covers the chunk, else on the CPU. Returns each
        chunk's (T, nh, hd)."""
        window = self.window if kind == "swa" else 0
        progs = [self.attn.program(p0, window) if npu and self.attn is not None else None for _, p0, npu in chunks]
        on_npu = [(q, p0, prog) for (q, p0, _), prog in zip(chunks, progs, strict=True) if prog is not None]
        npu_outs = iter(self.attn.run(st, i, on_npu, window) if on_npu else ())
        return [
            next(npu_outs) if prog is not None else self._attend_cpu(i, kind, q, st, p0)
            for (q, p0, _), prog in zip(chunks, progs, strict=True)
        ]

    def _attend_cpu(self, i, kind, q, st, p0):
        t0 = time.perf_counter()
        T, nh, nkv = q.shape[0], self.nh, self.kv_heads[i]
        lo = max(0, p0 - self.window + 1) if kind == "swa" else 0  # earlier keys are outside every query's window
        rep, K, V = nh // nkv, st.k[i][:, lo : p0 + T], st.v[i][:, lo : p0 + T]  # (nkv, L, hd)
        i_pos, j_pos = p0 + np.arange(T)[:, None], lo + np.arange(K.shape[1])[None, :]
        mask = j_pos > i_pos
        if kind == "swa":
            mask |= (i_pos - j_pos) >= self.window
        add = np.where(mask, np.float32(-np.inf), np.float32(0))
        o = np.empty_like(q)

        def head(h):  # query head h reads kv head h // rep; every operand is a BLAS-ready matrix view
            s = q[:, h] @ K[h // rep].T  # (T, L)
            s += add
            s -= s.max(-1, keepdims=True)
            np.exp(s, out=s)
            s /= s.sum(-1, keepdims=True)
            o[:, h] = s @ V[h // rep]

        list(_threads().map(head, range(nh)))
        self.cpu_ms += (time.perf_counter() - t0) * 1e3
        return o

    def forward(self, ids, st: State | None = None, keep_hidden: bool = False, all_logits: bool = False):
        """Prefill `ids` at st.pos (a fresh state if None), in chunks of up to 256 tokens, extending the state.
        Returns (logits of the last position, or of every position if all_logits, and the list of per-layer
        hidden states if keep_hidden).

        Layer by layer: every chunk goes through a layer before the next layer starts. An attention layer runs
        every chunk's q/k/v, then every chunk's attention, then every chunk's o_proj and FFN, so with attention
        on the NPU the device switches between the GEMM's and attention's contexts (~1.45 ms each time) twice per
        layer, not twice per chunk. Within each of these steps two chunks overlap (_drive)."""
        ids = np.asarray(ids)
        st = st or self.new_state(len(ids))
        if st.pos + len(ids) > st.max_len:
            raise ValueError(f"context full ({st.pos} + {len(ids)} tokens > {st.max_len})")
        chunks = [(st.pos + c, ids[c : c + CHUNK]) for c in range(0, len(ids), CHUNK)]
        n, p0s = len(chunks), [p0 for p0, _ in chunks]
        gemm = [self.prefill_lin is not None and len(ch) >= PREFILL_MIN for _, ch in chunks]
        backends = [self.prefill_lin if g else self.lin for g in gemm]
        hs = [self.embed(ch) for _, ch in chunks]
        qs = [np.empty((len(ch), self.nh, self.hd), np.float32) for _, ch in chunks]
        snaps = [{} for _ in chunks]  # conv windows at each chunk's end
        hidden = []
        self.ropes.clear()
        for i in range(self.n_layers):
            b, kind = f"blk.{i}.", self.kind(i)
            if kind == "conv":
                tails = [st.conv[i]] + [None] * n  # the conv inputs before each chunk
                self._drive([(partial(self._conv_layer, b, hs[j], tails, j), backends[j]) for j in range(n)])
                for j, snap in enumerate(snaps):
                    snap[i] = tails[j + 1]
                st.conv[i] = tails[-1].copy()
            else:
                qkv = partial(self._qkv, b, i, kind=kind, st=st)
                gates = self._drive([(partial(qkv, hs[j], p0=p0s[j], q_out=qs[j]), backends[j]) for j in range(n)])
                outs = self._attend(i, kind, st, [(qs[j], p0s[j], gemm[j]) for j in range(n)])
                self._drive([(partial(self._attn_out, b, hs[j], outs[j], gates[j]), backends[j]) for j in range(n)])
            if keep_hidden:
                hidden.append(np.concatenate(hs))
        self.ropes.clear()
        for (p0, ch), snap in zip(chunks, snaps, strict=True):
            st.snaps[p0 + len(ch)] = snap
        st.pos += len(ids)
        if all_logits:
            return np.concatenate([self.head(h) for h in hs]), hidden
        return self.head(hs[-1][-1:])[0], hidden

    # ---- decode
    def new_state(self, max_len: int = 4096) -> State:
        return State(self, max_len)

    def _block(self, b, h, mixer):
        """One decode layer around a token mixer (conv or attention): norms, residuals and the SwiGLU FFN."""
        work = self.work[DEPTH]
        return self._run(self._mix_out(b, h, mixer(self._mix_in(b, h, work)), work), self.lin)

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
        cos, sin = self._rope(kind, 1, pos)  # not cached: a new position every step
        st.k[i][:, pos] = apply_rope(k, cos, sin)[0]
        st.v[i][:, pos] = qkv[(nh + nkv) * hd :].reshape(nkv, hd)
        q = apply_rope(q, cos, sin)[0].reshape(nkv, nh // nkv, hd)  # GQA: query heads grouped by kv head
        lo = max(0, pos - self.window + 1) if kind == "swa" else 0
        K, V = st.k[i][:, lo : pos + 1], st.v[i][:, lo : pos + 1]  # (nkv, L, hd)
        s = np.matmul(q, K.transpose(0, 2, 1)) * np.float32(1.0 / np.sqrt(hd))  # (nkv, rep, L)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = np.matmul(s, V).reshape(nh, hd)
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
