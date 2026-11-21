"""Prompt attention on the NPU (kernels/designs/attn.py): softmax(q k^T) v for a chunk of up to 256 query rows,
16 query heads over 4 kv heads (LFM3's GQA), head dim 128, causal, optionally with a sliding window.

The cores do not depend on how many keys a call reads, only the instruction streams do, so one context runs
every build: `attn_<L>.bin` covers full attention for a chunk that starts at up to L - 256 (its query block qb
reads key blocks 0 .. (L - 256) / 32 + qb), `attn_swa.bin` the 256-key window for a chunk that starts at 255 or
later (key blocks qb .. qb + 9 of the window's sub-buffer). The cores mask by position, from a header block the
host writes for each block of 32 queries.

q goes in as rows and the output comes back as rows (the array's DMA tiles and untiles them), so the host only
converts q to bf16 and divides the output by its softmax sum. K carries the log2(e) of the cores' exp2. `run`
takes every chunk of a layer: while the NPU runs one, the host copies the next one's keys and q, and reads the
previous one's output (two sets of q and output buffers, used in turn).

Keys and values: each State keeps a bf16 copy of every attention layer's on the device, in the kernel's tile
layout (st.npu_kv). The fp32 CPU cache stays the reference, and decode only extends that one: before a call, the
key blocks from where the device copy is valid to the end of the chunk are tiled again from it.
"""

from __future__ import annotations

import itertools
import os
import time
from concurrent.futures import ThreadPoolExecutor

import ml_dtypes
import numpy as np

from .device import FROM_DEV, TO_DEV, finish, mapped, sub_bo

QB, LQ, LK, HD, HEADS, KV_HEADS = 8, 32, 32, 128, 16, 4
ROWS = QB * LQ  # query rows per call
KV_BLOCK = 2 * LK * HD  # bf16 per key block and kv head: K tiles, then V tiles
HDR = LK * HD  # bf16 per header block (int32 [key blocks, query position, first key position, window] first)
OD = HD + 4  # output columns: o, then the softmax sum l
SWA_WINDOW, SWA_SPAN = 256, 10  # attn_swa's window, and the key blocks each query block reads
LOG2E = np.float32(1.4426950408889634)  # the cores take exp2
THREADS = 4


def spans(kdir: str) -> list[int]:
    """The key spans L of the full-attention builds in kdir (attn_<L>.bin), ascending."""
    names = [f[5:-4] for f in os.listdir(kdir) if f.startswith("attn_") and f.endswith(".bin")]
    return sorted(int(n) for n in names if n.isdigit())


class DeviceKV:
    """One attention layer's keys and values on the device: [key block][kv head][K tiles | V tiles], bf16, and
    the number of positions that match the CPU cache (State.rewind lowers it)."""

    def __init__(self, npu, group: int, blocks: int):
        self.bo = npu.bo(blocks * KV_HEADS * KV_BLOCK * 2, group)
        self.tiles = mapped(self.bo, np.uint16, (blocks, KV_HEADS, 2, LK * HD))
        self.tiles[:] = 0  # padding keys are masked, but 0 * NaN would not be
        self.bo.sync(TO_DEV)
        self.valid = 0


class NpuAttn:
    def __init__(self, npu, kdir: str):
        self.npu, self.kdir = npu, kdir
        self.spans = spans(kdir)
        if not self.spans:
            raise FileNotFoundError(f"no NPU attention in {kdir} (kernels/build_kernels.sh <out> attn)")
        first = os.path.join(kdir, f"attn_{self.spans[0]}")
        self.kern = npu.load("attn", first + ".xclbin", first + ".bin")  # every build has the same cores
        for L in self.spans[1:]:
            self.kern.add_program(f"attn_{L}", os.path.join(kdir, f"attn_{L}.bin"))
        self.kern.programs[f"attn_{self.spans[0]}"] = self.kern.programs["default"]
        self.swa = os.path.exists(os.path.join(kdir, "attn_swa.bin"))
        if self.swa:
            self.kern.add_program("attn_swa", os.path.join(kdir, "attn_swa.bin"))
        self.bufs = [self._buffers() for _ in range(2)]  # used in turn: one is filled while the NPU reads the other
        self.pool = ThreadPoolExecutor(THREADS, thread_name_prefix="attn")
        self.npu_ms, self.calls = 0.0, 0

    def _buffers(self):
        """A q buffer (rows, then the header blocks) and an output buffer, with numpy views."""
        bo_q = self.npu.bo((ROWS * HEADS * HD + QB * HDR) * 2, self.kern.group(0))
        q = mapped(bo_q, np.uint16)
        q[:] = 0  # rows past a chunk's end: any finite values
        rows = q[: ROWS * HEADS * HD].view(ml_dtypes.bfloat16).reshape(ROWS, HEADS, HD)
        hdr = q[ROWS * HEADS * HD :].view(np.int32).reshape(QB, HDR // 2)
        bo_o = self.npu.bo(ROWS * HEADS * OD * 4, self.kern.group(2))
        return bo_q, rows, hdr, bo_o, mapped(bo_o, np.float32, (ROWS, HEADS, OD))

    def program(self, p0: int, window: int) -> str | None:
        """The build for a chunk starting at p0 (window: the layer's sliding window, 0 for none); None if no build
        covers it (the caller runs the chunk on the CPU)."""
        if window and p0 >= window - 1:
            return "attn_swa" if self.swa and window == SWA_WINDOW else None
        L = next((L for L in self.spans if L >= p0 + ROWS), None)
        return None if L is None else f"attn_{L}"

    def _kv(self, st, i: int) -> DeviceKV:
        kv = st.npu_kv.get(i)
        if kv is None:
            # the last key block a call may read: full, block L / 32 - 1 with L < p0 + 512 or the largest span;
            # sliding window, block (p0 - 255) / 32 + 16
            keys = max(st.max_len + 2 * ROWS, self.spans[-1])
            kv = st.npu_kv[i] = DeviceKV(self.npu, self.kern.group(1), -(-keys // LK))
        return kv

    def _sync_kv(self, kv: DeviceKV, k: np.ndarray, v: np.ndarray, end: int):
        """Tile the key blocks from kv.valid's block to the one holding position end - 1 from the CPU cache
        (k, v: (kv heads, positions, head dim) fp32; K times log2(e)) and copy them to the device."""
        b0, b1 = kv.valid // LK, -(-end // LK)
        n = b1 - b0
        kk = np.zeros((KV_HEADS, n * LK, HD), ml_dtypes.bfloat16)
        vv = np.zeros_like(kk)
        have = min(b1 * LK, k.shape[1]) - b0 * LK
        np.multiply(k[:, b0 * LK : b0 * LK + have], LOG2E, out=kk[:, :have], casting="unsafe")
        vv[:, :have] = v[:, b0 * LK : b0 * LK + have]
        # K: B tiles [tj][td][r][c] = K[4 tj + c][8 td + r];  V: B tiles [tn][tk][r][c] = V[8 tk + r][4 tn + c]
        kt = kk.view(np.uint16).reshape(KV_HEADS, n, LK // 4, 4, HD // 8, 8).transpose(1, 0, 2, 4, 5, 3)
        vt = vv.view(np.uint16).reshape(KV_HEADS, n, LK // 8, 8, HD // 4, 4).transpose(1, 0, 4, 2, 3, 5)
        kv.tiles[b0:b1, :, 0] = kt.reshape(n, KV_HEADS, LK * HD)
        kv.tiles[b0:b1, :, 1] = vt.reshape(n, KV_HEADS, LK * HD)
        per_block = KV_HEADS * KV_BLOCK * 2
        kv.bo.sync(TO_DEV, n * per_block, b0 * per_block)
        kv.valid = end

    def _split(self, fn, T: int):
        """fn(i, j) on row ranges of 0 .. T over the pool (numpy releases the GIL)."""
        bounds = np.linspace(0, T, THREADS + 1).astype(int)
        for f in [self.pool.submit(fn, i, j) for i, j in itertools.pairwise(bounds) if j > i]:
            f.result()

    def _start(self, buf, kv: DeviceKV, q: np.ndarray, p0: int, prog: str, window: int):
        """Fill buf with q (T rows) and its headers, start the program on it; returns (run, sub-buffer to keep)."""
        bo_q, rows, hdr, bo_o, _ = buf
        T = len(q)
        self._split(lambda a, b: np.copyto(rows[a:b], q[a:b], casting="unsafe"), T)
        qb = np.arange(QB)
        if prog == "attn_swa":
            kbase = (p0 - (window - 1)) // LK  # the window's first key block: the sub-buffer starts there
            h = [np.full(QB, SWA_SPAN), p0 - kbase * LK + LQ * qb, LK * qb, np.full(QB, window)]
            per_block = KV_HEADS * KV_BLOCK * 2
            bo_kv = sub_bo(kv.bo, (QB - 1 + SWA_SPAN) * per_block, kbase * per_block)
        else:
            L = int(prog[5:])
            h = [(L - ROWS) // LK + qb + 1, p0 + LQ * qb, np.zeros(QB, int), np.full(QB, window)]
            bo_kv = kv.bo
        hdr[:, :4] = np.stack(h, 1)
        bo_q.sync(TO_DEV)
        return self.kern.start(bo_q, bo_kv, bo_o, program=prog), bo_kv

    def _result(self, buf, T: int) -> np.ndarray:
        _, _, _, bo_o, o = buf
        bo_o.sync(FROM_DEV, T * HEADS * OD * 4, 0)
        out = np.empty((T, HEADS, HD), np.float32)
        self._split(lambda a, b: np.divide(o[a:b, :, :HD], o[a:b, :, HD : HD + 1], out=out[a:b]), T)
        return out

    def run(self, st, i: int, chunks: list[tuple[np.ndarray, int, str]], window: int) -> list[np.ndarray]:
        """Attention of a layer's chunks [(q, p0, program)], in position order: q (T <= 256, 16 heads, 128;
        scaled by 1 / sqrt(128)) at positions p0 .. p0 + T - 1 over layer i's keys in st up to p0 + T - 1 (all
        already written there); program from program(). Returns each chunk's (T, 16, 128) fp32."""
        t0 = time.perf_counter()
        kv = self._kv(st, i)
        kv.valid = min(kv.valid, chunks[0][1])  # these chunks' keys were just written
        outs, prev = [], None
        for j, (q, p0, prog) in enumerate(chunks):
            # the running program may read these blocks too, but only as masked keys (any finite values do)
            self._sync_kv(kv, st.k[i], st.v[i], p0 + len(q))
            run = self._start(self.bufs[j % 2], kv, q, p0, prog, window)
            if prev is not None:
                finish(prev[0][0], "attention")
                outs.append(self._result(self.bufs[(j - 1) % 2], prev[1]))
            prev = run, len(q)
        finish(prev[0][0], "attention")
        outs.append(self._result(self.bufs[(len(chunks) - 1) % 2], prev[1]))
        self.npu_ms += (time.perf_counter() - t0) * 1e3
        self.calls += len(chunks)
        return outs
