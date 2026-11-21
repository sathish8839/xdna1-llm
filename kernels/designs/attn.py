"""Prompt attention on npu1: O = softmax(Q K^T) V for a chunk of 256 query rows and 16 query heads over 4 kv heads
(LFM3), flash-attention style on the cores (attn.cc).

Array column c is kv head c, row r its query head 4c + r: each K / V block goes down the column to all 4 cores.
Every core runs the 8 blocks of 32 query rows of its head one after another, each over its key blocks.

DDR layouts:
  Q   bf16 [row 256][head 16][128], then 8 header blocks (one per query block qb, 8 KB each, int32
      [key blocks, position of query row 0, position of key 0 of the first block, window]); the memory tile
      cuts the rows into the cores' A tiles
  KV  bf16 [key block][kv head 4][K 8 KB | V 8 KB], in attn.cc's tile layout; one per attention layer, kept by
      the host. Each column gets qb's header block, then its key blocks, K then V.
  O   fp32 [row 256][head 16][128 + 4]: unnormalised, l in column 128 (the memory tile untiles the cores' C tiles)

The key blocks each query block reads are fixed per program, so one build per key span (the cores take the
positions from the headers, and mask):
  --keys L      full attention over keys 0 .. L - 1 (L a multiple of 256): query block qb reads key blocks
                0 .. (L - 256) / 32 + qb, enough for a chunk that starts at any position up to L - 256
  --swa         sliding window (256): query block qb reads key blocks qb .. qb + 9 of a KV sub-buffer that the
                host starts at the window's first block, enough for any chunk starting at position >= 255

    python attn.py --dev npu (--keys L | --swa) [--cols 4] --xclbin-path X --insts-path Y
"""

import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli
from ml_dtypes import bfloat16

ROWS, QB, LQ, LK, HD = 4, 8, 32, 32, 128
KV_ELEMS = LK * HD  # bf16 per K or V block, and per header block
OD = HD + 4  # O columns: o, then l
SWA_SPAN = 10  # key blocks per query block in the sliding-window program
SRC = str(Path(__file__).parent / "attn.cc")


def key_blocks(keys: int, swa: bool) -> list[tuple[int, int]]:
    """Per query block: (first key block, number of key blocks) the program sends (xdna1_llm.npu.attn mirrors it)."""
    if swa:
        return [(qb, SWA_SPAN) for qb in range(QB)]
    return [(0, (keys - 256) // LK + qb + 1) for qb in range(QB)]


@iron.jit
def attn(
    Qd: In,
    KVd: In,
    Od: Out,
    *,
    KEYS: CompileTime[int],
    SWA: CompileTime[bool],
    COLS: CompileTime[int] = 4,
):
    spans = key_blocks(KEYS, SWA)
    kv_blocks = max(s + n for s, n in spans)  # blocks the KV argument covers
    heads = ROWS * COLS
    q_ty = np.ndarray[(LQ, HD), np.dtype[bfloat16]]
    q4_ty = np.ndarray[(ROWS * LQ, HD), np.dtype[bfloat16]]
    kv_ty = np.ndarray[(KV_ELEMS,), np.dtype[bfloat16]]
    o_ty = np.ndarray[(LQ * OD,), np.dtype[np.float32]]
    o4_ty = np.ndarray[(ROWS * LQ, OD), np.dtype[np.float32]]
    s_ty = np.ndarray[(LQ * LK,), np.dtype[np.float32]]
    p_ty = np.ndarray[(LQ * LK,), np.dtype[bfloat16]]
    st_ty = np.ndarray[(LQ // 4 * 16 + 16,), np.dtype[np.float32]]
    n_ty = np.ndarray[(4,), np.dtype[np.int32]]

    def fn(name, args, i):
        return ExternalFunction(name, source_file=SRC, arg_types=args, compile_flags=[f"-DATTN_FN={i}"])

    begin = fn("attn_begin", [kv_ty, o_ty, st_ty, n_ty], 0)
    qk = fn("attn_qk", [q_ty, kv_ty, s_ty, p_ty, st_ty], 1)
    pv = fn("attn_pv", [p_ty, kv_ty, o_ty, st_ty], 2)

    q_tiles = TensorAccessPattern.full((LQ, HD)).tile((4, 8))  # rows -> mmul A tiles
    o_rows = TensorAccessPattern.full((LQ, OD)).tile((4, 4)).inverse()  # each core's C tiles -> rows
    q_l3, kv_l3, o_l3 = [], [], []
    q_l1, kv_l1, o_l1 = [[] for _ in range(ROWS)], [], [[] for _ in range(ROWS)]
    for c in range(COLS):
        fq = ObjectFifo(q4_ty, name=f"Q{c}", depth=2)
        q_l3.append(fq)
        parts = fq.cons().split(
            [LQ * HD * r for r in range(ROWS)],
            obj_types=[q_ty] * ROWS,
            names=[f"Q{c}_{r}" for r in range(ROWS)],
            depths=[1] * ROWS,
            to_stream=[q_tiles] * ROWS,
        )
        for r in range(ROWS):
            q_l1[r].append(parts[r])
        fkv = ObjectFifo(kv_ty, name=f"KV{c}", depth=2)
        kv_l3.append(fkv)
        kv_l1.append(fkv.cons().forward(obj_type=kv_ty, name=f"KV{c}_l1", depth=2))
        fo = ObjectFifo(o4_ty, name=f"O{c}", depth=2, to_stream=o_rows)
        o_l3.append(fo)
        parts = fo.prod().join(
            [LQ * OD * r for r in range(ROWS)],
            obj_types=[o_ty] * ROWS,
            names=[f"O{c}_{r}" for r in range(ROWS)],
            depths=[1] * ROWS,
        )
        for r in range(ROWS):
            o_l1[r].append(parts[r])

    def core_fn(q_in, kv_in, o_out, s, p, st, n, begin, qk, pv):
        for _ in range_(QB):
            q = q_in.acquire(1)
            o = o_out.acquire(1)
            h = kv_in.acquire(1)  # the header block
            begin(h, o, st, n)
            kv_in.release(1)
            for _ in range_(n[0]):
                k = kv_in.acquire(1)
                qk(q, k, s, p, st)
                kv_in.release(1)
                v = kv_in.acquire(1)
                pv(p, v, o, st)
                kv_in.release(1)
            o_out.release(1)
            q_in.release(1)

    workers = Worker.grid(
        ROWS,
        COLS,
        lambda r, c: Worker(
            core_fn,
            [
                q_l1[r][c].cons(),
                kv_l1[c].cons(),
                o_l1[r][c].prod(),
                Buffer(s_ty, name=f"s_{r}_{c}"),
                Buffer(p_ty, name=f"p_{r}_{c}"),
                Buffer(st_ty, name=f"st_{r}_{c}"),
                Buffer(n_ty, name=f"n_{r}_{c}"),
                begin,
                qk,
                pv,
            ],
            stack_size=0xC00,
        ),
    )

    q_rows = QB * LQ * heads * HD
    q_total, o_total = q_rows + QB * KV_ELEMS, QB * LQ * heads * OD
    kv_total = kv_blocks * COLS * 2 * KV_ELEMS

    def sequence(qd, kvd, od, q_hs, kv_hs, o_hs):
        # one task group per query block, finished a block later: the next block's transfers queue behind it,
        # and the shim's 16 buffer descriptors hold two blocks' (4 each)
        for qb, tg in TaskGroup.pipelined(QB):
            k0, nk = spans[qb]
            for c in range(COLS):
                off = (qb * LQ * heads + ROWS * c) * HD  # heads 4c .. 4c + 3 of rows 32 qb ..
                q_hs[c].fill(qd, TensorAccessPattern((q_total,), off, [ROWS, LQ, HD], [HD, heads * HD, 1]), group=tg)
                hdr = TensorAccessPattern((q_total,), q_rows + qb * KV_ELEMS, [1, KV_ELEMS], [0, 1])
                kv_hs[c].fill(qd, hdr, group=tg)
                off = (k0 * COLS + c) * 2 * KV_ELEMS
                tap = TensorAccessPattern(
                    (kv_total,), off, [nk, 2 * KV_ELEMS // 512, 512], [COLS * 2 * KV_ELEMS, 512, 1]
                )
                kv_hs[c].fill(kvd, tap, group=tg)
            for c in range(COLS):
                off = (qb * LQ * heads + ROWS * c) * OD
                tap = TensorAccessPattern((o_total,), off, [ROWS, LQ, OD], [OD, heads * OD, 1])
                o_hs[c].drain(od, tap, wait=True, group=tg)

    rt = Runtime(
        sequence,
        [
            np.ndarray[(q_total,), np.dtype[bfloat16]],
            np.ndarray[(kv_total,), np.dtype[bfloat16]],
            np.ndarray[(o_total,), np.dtype[np.float32]],
            [f.prod() for f in q_l3],
            [f.prod() for f in kv_l3],
            [f.cons() for f in o_l3],
        ],
    )
    flat = [w for row in workers for w in row]
    return Program(iron.get_current_device(), rt, workers=flat).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="attn")
    add_compile_args(p)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--keys", type=int, help="full attention over this many keys (a multiple of 256)")
    g.add_argument("--swa", action="store_true", help="sliding-window attention (window 256)")
    p.add_argument("--cols", type=int, default=4, choices=(1, 4))
    opts = p.parse_args()
    if opts.keys is not None:
        assert opts.keys % 256 == 0 and opts.keys >= 256, opts.keys
    run_design_cli(
        attn,
        opts,
        compile_kwargs=lambda o: dict(KEYS=o.keys or 0, SWA=o.swa, COLS=o.cols),
        run_and_verify=lambda o: None,
        device=lambda o: device_from_args(o, n_cols=4),
    )


if __name__ == "__main__":
    main()
