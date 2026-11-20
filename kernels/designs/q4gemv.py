"""4-bit GEMV for decode on npu1 (layout and math in q4gemv.cc): y[N] (fp32) = W[N, K] (4-bit) . x[K] (bf16).

Weights stream over 8 lanes (two shim channels per column, cores in rows 2 and 3); each lane owns N/8
contiguous rows, in blocks of 16 rows. A block is split along K into tiles of KC columns (nibbles plus Q4_K's
scales, 5.25 bits per weight; see q4gemv.cc) so long rows fit in core memory; the
core carries partial sums across a block's tiles and streams the block's 16 outputs back as soon as it is done.

x travels on the same lane, ahead of the weights, with the number of blocks to expect. So the core program does
not depend on N: one xclbin per K serves every N, and only the instruction stream differs.

    python q4gemv.py --dev npu -N 2048 -K 2048 --xclbin-path X --insts-path Y
"""

import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli
from ml_dtypes import bfloat16

LANES = 8
RB = 16  # rows per block
SRC = str(Path(__file__).parent / "q4gemv.cc")


def layout(K: int) -> tuple[int, int, int]:
    """(KC, tile bytes, x tiles): K per tile (at most 2048, at most 1536 when K must be split), the tile size,
    and how many tiles carry x plus the 4-byte block count."""
    KC = K if K <= 2048 else max(d for d in range(256, 1537, 256) if K % d == 0)
    TB = KC // 2 * RB + KC // 32 * 2 * RB * 2 + KC // 256 * 4 * RB * 2
    XT = -(-(2 * K + 4) // TB)
    return KC, TB, XT


@iron.jit
def q4gemv(W: In, X: In, Y: Out, *, N: CompileTime[int], K: CompileTime[int]):
    KC, TB, XT = layout(K)
    NC = K // KC
    assert N % (LANES * RB) == 0 and K % 256 == 0, (N, K)

    u8 = np.dtype[np.uint8]
    tile_ty = np.ndarray[(TB,), u8]
    x_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    cnt_ty = np.ndarray[(1,), np.dtype[np.int32]]
    y_ty = np.ndarray[(RB,), np.dtype[np.float32]]

    flags = [f"-DQ4_K={K}", f"-DQ4_KC={KC}", f"-DQ4_TB={TB}", f"-DQ4_XT={XT}"]
    load_x = ExternalFunction(
        "q4_load_x", source_file=SRC, arg_types=[tile_ty, x_ty, cnt_ty], compile_flags=flags + ["-DQ4_FN=0"]
    )
    q4_rows = ExternalFunction(
        "q4_rows", source_file=SRC, arg_types=[tile_ty, x_ty, y_ty], compile_flags=flags + ["-DQ4_FN=1"]
    )

    def core_fn(of_in, of_out, xbuf, cnt, load_x, q4_rows):
        for _ in range_(XT):
            t = of_in.acquire(1)
            load_x(t, xbuf, cnt)
            of_in.release(1)
        for _ in range_(cnt[0]):
            y = of_out.acquire(1)
            for _ in range_(NC):
                t = of_in.acquire(1)
                q4_rows(t, xbuf, y)
                of_in.release(1)
            of_out.release(1)

    ins, outs, workers = [], [], []
    for lane in range(LANES):
        col, row = lane % 4, 2 + lane // 4
        fin = ObjectFifo(tile_ty, name=f"w{lane}", depth=2)
        fout = ObjectFifo(y_ty, name=f"y{lane}", depth=2)
        ins.append(fin.prod(tile=Tile(col=col, row=0)))
        outs.append(fout.cons(tile=Tile(col=col, row=0)))
        xbuf = Buffer(type=x_ty, name=f"x{lane}")
        cnt = Buffer(type=cnt_ty, name=f"n{lane}")
        workers.append(
            Worker(core_fn, [fin.cons(), fout.prod(), xbuf, cnt, load_x, q4_rows], tile=Tile(col=col, row=row))
        )

    n_bytes = N // RB * NC * TB
    w_taps = TensorAccessPattern.full((n_bytes,)).partition(LANES)
    y_taps = TensorAccessPattern.full((N,)).partition(LANES)

    def sequence(w, x, y, in_hs, out_hs):
        for i in range(LANES):
            in_hs[i].fill(x)
            in_hs[i].fill(w, w_taps[i])
        for i in range(LANES):
            out_hs[i].drain(y, y_taps[i], wait=True)

    x_all = np.ndarray[(XT * TB,), u8]
    rt = Runtime(sequence, [np.ndarray[(n_bytes,), u8], x_all, np.ndarray[(N,), np.dtype[np.float32]], ins, outs])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="q4gemv")
    add_compile_args(p)
    p.add_argument("-N", type=int, required=True)
    p.add_argument("-K", type=int, required=True)
    opts = p.parse_args()
    run_design_cli(
        q4gemv,
        opts,
        compile_kwargs=lambda o: dict(N=o.N, K=o.K),
        run_and_verify=lambda o: None,
        device=lambda o: device_from_args(o, n_cols=4),
    )


if __name__ == "__main__":
    main()
