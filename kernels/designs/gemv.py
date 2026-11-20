"""bf16 GEMV for decode on npu1: y[N] (fp32) = W[N, K] (bf16) . x[K] (bf16).

Weights stream over 8 lanes (two shim channels per column, cores in rows 2 and 3); each lane owns N/8
contiguous rows. The activation x travels on the same lane, as one tile in front of the weights, together with
the number of weight tiles that follow. So the cores do not depend on N: every N up to N_MAX compiles to the
same core program, and only the instruction stream (the host-side DMA program) differs. The engine loads one
context per K and switches instruction streams to change N.

Host layout: W rows are padded to a stride of KS = layout(K)[1] elements; the x tile holds x, then the tile
count as an int32 at element K. Y holds N_MAX/8 floats per lane; lane l's rows are at Y[l * N_MAX/8 :].

    python gemv.py --dev npu -N 2048 -K 2048 --xclbin-path X --insts-path Y
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
TILE_BYTES = 16 * 1024
N_MAX = 15360  # largest projection (LFM3 FFN gate+up)
SRC = str(Path(__file__).parent / "gemv.cc")


def layout(K: int) -> tuple[int, int]:
    """(rows per weight tile, row stride). Rows are padded only when a tile holds a single row, to leave room
    for the tile count after x."""
    R = max(1, TILE_BYTES // (2 * K))
    return R, (K if R > 1 else K + 32)


@iron.jit
def gemv(W: In, X: In, Y: Out, *, N: CompileTime[int], K: CompileTime[int]):
    R, KS = layout(K)
    rows, rows_max = N // LANES, N_MAX // LANES
    assert N <= N_MAX and N % LANES == 0 and rows % R == 0 and K % 64 == 0, (N, K, R)

    bf = np.dtype[bfloat16]
    tile_ty = np.ndarray[(R * KS,), bf]
    x_ty = np.ndarray[(K,), bf]
    cnt_ty = np.ndarray[(1,), np.dtype[np.int32]]
    y_ty = np.ndarray[(rows_max,), np.dtype[np.float32]]

    flags = [f"-DGEMV_K={K}", f"-DGEMV_KS={KS}", f"-DGEMV_R={R}"]
    load_x = ExternalFunction(
        "gemv_load_x", source_file=SRC, arg_types=[tile_ty, x_ty, cnt_ty], compile_flags=flags + ["-DGEMV_FN=0"]
    )
    gemv_rows = ExternalFunction(
        "gemv_rows", source_file=SRC, arg_types=[tile_ty, x_ty, y_ty], compile_flags=flags + ["-DGEMV_FN=1"]
    )

    def core_fn(of_in, of_out, xbuf, cnt, load_x, gemv_rows):
        y = of_out.acquire(1)
        t = of_in.acquire(1)
        load_x(t, xbuf, cnt)
        of_in.release(1)
        for _ in range_(cnt[0]):
            t = of_in.acquire(1)
            gemv_rows(t, xbuf, y)
            of_in.release(1)
        of_out.release(1)

    ins, outs, workers = [], [], []
    for lane in range(LANES):
        col, row = lane % 4, 2 + lane // 4
        fin = ObjectFifo(tile_ty, name=f"w{lane}", depth=2)
        fout = ObjectFifo(y_ty, name=f"y{lane}", depth=1)
        ins.append(fin.prod(tile=Tile(col=col, row=0)))
        outs.append(fout.cons(tile=Tile(col=col, row=0)))
        xbuf = Buffer(type=x_ty, name=f"x{lane}")
        cnt = Buffer(type=cnt_ty, name=f"n{lane}")
        workers.append(
            Worker(core_fn, [fin.cons(), fout.prod(), xbuf, cnt, load_x, gemv_rows], tile=Tile(col=col, row=row))
        )

    w_taps = TensorAccessPattern.full((N * KS,)).partition(LANES)
    y_taps = TensorAccessPattern.full((LANES * rows_max,)).partition(LANES)

    def sequence(w, x, y, in_hs, out_hs):
        for i in range(LANES):
            in_hs[i].fill(x)
            in_hs[i].fill(w, w_taps[i])
        for i in range(LANES):
            out_hs[i].drain(y, y_taps[i], wait=True)

    w_ty = np.ndarray[(N * KS,), bf]
    out_ty = np.ndarray[(LANES * rows_max,), np.dtype[np.float32]]
    rt = Runtime(sequence, [w_ty, tile_ty, out_ty, ins, outs])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="gemv")
    add_compile_args(p)
    p.add_argument("-N", type=int, required=True)
    p.add_argument("-K", type=int, required=True)
    opts = p.parse_args()
    run_design_cli(
        gemv,
        opts,
        compile_kwargs=lambda o: dict(N=o.N, K=o.K),
        run_and_verify=lambda o: None,
        device=lambda o: device_from_args(o, n_cols=4),
    )


if __name__ == "__main__":
    main()
