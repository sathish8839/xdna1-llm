"""4-bit GEMV for decode on npu1 (layout and math in q4gemv.cc): y[N] (fp32) = W[N, K] (4-bit) . x[K] (bf16).

Same streaming scheme as gemv.py: 8 lanes, x plus the tile count travel ahead of the weights on each lane, so one
core program serves every N up to N_MAX. A tile is 16 rows: K/2*16 bytes of nibbles plus K/32*64 bytes of
bf16 scales and mins (5 bits per weight).

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
RB = 16  # rows per tile
N_MAX = 15360
SRC = str(Path(__file__).parent / "q4gemv.cc")


def tile_bytes(K: int) -> int:
    return K // 2 * RB + K // 32 * 2 * RB * 2


@iron.jit
def q4gemv(W: In, X: In, Y: Out, *, N: CompileTime[int], K: CompileTime[int]):
    TB = tile_bytes(K)
    rows, rows_max = N // LANES, N_MAX // LANES
    assert N <= N_MAX and rows % RB == 0 and K % 64 == 0, (N, K)

    u8 = np.dtype[np.uint8]
    tile_ty = np.ndarray[(TB,), u8]
    x_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    cnt_ty = np.ndarray[(1,), np.dtype[np.int32]]
    y_ty = np.ndarray[(rows_max,), np.dtype[np.float32]]

    flags = [f"-DQ4_K={K}"]
    load_x = ExternalFunction(
        "q4_load_x", source_file=SRC, arg_types=[tile_ty, x_ty, cnt_ty], compile_flags=flags + ["-DQ4_FN=0"]
    )
    q4_rows = ExternalFunction(
        "q4_rows", source_file=SRC, arg_types=[tile_ty, x_ty, y_ty], compile_flags=flags + ["-DQ4_FN=1"]
    )

    def core_fn(of_in, of_out, xbuf, cnt, load_x, q4_rows):
        y = of_out.acquire(1)
        t = of_in.acquire(1)
        load_x(t, xbuf, cnt)
        of_in.release(1)
        for _ in range_(cnt[0]):
            t = of_in.acquire(1)
            q4_rows(t, xbuf, y)
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
            Worker(core_fn, [fin.cons(), fout.prod(), xbuf, cnt, load_x, q4_rows], tile=Tile(col=col, row=row))
        )

    n_bytes = N // RB * TB
    w_taps = TensorAccessPattern.full((n_bytes,)).partition(LANES)
    y_taps = TensorAccessPattern.full((LANES * rows_max,)).partition(LANES)

    def sequence(w, x, y, in_hs, out_hs):
        for i in range(LANES):
            in_hs[i].fill(x)
            in_hs[i].fill(w, w_taps[i])
        for i in range(LANES):
            out_hs[i].drain(y, y_taps[i], wait=True)

    out_ty = np.ndarray[(LANES * rows_max,), np.dtype[np.float32]]
    rt = Runtime(sequence, [np.ndarray[(n_bytes,), u8], tile_ty, out_ty, ins, outs])
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
