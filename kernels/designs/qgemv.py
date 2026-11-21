"""Quantized GEMV for decode on npu1: y[N] (fp32) = W[N, K] . x[K] (bf16), with W in GGUF Q4_K or Q6_K, repacked
at load by xdna1_llm.q4k / xdna1_llm.q6k. Kernel math and tile layout: qgemv.cc.

Weights stream over 8 lanes (two shim channels per column, cores in rows 2 and 3); each lane owns N/8
contiguous rows, in blocks of 16 rows, each block split along K into fixed-size tiles. The core carries partial
sums across a block's tiles and streams the block's 16 outputs back as soon as it is done.

x travels on the same lane ahead of the weights, with a header (blocks, tiles per block, format, K). So the core
program depends on nothing: every build has the same cores, and one context runs every format and shape; only
the instruction stream (the host-side DMA program, sized by -Q -N -K) differs per matrix.

--glu builds an FFN gate + up program (rows stored in 16-row blocks gate, up, ...; the host sets the glu flag in
the header): the cores emit bf16(silu(gate) * up) after each up block, and the output goes into one buffer laid
out [junk: N/4 words][the down projection's x buffer][junk], so the down GEMV (reading x through a sub-buffer
at word N/4) can run right after it, queued, with no host work in between. Its header is the host's, written
once; only act and its slack land around it.

    python qgemv.py --dev npu -Q 4 -N 2048 -K 2048 [--geometry q6only] [--glu] --xclbin-path X --insts-path Y

--geometry picks the tile size (xdna1_llm.q4k.Geometry): `unified` serves both formats in one program, `q6only`
uses 512-column Q6_K tiles with no padding for LFM3's K = 2048 and 7680.
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

LANES, RB = 8, 16  # lanes, rows per block
K_MAX = 7680  # must match qgemv.cc
# geometry -> (tile bytes, columns per tile for Q4_K, for Q6_K); must match xdna1_llm.q4k.GEOMETRIES
GEOMETRIES = {"unified": (10944, 1024, 768), "q6only": (7296, 512, 512)}
SB_BYTES = {4: 2688, 6: 3648}  # bytes per super-block of 16 rows; must match xdna1_llm.q4k.SB_BYTES
SRC = str(Path(__file__).parent / "qgemv.cc")


@iron.jit
def qgemv(
    W: In,
    X: In,
    Y: Out,
    *,
    Q: CompileTime[int],
    N: CompileTime[int],
    K: CompileTime[int],
    G: CompileTime[str],
    GLU: CompileTime[bool] = False,
):
    tile_bytes, kc4, kc6 = GEOMETRIES[G]
    assert Q in (4, 6) and N % (LANES * RB) == 0 and K % 256 == 0 and K <= K_MAX, (Q, N, K)
    n_tiles = -(-K // (kc4 if Q == 4 else kc6))
    x_tiles = -(-2 * K_MAX // tile_bytes)
    geo = [] if G == "unified" else [f"-DTILE_BYTES={tile_bytes}", f"-DKC4={kc4}", f"-DKC6={kc6}"]

    u8 = np.dtype[np.uint8]
    tile_ty = np.ndarray[(tile_bytes,), u8]
    x_ty = np.ndarray[(K_MAX,), np.dtype[bfloat16]]
    hdr_ty = np.ndarray[(2,), np.dtype[np.int32]]
    y_ty = np.ndarray[(RB,), np.dtype[np.float32]]
    load_x = ExternalFunction(
        "qg_load_x", source_file=SRC, arg_types=[tile_ty, x_ty, hdr_ty], compile_flags=["-DQG_FN=0", *geo]
    )
    rows = ExternalFunction(
        "qg_rows",
        source_file=SRC,
        arg_types=[tile_ty, x_ty, y_ty],
        compile_flags=["-DQG_FN=1", *geo],
        include_dirs=[str(Path(__file__).parent)],  # silu_lut.h
    )

    def core_fn(of_in, of_out, xbuf, hdr, load_x, rows):
        for _ in range_(x_tiles):  # x and the header
            t = of_in.acquire(1)
            load_x(t, xbuf, hdr)
            of_in.release(1)
        for _ in range_(hdr[0]):  # 16-row blocks
            y = of_out.acquire(1)
            for _ in range_(hdr[1]):  # tiles per block
                t = of_in.acquire(1)
                rows(t, xbuf, y)
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
        hdr = Buffer(type=hdr_ty, name=f"h{lane}")
        workers.append(Worker(core_fn, [fin.cons(), fout.prod(), xbuf, hdr, load_x, rows], tile=Tile(col=col, row=row)))

    # Q4_K tiles are 4 super-blocks (10752 bytes) back to back, shorter than the tile_bytes the cores take:
    # each lane reads tile_bytes per tile at the format's stride (the overrun is the next tile's start, which
    # the core ignores, and the host pads the buffer), in pieces of one Q6_K super-block (<= 1023 words per dim).
    tq = (kc4 if Q == 4 else kc6) // 256 * SB_BYTES[Q]
    lane_tiles = N // RB // LANES * n_tiles
    n_bytes = N // RB * n_tiles * tq + tile_bytes
    piece = SB_BYTES[6]
    w_taps = [
        TensorAccessPattern(
            (n_bytes,), lane * lane_tiles * tq, [lane_tiles, tile_bytes // piece, piece], [tq, piece, 1]
        )
        for lane in range(LANES)
    ]
    y_words = N
    y_taps = TensorAccessPattern.full((N,)).partition(LANES)
    if GLU:  # per lane, pair of blocks, half: gate block -> junk, up block's first 8 words (act) -> down's x
        act, xw = N // 4, x_tiles * tile_bytes // 4  # act and the x buffer, in words
        pairs = N // (2 * RB * LANES)  # per lane
        y_words = 2 * act + act + xw
        y_taps = [
            TensorAccessPattern((y_words,), lane * pairs * 8, [pairs, 2, 2, 8], [8, act, act + xw, 1])
            for lane in range(LANES)
        ]

    def sequence(w, x, y, in_hs, out_hs):
        for i in range(LANES):
            in_hs[i].fill(x)
            in_hs[i].fill(w, w_taps[i])
        for i in range(LANES):
            out_hs[i].drain(y, y_taps[i], wait=True)

    x_all = np.ndarray[(x_tiles * tile_bytes,), u8]
    y_all = np.ndarray[(y_words,), np.dtype[np.float32]]
    rt = Runtime(sequence, [np.ndarray[(n_bytes,), u8], x_all, y_all, ins, outs])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="qgemv")
    add_compile_args(p)
    p.add_argument("-Q", type=int, choices=(4, 6), required=True)
    p.add_argument("--geometry", choices=sorted(GEOMETRIES), default="unified")
    p.add_argument("-N", type=int, required=True)
    p.add_argument("-K", type=int, required=True)
    p.add_argument("--glu", action="store_true", help="FFN gate + up: emit silu(gate) * up into the down's x")
    opts = p.parse_args()
    run_design_cli(
        qgemv,
        opts,
        compile_kwargs=lambda o: dict(Q=o.Q, N=o.N, K=o.K, G=o.geometry, GLU=o.glu),
        run_and_verify=lambda o: None,
        device=lambda o: device_from_args(o, n_cols=4),
    )


if __name__ == "__main__":
    main()
