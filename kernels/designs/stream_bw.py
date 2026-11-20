"""DDR -> core streaming bandwidth probe for npu1.

Streams `total_bytes` from host memory into `lanes` compute cores (lane l: shim column l % 4, core row 2 + l // 4),
`tile_bytes` at a time, with no compute. Each core writes the first word of its last tile back so the host can
check the data arrived. This bounds what any decode GEMV design can reach.

    python stream_bw.py --dev npu --lanes 8 --tile-kb 16 --total-mb 16 --xclbin-path X --insts-path Y
"""

import argparse

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli

OUT_WORDS = 16


@iron.jit
def stream_bw(
    A: In,
    C: Out,
    *,
    lanes: CompileTime[int],
    tile_words: CompileTime[int],
    iters: CompileTime[int],
):
    total_words = lanes * iters * tile_words
    a_ty = np.ndarray[(total_words,), np.dtype[np.int32]]
    c_ty = np.ndarray[(lanes * OUT_WORDS,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(tile_words,), np.dtype[np.int32]]
    out_ty = np.ndarray[(OUT_WORDS,), np.dtype[np.int32]]

    def core_fn(of_in, of_out):
        out = of_out.acquire(1)
        for _ in range_(iters):
            t = of_in.acquire(1)
            out[0] = t[0]
            of_in.release(1)
        of_out.release(1)

    ins, outs, workers = [], [], []
    for lane in range(lanes):
        col, row = lane % 4, 2 + lane // 4
        fin = ObjectFifo(tile_ty, name=f"in{lane}", depth=2)
        fout = ObjectFifo(out_ty, name=f"out{lane}", depth=1)
        ins.append(fin.prod(tile=Tile(col=col, row=0)))
        outs.append(fout.cons(tile=Tile(col=col, row=0)))
        workers.append(Worker(core_fn, [fin.cons(), fout.prod()], tile=Tile(col=col, row=row)))

    a_taps = TensorAccessPattern.full((total_words,)).partition(lanes)
    c_taps = TensorAccessPattern.full((lanes * OUT_WORDS,)).partition(lanes)

    def sequence(a, c, in_hs, out_hs):
        for i in range(lanes):
            in_hs[i].fill(a, a_taps[i])
        for i in range(lanes):
            out_hs[i].drain(c, c_taps[i], wait=True)

    rt = Runtime(sequence, [a_ty, c_ty, ins, outs])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="stream_bw")
    add_compile_args(p)
    p.add_argument("--lanes", type=int, default=8)
    p.add_argument("--tile-kb", type=int, default=16)
    p.add_argument("--total-mb", type=int, default=16)
    opts = p.parse_args()
    tile_words = opts.tile_kb * 1024 // 4
    iters = opts.total_mb * 1024 * 1024 // (opts.tile_kb * 1024 * opts.lanes)

    run_design_cli(
        stream_bw,
        opts,
        compile_kwargs=lambda o: dict(lanes=o.lanes, tile_words=tile_words, iters=iters),
        run_and_verify=lambda o: None,
        device=lambda o: device_from_args(o, n_cols=4),  # whole npu1 array
    )


if __name__ == "__main__":
    main()
