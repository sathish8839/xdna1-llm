"""W4A16 GEMM for prompts on npu1: C[256, N] (fp32) = A[256, K] (bf16) . W[N, K]^T, with W in GGUF Q4_K or
Q6_K as the decode GEMV holds it (xdna1_llm.q4k / q6k, qgemv.cc). Prompts read the same device buffers as
decode, so no second copy of the weights is needed.

A 4 x 4 array as in mlir-aie's whole_array GEMM: core (row r, column c) computes rows 64r .. 64r + 63 of C for
32 weight rows at a time (two 16-row blocks); array column c takes the 32-row pairs c, c + 4, c + 8, ...
Activation tiles (64 x 64) are broadcast along an array row, weights down an array column, one super-block
(256 columns of both 16-row blocks) per object. For each 64-column slice of it the core makes the bf16 B
operand (qgemm.cc) and runs mlir-aie's bf16 matmul kernel on it.

The cores take their trip counts and the format as runtime parameters, so every build has the same cores and
one context runs every format and shape; only the instruction stream (sized by -Q -N -K) differs per matrix.

--glu builds an FFN gate + up program: the weight's rows are gate and up interleaved in 16-row blocks, so each
core's 32 columns are 16 gate and the 16 matching up columns, and the core turns its tile into
silu(gate) * up in bf16 (qgemm.cc, qm_finish). The output is then act[256, N / 2] in bf16, the A operand of the
down projection, at the start of the C buffer (whose size is unchanged): the C stream carries 32 words per
row, the first 8 are act and the other 24 go to the rest of the buffer.

    python qgemm.py --dev npu -Q 4 -N 2048 -K 2048 [--geometry q6only] [--glu] --xclbin-path X --insts-path Y
"""

import argparse
from pathlib import Path

import aie.iron as iron
import aie.iron.kernels as kernels
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker, WorkerRuntimeBarrier
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli
from ml_dtypes import bfloat16

M, m, k, n = 256, 64, 64, 32  # rows per call; per core: rows, columns per slice, weight rows
ROWS = COLS = 4
RB = 16
# must match xdna1_llm.q4k: geometry -> (GEMV tile bytes, columns per tile for Q4_K, for Q6_K); super-block bytes
GEOMETRIES = {"unified": (10944, 1024, 768), "q6only": (7296, 512, 512)}
SB_BYTES = {4: 2688, 6: 3648}
WINDOW = SB_BYTES[6]  # bytes per super-block in a B object (a Q4_K one is followed by bytes the core ignores)
MAX_ITER = 64  # shim DMA iteration count: row pairs per fill
SRC = str(Path(__file__).parent / "qgemm.cc")


@iron.jit
def qgemm(
    A: In,
    W: In,
    C: Out,
    *,
    Q: CompileTime[int],
    N: CompileTime[int],
    K: CompileTime[int],
    G: CompileTime[str],
    GLU: CompileTime[bool] = False,
):
    tile_bytes, kc4, kc6 = GEOMETRIES[G]
    kc = kc4 if Q == 4 else kc6
    assert Q in (4, 6) and N % (n * COLS) == 0 and K % 256 == 0, (Q, N, K)
    sbs = SB_BYTES[Q]
    bs = -(-K // kc) * (kc // 256) * sbs  # bytes per 16-row block (the last tile of a row may be padded)
    n_sb = K // 256
    pairs = N // (n * COLS)  # 32-row pairs per array column
    w_bytes = N // RB * bs + tile_bytes  # the GEMV's buffer, with its slack

    mm = kernels.mm(dim_m=m, dim_k=k, dim_n=n, input_dtype=bfloat16, output_dtype=np.float32)
    zero = kernels.zero(m * n, np.float32)
    dims = mm.stream_dims
    a_ty = np.ndarray[(m, k), np.dtype[bfloat16]]
    b_ty = np.ndarray[(2 * WINDOW,), np.dtype[np.uint8]]
    bb_ty = np.ndarray[(k * n,), np.dtype[bfloat16]]
    c_ty = np.ndarray[(m, n), np.dtype[np.float32]]
    c2_ty = np.ndarray[(m * n * ROWS,), np.dtype[np.float32]]
    rtp_ty = np.ndarray[(4,), np.dtype[np.int32]]
    dequant = ExternalFunction(
        "qm_dequant", source_file=SRC, arg_types=[b_ty, bb_ty, rtp_ty], compile_flags=["-DQM_FN=0"]
    )
    finish = ExternalFunction(
        "qm_finish",
        source_file=SRC,
        arg_types=[c_ty, rtp_ty],
        compile_flags=["-DQM_FN=1"],
        include_dirs=[str(Path(__file__).parent)],  # silu_lut.h
    )

    a_l3, b_l3, c_l3 = [], [], []
    a_l1, b_l1, c_l1 = [], [], [[] for _ in range(ROWS)]
    for i in range(COLS):
        fa = ObjectFifo(a_ty, name=f"A{i}", depth=2)  # activation rows 64i .. 64i + 63, for array row i
        a_l3.append(fa)
        a_l1.append(fa.cons().forward(obj_type=a_ty, name=f"A{i}_l1", to_stream=dims.A))
        fb = ObjectFifo(b_ty, name=f"B{i}", depth=2)
        b_l3.append(fb)
        b_l1.append(fb.cons().forward(obj_type=b_ty, name=f"B{i}_l1"))
        fc = ObjectFifo(c2_ty, name=f"C{i}", depth=2, to_stream=dims.C)
        c_l3.append(fc)
        parts = fc.prod().join(
            [m * n * r for r in range(ROWS)],
            obj_types=[c_ty] * ROWS,
            names=[f"C{i}_{r}" for r in range(ROWS)],
            depths=[2] * ROWS,
        )
        for r in range(ROWS):
            c_l1[r].append(parts[r])

    rtps = [[Buffer(rtp_ty, name=f"rtp_{r}_{c}", use_write_rtp=True) for c in range(COLS)] for r in range(ROWS)]
    barriers = [[WorkerRuntimeBarrier() for _ in range(COLS)] for _ in range(ROWS)]

    def core_fn(in_a, in_b, out_c, bbuf, rtp, barrier, zero, matmul, dequant, finish):
        barrier.wait_for_value(1)
        n_out, n_sb = rtp[0], rtp[1]
        barrier.release_with_value(1)
        for _ in range_(n_out):
            c = out_c.acquire(1)
            zero(c)
            for _ in range_(n_sb):
                b = in_b.acquire(1)
                for _ in range_(4):  # 64-column slices
                    dequant(b, bbuf, rtp)
                    a = in_a.acquire(1)
                    matmul(a, bbuf, c)
                    in_a.release(1)
                in_b.release(1)
            finish(c, rtp)
            out_c.release(1)

    workers = Worker.grid(
        ROWS,
        COLS,
        lambda r, c: Worker(
            core_fn,
            [
                a_l1[r].cons(),
                b_l1[c].cons(),
                c_l1[r][c].prod(),
                Buffer(bb_ty, name=f"bb_{r}_{c}"),
                rtps[r][c],
                barriers[r][c],
                zero,
                mm,
                dequant,
                finish,
            ],
            stack_size=0xD00,
        ),
    )

    def sequence(a, w, cm, a_hs, b_hs, c_hs):
        for r in range(ROWS):
            for c in range(COLS):
                rtps[r][c][0] = pairs
                rtps[r][c][1] = n_sb
                rtps[r][c][2] = Q
                rtps[r][c][3] = int(GLU)
                barriers[r][c].set(1)
        for p0 in range(0, pairs, MAX_ITER):
            np_ = min(MAX_ITER, pairs - p0)
            for c in range(COLS):
                # activation rows of array row c, every 64-column tile, once per row pair
                a_hs[c].fill(a, TensorAccessPattern((M * K,), c * m * K, [np_, K // k, m, k], [0, k, K, 1]))
                # per pair: super-block s of block 0, then of block 1
                off = 2 * (c + COLS * p0) * bs
                tap = TensorAccessPattern((w_bytes,), off, [np_, n_sb, 2, WINDOW], [2 * COLS * bs, sbs, bs, 1])
                b_hs[c].fill(w, tap)
            for c in range(COLS):
                if GLU:  # per row: words 0-7 -> act (bf16 pairs), words 8-31 -> three act-sized regions after it
                    row, region = N // 4, M * N // 4  # act row and act size, in words
                    off = n // 4 * (c + COLS * p0)
                    tap = TensorAccessPattern((M * N,), off, [np_, M, 4, n // 4], [n, row, region, 1])
                else:
                    tap = TensorAccessPattern((M * N,), n * (c + COLS * p0), [np_, M, n], [n * COLS, N, 1])
                c_hs[c].drain(cm, tap, wait=True)

    rt = Runtime(
        sequence,
        [
            np.ndarray[(M * K,), np.dtype[bfloat16]],
            np.ndarray[(w_bytes,), np.dtype[np.uint8]],
            np.ndarray[(M * N,), np.dtype[np.float32]],
            [f.prod() for f in a_l3],
            [f.prod() for f in b_l3],
            [f.cons() for f in c_l3],
        ],
    )
    flat = [w for row in workers for w in row]
    return Program(iron.get_current_device(), rt, workers=flat).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="qgemm")
    add_compile_args(p)
    p.add_argument("-Q", type=int, choices=(4, 6), required=True)
    p.add_argument("--geometry", choices=sorted(GEOMETRIES), default="unified")
    p.add_argument("-N", type=int, required=True)
    p.add_argument("-K", type=int, required=True)
    p.add_argument("--glu", action="store_true", help="FFN gate + up: output silu(gate) * up in bf16")
    opts = p.parse_args()
    run_design_cli(
        qgemm,
        opts,
        compile_kwargs=lambda o: dict(Q=o.Q, N=o.N, K=o.K, G=o.geometry, GLU=o.glu),
        run_and_verify=lambda o: None,
        device=lambda o: device_from_args(o, n_cols=4),
    )


if __name__ == "__main__":
    main()
