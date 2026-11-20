r"""What does quantization cost, and does our engine add anything on top? Scores next-token predictions on a text.

    python benchmarks/eval_quality.py --base LFM3-1.7B-BF16.gguf --quant LFM3-1.7B-Q4_K_M.gguf [more GGUFs] \
        --llama-server llama-server.exe [--text benchmarks/data/eval.txt] [--npu]

Variants, all compared with `bf16` (the original weights in fp32 on the CPU, the reference):
  * one per --quant GGUF, named after its type (q4_k_m, q6_k, ...), in fp32 on the CPU: the cost of the format;
  * with --npu, the first one again through the NPU decode path (`q4_k_m-npu`): its line against the CPU run of
    the same file is the cost of our engine, which should be ~0.
Each chunk of the text starts from an empty context (metrics: xdna1_llm.quality). The CPU variants are slow
(weights are dequantized on every call): ~1 min per 256-token chunk.
"""

import argparse
import os
import sys
import time
import traceback

import numpy as np

from xdna1_llm.cpu_linear import CpuLinear
from xdna1_llm.gguf_weights import GGUFWeights
from xdna1_llm.models.lfm3 import LFM3
from xdna1_llm.quality import Score, chunks, log_softmax
from xdna1_llm.tokenizer import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


def prefill_logits(m, chunk):
    return m.forward(chunk[:-1], all_logits=True)[0]


def decode_logits(m, chunk):
    st = m.new_state(len(chunk))
    return np.stack([m.step(int(t), st) for t in chunk[:-1]])


def main():
    code = 0
    try:
        evaluate()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        code = 1
    if "xdna1_llm.npu.device" in sys.modules:  # pyxrt objects can access-violate when destroyed at exit
        sys.modules["xdna1_llm.npu.device"].hard_exit(code)
    sys.exit(code)


def evaluate():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="BF16 GGUF (the reference)")
    ap.add_argument("--quant", required=True, nargs="+", help="quantized GGUFs (Q4_K_M, Q6_K, ...)")
    ap.add_argument("--text", default=os.path.join(HERE, "data", "eval.txt"))
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--tokens", type=int, help="score at most this many tokens of the text")
    ap.add_argument("--npu", action="store_true", help="also run the first --quant on the NPU decode path")
    ap.add_argument("--kernels", help="directory with built xclbins")
    a = ap.parse_args()

    base = GGUFWeights(a.base)
    tok = Tokenizer.from_gguf(base)
    with open(a.text, encoding="utf-8") as f:
        parts = chunks(tok.encode(f.read(), add_bos=False), tok.bos, a.chunk, a.tokens)
    print(f"{sum(len(c) - 1 for c in parts)} scored tokens in {len(parts)} chunks of up to {a.chunk}", flush=True)

    variants = {"bf16": (LFM3(base, CpuLinear()), prefill_logits)}
    for path in a.quant:
        name = os.path.splitext(os.path.basename(path))[0].rsplit("-", 1)[-1].lower()  # LFM3-1.7B-Q4_K_M -> q4_k_m
        variants[name] = (LFM3(GGUFWeights(path), CpuLinear()), prefill_logits)
    first = list(variants)[1]
    pairs = [(v, "bf16") for v in variants]
    if a.npu:
        from xdna1_llm.npu.qlinear import NpuQLinear  # imports pyxrt

        w = variants[first][0].w
        variants[first + "-npu"] = (LFM3(w, NpuQLinear(kernel_dir=a.kernels, formats=w.types())), decode_logits)
        pairs += [(first + "-npu", "bf16"), (first + "-npu", first)]
    scores = {p: Score() for p in pairs}

    for n, c in enumerate(parts):
        t0 = time.perf_counter()
        lp = {v: log_softmax(run(m, c)) for v, (m, run) in variants.items()}
        for v, ref in pairs:
            scores[v, ref].add(lp[ref], lp[v], c[1:])
        print(f"chunk {n + 1}/{len(parts)}: {time.perf_counter() - t0:.0f} s", flush=True)

    print("\n| model | vs | tokens | perplexity | mean KLD | KLD p99 | same top-1 |")
    print("|---|---|---|---|---|---|---|")
    for (v, ref), s in scores.items():
        r = s.summary()
        cells = "— | — | —" if v == ref else f"{r['kld']:.5f} | {r['kld_p99']:.4f} | {r['top1'] * 100:.2f}%"
        print(f"| {v} | {ref} | {r['tokens']} | {r['ppl']:.3f} | {cells} |")


if __name__ == "__main__":
    main()
