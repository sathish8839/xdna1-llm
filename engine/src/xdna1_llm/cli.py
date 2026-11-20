"""Command line: prefill a prompt on the NPU (or CPU reference), or generate from it, optionally next to llama.cpp.

  python -m xdna1_llm.cli --gguf LFM3-1.7B-BF16.gguf --prompt "The capital of France is" --compare
  python -m xdna1_llm.cli --gguf LFM3-1.7B-Q4_K_M.gguf --chat --prompt "Name three rivers." --generate 200 --sample

Tokenization and the chat template come from the GGUF. llama.cpp is optional, as a reference: pass --llama-server
PATH to start one, or --server URL for a running one, to get its [llama.cpp] lines.
"""

from __future__ import annotations

import argparse
import codecs
import os
import time

import numpy as np

from . import chat
from .cpu_linear import CpuLinear
from .engine import gemm_backend
from .gguf_weights import GGUFWeights
from .models.lfm3 import LFM3
from .reference import LlamaServer
from .sampling import Sampler
from .tokenizer import Tokenizer

M_TILE = 256


def main(argv=None):
    ap = argparse.ArgumentParser(prog="xdna1-llm")
    ap.add_argument("--gguf", default=os.environ.get("XDNA1_LLM_GGUF"), help="path to an LFM3 GGUF")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--chat", action="store_true", help="send --prompt as a user message through the chat template")
    ap.add_argument("--no-think", action="store_true", help="with --chat: answer without a thinking block")
    ap.add_argument("--backend", choices=("npu", "cpu"), default="npu")
    ap.add_argument("--compare", action="store_true", help="also run the CPU reference and compare per layer")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--server", default="http://127.0.0.1:18080", help="a running llama-server to compare with")
    ap.add_argument("--llama-server", help="path to llama-server.exe to start as the reference")
    ap.add_argument("--kernels", help="directory with built xclbins (default: kernels/build or $XDNA1_LLM_KERNELS)")
    ap.add_argument(
        "--generate", type=int, metavar="N", help="decode up to N tokens greedily (Q4_K_M or Q6_K GGUF, NPU GEMVs)"
    )
    ap.add_argument(
        "--no-gemm-prompt",
        action="store_true",
        help="with --generate on the NPU: run the prompt token by token on the quantized GEMV instead of keeping a "
        "bf16 copy of the weights (~2.9 GB) for the batched GEMM",
    )
    g = ap.add_argument_group("sampling (with --generate; greedy unless one of these is given)")
    g.add_argument("--sample", action="store_true", help="sample with the model's recommended settings (from the GGUF)")
    g.add_argument("--temperature", type=float)
    g.add_argument("--top-k", type=int)
    g.add_argument("--top-p", type=float)
    g.add_argument("--min-p", type=float)
    g.add_argument("--repeat-penalty", type=float)
    g.add_argument("--seed", type=int)
    a = ap.parse_args(argv)
    if not a.gguf:
        ap.error("--gguf (or XDNA1_LLM_GGUF) is required")

    w = GGUFWeights(a.gguf)
    tok = Tokenizer.from_gguf(w)
    text = chat.render([{"role": "user", "content": a.prompt}], enable_thinking=not a.no_think) if a.chat else a.prompt
    ids = tok.encode(text)
    if not ids:
        ap.error("the prompt is empty")
    srv = reference_server(a)
    if a.generate:
        return generate(a, w, tok, srv, ids)
    if len(ids) > M_TILE:
        print(f"prompt has {len(ids)} tokens; keeping the last {M_TILE} (single-tile prefill)")
        ids = ids[-M_TILE:]
    print(f"prompt: {len(ids)} tokens")

    backends = ["npu", "cpu"] if a.compare else [a.backend]
    results, keep = {}, []
    for be in backends:
        if be == "npu":
            from .npu.linear import NpuLinear  # imports pyxrt

            lin = NpuLinear(kernel_dir=a.kernels)
        else:
            lin = CpuLinear()
        m = LFM3(w, lin)
        print(f"[{be}] weights ready in {m.load_s:.1f} s", flush=True)
        for r in range(a.repeat):
            m.cpu_ms, lin.npu_ms, lin.calls = 0.0, 0.0, 0
            t0 = time.perf_counter()
            logits, hidden = m.forward(ids, keep_hidden=a.compare)
            wall = (time.perf_counter() - t0) * 1e3
            extra = f"; npu linear {lin.npu_ms:.0f} ms / {lin.calls} calls" if be == "npu" else ""
            print(
                f"[{be}] run {r}: {len(ids)} tok in {wall:.0f} ms ({len(ids) / wall * 1e3:.1f} tok/s); "
                f"cpu ops {m.cpu_ms:.0f} ms{extra}",
                flush=True,
            )
        top = np.argsort(-logits)[:5]
        print(f"[{be}] top-5 next: " + ", ".join(f"{tok.decode([t], True)!r}({logits[t]:.2f})" for t in top))
        results[be] = (logits, hidden)
        keep.append((m, lin))  # keep pyxrt objects alive until hard exit

    if a.compare:
        (ln, hn), (lc, hc) = results["npu"], results["cpu"]
        for i in sorted({0, 2, 4, len(hn) // 2, len(hn) - 1}):
            cs = float(np.sum(hn[i] * hc[i]) / (np.linalg.norm(hn[i]) * np.linalg.norm(hc[i])))
            print(f"  layer {i:2d} ({m.kind(i)}): cosine {cs:.6f}")
        cs = float(ln @ lc / (np.linalg.norm(ln) * np.linalg.norm(lc)))
        print(f"  logits cosine {cs:.6f}; top-1 match: {int(np.argmax(ln)) == int(np.argmax(lc))}")
    if srv:
        print("[llama.cpp] top-5 next: " + ", ".join(f"{t!r}({p:.2f})" for t, p in srv.top_next(ids)))
        srv.stop()
    if "npu" in backends:
        from .npu.device import hard_exit

        hard_exit(0)


def reference_server(a) -> LlamaServer | None:
    """The llama-server to compare with: started from --llama-server, or one already running at --server."""
    srv = LlamaServer(a.server)
    if not srv.healthy():
        if not a.llama_server:
            return None
        print("starting llama-server (CPU only) ...", flush=True)
        srv.start(a.llama_server, a.gguf)
    loaded = srv.model_path()
    if loaded and os.path.normcase(os.path.abspath(loaded)) != os.path.normcase(os.path.abspath(a.gguf)):
        print(f"note: the llama-server at {a.server} has {loaded} loaded; [llama.cpp] lines are from that model")
    return srv


def sampler(a, w) -> Sampler | None:
    """None (greedy) unless a sampling option was given; options override the GGUF's recommended values."""
    opts = {"temperature": a.temperature, "top_k": a.top_k, "top_p": a.top_p, "min_p": a.min_p,
            "repetition_penalty": a.repeat_penalty, "seed": a.seed}  # fmt: skip
    if not a.sample and all(v is None for k, v in opts.items() if k != "seed"):
        return None
    return Sampler.from_gguf(w, **opts)


def generate(a, w, tok, srv, ids):
    """Decode on the NPU (or the CPU reference), print the text as it comes, and compare with llama.cpp if given."""
    pick = sampler(a, w)
    if a.backend == "npu":
        from .npu.qlinear import NpuQLinear  # imports pyxrt

        lin = NpuQLinear(kernel_dir=a.kernels, formats=w.types())
        prefill_lin = None if a.no_gemm_prompt else gemm_backend(lin.npu, lin.kdir)
    else:
        lin, prefill_lin = CpuLinear(), None
    m = LFM3(w, lin, prefill_lin)
    repack = f" (repack {lin.repack_s:.1f} s, {lin.geo.name} tiles)" if a.backend == "npu" else ""
    prompt_on = "bf16 GEMM" if prefill_lin else "decode GEMV"
    print(
        f"[{a.backend}] weights ready in {m.load_s:.1f} s{repack}; prompt {len(ids)} tokens on the {prompt_on}",
        flush=True,
    )
    if pick:
        print(f"[{a.backend}] {pick}", flush=True)

    m.cpu_ms, lin.npu_ms, lin.calls = 0.0, 0.0, 0
    logits = []
    utf8 = codecs.getincrementaldecoder("utf-8")("replace")  # a character can span tokens

    def show(t):
        if not shown:  # the first token comes right after the prompt: count decode work from here
            m.cpu_ms, lin.npu_ms, lin.calls = 0.0, 0.0, 0
            shown.append(t)
        print(utf8.decode(tok.token_bytes(t)), end="", flush=True)

    shown = []
    out, st, t_prompt, t_gen = m.generate(ids, a.generate, on_token=show, logits_out=logits, stop=tok.eog, sample=pick)
    n_steps = max(len(out) - 1, 1)
    print(
        f"\n[{a.backend}] {len(out)} tokens; prompt {len(ids)} tokens in {t_prompt:.2f} s ({len(ids) / t_prompt:.1f} "
        f"tok/s), generation {(len(out) - 1) / max(t_gen, 1e-9):.1f} tok/s; per generated token: linears "
        f"{lin.npu_ms / n_steps:.1f} ms in {lin.calls / n_steps:.0f} calls, cpu ops {m.cpu_ms / n_steps:.1f} ms"
    )
    if a.compare and a.backend == "npu":  # the exact fp32 CPU reference along the same tokens
        cpu = LFM3(w, CpuLinear())
        cst = cpu.new_state(st.max_len)
        for t in ids[:-1]:
            cpu.step(int(t), cst)
        path = [ids[-1], *out[:-1]]
        for i, t in enumerate(path):
            ref_logits = cpu.step(int(t), cst)
            cs = float(logits[i] @ ref_logits / (np.linalg.norm(logits[i]) * np.linalg.norm(ref_logits)))
            same = int(np.argmax(logits[i])) == int(np.argmax(ref_logits))
            print(f"  token {i}: logits cosine vs fp32 CPU {cs:.6f}, same top-1 {same}", flush=True)
    if srv and pick:
        print("[llama.cpp] skipped: sampled output cannot be compared token for token")
        srv.stop()
    elif srv:
        ref = srv.generate(ids, a.generate)
        print(f"[llama.cpp] {tok.decode(ref)!r}")
        same = next((i for i, (x, y) in enumerate(zip(out, ref, strict=False)) if x != y), min(len(out), len(ref)))
        print(f"first {same} of {len(ref)} tokens identical to llama.cpp")
        srv.stop()
    if a.backend == "npu":
        from .npu.device import hard_exit

        hard_exit(0)


if __name__ == "__main__":
    main()
