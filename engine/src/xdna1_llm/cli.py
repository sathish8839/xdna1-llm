"""Command line: prefill a prompt on the NPU (or CPU reference) and compare with llama.cpp.

  python -m xdna1_llm.cli --gguf LFM3-1.7B-BF16.gguf --prompt "The capital of France is" --compare

Needs a llama-server for tokenization: pass --llama-server PATH to start one, or --server URL for a running one.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from .cpu_linear import CpuLinear
from .gguf_weights import GGUFWeights
from .models.lfm3 import LFM3
from .reference import LlamaServer

M_TILE = 256


def main(argv=None):
    ap = argparse.ArgumentParser(prog="xdna1-llm")
    ap.add_argument("--gguf", default=os.environ.get("XDNA1_LLM_GGUF"), help="path to a BF16 LFM3 GGUF")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--backend", choices=("npu", "cpu"), default="npu")
    ap.add_argument("--compare", action="store_true", help="also run the CPU reference and compare per layer")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--server", default="http://127.0.0.1:18080")
    ap.add_argument("--llama-server", help="path to llama-server.exe to start for tokenization/reference")
    ap.add_argument("--kernels", help="directory with built xclbins (default: kernels/build or $XDNA1_LLM_KERNELS)")
    ap.add_argument(
        "--generate", type=int, metavar="N", help="decode N tokens greedily (Q4_K_M or Q6_K GGUF, NPU GEMVs)"
    )
    a = ap.parse_args(argv)
    if not a.gguf:
        ap.error("--gguf (or XDNA1_LLM_GGUF) is required")

    srv = LlamaServer(a.server)
    if a.llama_server and not srv.healthy():
        print("starting llama-server (CPU only) ...", flush=True)
        srv.start(a.llama_server, a.gguf)
    loaded = srv.model_path()
    if loaded and os.path.normcase(os.path.abspath(loaded)) != os.path.normcase(os.path.abspath(a.gguf)):
        print(f"note: the llama-server at {a.server} has {loaded} loaded; [llama.cpp] lines are from that model")
    ids = srv.tokenize(a.prompt)
    if a.generate:
        return generate(a, srv, ids)
    if len(ids) > M_TILE:
        print(f"prompt has {len(ids)} tokens; keeping the last {M_TILE} (single-tile prefill)")
        ids = ids[-M_TILE:]
    print(f"prompt: {len(ids)} tokens")

    w = GGUFWeights(a.gguf)
    vocab = w.tokens()
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
        print(f"[{be}] top-5 next: " + ", ".join(f"{vocab[t]!r}({logits[t]:.2f})" for t in top))
        results[be] = (logits, hidden)
        keep.append((m, lin))  # keep pyxrt objects alive until hard exit

    if a.compare:
        (ln, hn), (lc, hc) = results["npu"], results["cpu"]
        for i in sorted({0, 2, 4, len(hn) // 2, len(hn) - 1}):
            cs = float(np.sum(hn[i] * hc[i]) / (np.linalg.norm(hn[i]) * np.linalg.norm(hc[i])))
            print(f"  layer {i:2d} ({m.kind(i)}): cosine {cs:.6f}")
        cs = float(ln @ lc / (np.linalg.norm(ln) * np.linalg.norm(lc)))
        print(f"  logits cosine {cs:.6f}; top-1 match: {int(np.argmax(ln)) == int(np.argmax(lc))}")
    print("[llama.cpp] top-5 next: " + ", ".join(f"{t!r}({p:.2f})" for t, p in srv.top_next(ids)))
    srv.stop()
    if "npu" in backends:
        from .npu.device import hard_exit

        hard_exit(0)


def generate(a, srv, ids):
    """Decode on the NPU (or the CPU reference) and compare the tokens with llama.cpp's greedy output."""
    w = GGUFWeights(a.gguf)
    if a.backend == "npu":
        from .npu.qlinear import NpuQLinear  # imports pyxrt

        lin = NpuQLinear(kernel_dir=a.kernels, formats=w.types())
    else:
        lin = CpuLinear()
    m = LFM3(w, lin)
    repack = f" (repack {lin.repack_s:.1f} s, {lin.geo.name} tiles)" if a.backend == "npu" else ""
    print(f"[{a.backend}] weights ready in {m.load_s:.1f} s{repack}; prompt {len(ids)} tokens", flush=True)

    m.cpu_ms, lin.npu_ms, lin.calls = 0.0, 0.0, 0
    logits = []
    dot = lambda t: print(".", end="", flush=True)  # noqa: E731
    out, st, t_prompt, t_gen = m.generate(ids, a.generate, on_token=dot, logits_out=logits)
    n_steps = len(ids) + len(out) - 1
    print(f"\n[{a.backend}] {srv.detokenize(out)!r}")
    print(
        f"[{a.backend}] prompt {len(ids) / t_prompt:.1f} tok/s, generation {(len(out) - 1) / t_gen:.1f} tok/s; "
        f"per token: linears {lin.npu_ms / n_steps:.1f} ms in {lin.calls / n_steps:.0f} calls, "
        f"cpu ops {m.cpu_ms / n_steps:.1f} ms"
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
    ref = srv.generate(ids, a.generate)
    print(f"[llama.cpp] {srv.detokenize(ref)!r}")
    same = next((i for i, (x, y) in enumerate(zip(out, ref, strict=False)) if x != y), min(len(out), len(ref)))
    print(f"first {same} of {len(ref)} tokens identical to llama.cpp")
    srv.stop()
    if a.backend == "npu":
        from .npu.device import hard_exit

        hard_exit(0)


if __name__ == "__main__":
    main()
