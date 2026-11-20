"""Power / energy for LFM3 256-token prefill: NPU engine vs llama.cpp CPU-only.

Sensors: Windows "Energy Meter" counters = AMD RAPL (mW). RAPL_Package0_PKG covers the whole SoC
(CPU cores + iGPU + NPU + fabric); the *_CORE instances are CPU cores only. The dGPU is logged via nvidia-smi.
Sampling: typeperf at 1 s, each phase runs DUR seconds (default 30).

Phases: idle | cpu_llama (llama-server, CUDA hidden, 8 threads) | npu_mm (only the 567 NPU GEMM calls) | npu_full

    python benchmarks/power_bench.py --gguf <LFM3 BF16 gguf> --llama-server <llama-server.exe>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time

import numpy as np

from xdna1_llm.gguf_weights import GGUFWeights
from xdna1_llm.models.lfm3 import LFM3
from xdna1_llm.reference import LlamaServer

COUNTERS = [r"\Energy Meter(RAPL_Package0_PKG)\Power", r"\Processor(_Total)\% Processor Time"] + [
    rf"\Energy Meter(RAPL_Package0_Core{i}_CORE)\Power" for i in range(8)
]
PROMPT = (
    " ".join(["The NPU in this laptop is a first-generation AMD XDNA engine with sixteen AI Engine tiles."] * 13)
    + " In one word, the engine is made by"
)


class Sampler:
    def __init__(self, tag, outdir):
        self.csv = os.path.join(outdir, f"power_{tag}.csv")
        if os.path.exists(self.csv):
            os.remove(self.csv)
        self.p = subprocess.Popen(
            ["typeperf", *COUNTERS, "-si", "1", "-f", "CSV", "-o", self.csv, "-y"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            self.gpu = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-lms", "1000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except FileNotFoundError:
            self.gpu = None
        time.sleep(2.5)

    def stop(self):
        self.p.terminate()
        gpu_out = ""
        if self.gpu:
            self.gpu.terminate()
            gpu_out = self.gpu.communicate()[0]
        self.p.wait()
        time.sleep(0.5)
        rows = list(csv.reader(open(self.csv, newline="")))[2:]
        v = np.array([[float(x) if x.strip() else np.nan for x in r[1:]] for r in rows], dtype=float)
        gpu = [float(x) for x in gpu_out.split() if x.replace(".", "", 1).isdigit()]
        return dict(
            pkg_w=float(np.nanmean(v[:, 0]) / 1000),
            cpu_pct=float(np.nanmean(v[:, 1])),
            cores_w=float(np.nanmean(np.nansum(v[:, 2:], axis=1)) / 1000),
            samples=len(rows),
            dgpu_w=float(np.mean(gpu)) if gpu else None,
        )


def phase(tag, work, dur, outdir):
    print(f"-- {tag}: {dur} s", flush=True)
    s = Sampler(tag, outdir)
    t0, toks = time.perf_counter(), 0
    while time.perf_counter() - t0 < dur:
        toks += work()
    wall = time.perf_counter() - t0
    r = s.stop()
    r.update(tag=tag, wall_s=wall, tokens=toks, tok_s=toks / wall)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--llama-server", required=True)
    ap.add_argument("--dur", type=int, default=30)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    res = {}
    srv = LlamaServer().start(a.llama_server, a.gguf, cpu_only=True)
    ids = srv.tokenize(PROMPT)[-256:]
    T = len(ids)
    time.sleep(5)
    res["idle"] = phase("idle", lambda: (time.sleep(1.0), 0)[1], a.dur, a.out)
    srv.timings(ids, 1)
    res["cpu_llama"] = phase("cpu_llama", lambda: (srv.timings(ids, 1), T)[1], a.dur, a.out)
    srv.stop()
    time.sleep(10)

    from xdna1_llm.npu.device import hard_exit
    from xdna1_llm.npu.linear import NpuLinear

    lin = NpuLinear()
    m = LFM3(GGUFWeights(a.gguf), lin)
    m.forward(ids)
    xs = {K: np.random.default_rng(0).standard_normal((T, K), dtype=np.float32) for K in lin.k}

    def npu_mm():
        for name, (K, _) in lin.chunks.items():
            lin(name, xs[K])
        return T

    res["npu_mm"] = phase("npu_mm", npu_mm, a.dur, a.out)
    time.sleep(5)
    res["npu_full"] = phase("npu_full", lambda: (m.forward(ids), T)[1], a.dur, a.out)

    idle = res["idle"]["pkg_w"]
    print("\nphase      tok/s   pkg W  (+idle)  cores W  CPU %  dGPU W  J/tok  J/tok over idle")
    for k in ("idle", "npu_mm", "npu_full", "cpu_llama"):
        r = res[k]
        jt = r["pkg_w"] / r["tok_s"] if r["tok_s"] else float("nan")
        ji = (r["pkg_w"] - idle) / r["tok_s"] if r["tok_s"] else float("nan")
        r.update(j_per_tok=jt, j_per_tok_over_idle=ji)
        g = f"{r['dgpu_w']:.1f}" if r["dgpu_w"] is not None else "n/a"
        print(
            f"{k:10s} {r['tok_s']:6.1f}  {r['pkg_w']:6.2f}  {r['pkg_w'] - idle:+6.2f}  {r['cores_w']:6.2f}  "
            f"{r['cpu_pct']:5.1f}  {g:>6s}  {jt:.4f}  {ji:.4f}"
        )
    json.dump(res, open(os.path.join(a.out, "power_results.json"), "w"), indent=2)
    hard_exit(0)


if __name__ == "__main__":
    main()
