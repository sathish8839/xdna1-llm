"""llama.cpp `llama-server` as tokenizer and ground truth (until a native GGUF tokenizer lands, see docs/roadmap.md).

Note: CUDA builds of llama.cpp offload large prompt batches to the GPU even with -ngl 0. `start_server` hides CUDA
devices so CPU baselines are honest.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request


class LlamaServer:
    def __init__(self, url: str = "http://127.0.0.1:18080"):
        self.url = url.rstrip("/")
        self.proc = None

    def start(self, exe: str, gguf: str, threads: int = 8, cpu_only: bool = True, timeout_s: int = 180):
        port = self.url.rsplit(":", 1)[1]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1") if cpu_only else None
        self.proc = subprocess.Popen(
            [
                exe,
                "-m",
                gguf,
                "-ngl",
                "0",
                "--host",
                "127.0.0.1",
                "--port",
                port,
                "-c",
                "4096",
                "-t",
                str(threads),
                "--no-webui",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.healthy():
                return self
            time.sleep(1)
        raise TimeoutError("llama-server did not become healthy")

    def stop(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait()
            self.proc = None

    def healthy(self) -> bool:
        try:
            urllib.request.urlopen(self.url + "/health", timeout=2)
            return True
        except Exception:  # noqa: BLE001
            return False

    def model_path(self) -> str | None:
        """The GGUF the running server has loaded (a server left over from another run may hold another one)."""
        with urllib.request.urlopen(self.url + "/props", timeout=5) as r:
            return json.loads(r.read()).get("model_path")

    def post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())

    def tokenize(self, text: str) -> list[int]:
        return self.post("/tokenize", {"content": text, "add_special": True})["tokens"]

    def top_next(self, ids: list[int], n: int = 5) -> list[tuple[str, float]]:
        r = self.post(
            "/completion", {"prompt": ids, "n_predict": 1, "n_probs": n, "temperature": 0, "cache_prompt": False}
        )
        p = r.get("completion_probabilities", [{}])[0]
        cand = p.get("top_logprobs") or p.get("probs") or []
        return [(c.get("token"), c.get("logprob", c.get("prob", 0.0))) for c in cand]

    def detokenize(self, ids: list[int]) -> str:
        return self.post("/detokenize", {"tokens": ids})["content"]

    def generate(self, ids: list[int], n: int) -> list[int]:
        """Greedy continuation as token ids (stops early at end of generation)."""
        r = self.post(
            "/completion",
            {"prompt": ids, "n_predict": n, "temperature": 0, "cache_prompt": False, "return_tokens": True},
        )
        return r["tokens"]

    def timings(self, ids: list[int], n_predict: int = 16) -> dict:
        return self.post(
            "/completion", {"prompt": ids, "n_predict": n_predict, "temperature": 0, "cache_prompt": False}
        )["timings"]
