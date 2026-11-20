"""OpenAI-compatible HTTP server: point Open WebUI, an IDE plugin or the OpenAI SDKs at http://127.0.0.1:8080/v1.

    xdna1-llm-server --gguf LFM3-1.7B-Q4_K_M.gguf [--port 8080] [--thinking]

Endpoints:
  GET  /health                 {"status": "ok", "model", "device", "ctx"} once the model is loaded
  GET  /v1/models              the one loaded model
  POST /v1/chat/completions    messages, tools, stream (SSE), max_tokens / max_completion_tokens, temperature,
                               top_p, seed, stop; extensions: top_k, min_p, repeat_penalty, and thinking on or off
                               through chat_template_kwargs.enable_thinking or reasoning_effort ("none" = off)

Reasoning comes back as `reasoning_content` next to `content` (the llama.cpp / DeepSeek convention). The NPU runs
one turn at a time, so concurrent requests wait their turn. Binds to localhost unless --host says otherwise.
Standard library only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .engine import Engine, Event, RequestError


def request_args(body: dict) -> dict:
    """A /v1/chat/completions body -> keyword arguments for Engine.chat."""
    if not isinstance(body, dict):
        raise RequestError("the body must be a JSON object")
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise RequestError("messages must be a non-empty list")
    tools = body.get("tools") or None
    if body.get("tool_choice") == "none":
        tools = None
    stop = body.get("stop")
    stop = [stop] if isinstance(stop, str) else list(stop or [])
    kw = dict(
        messages=msgs,
        tools=tools,
        max_tokens=body.get("max_completion_tokens") or body.get("max_tokens"),
        stop=stop,
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        top_k=body.get("top_k"),
        min_p=body.get("min_p"),
        repetition_penalty=body.get("repeat_penalty"),
        seed=body.get("seed"),
    )
    tk = body.get("chat_template_kwargs") or {}
    if "enable_thinking" in tk:
        kw["thinking"] = bool(tk["enable_thinking"])
    elif "reasoning_effort" in body:
        kw["thinking"] = body["reasoning_effort"] not in (None, "none", "minimal")
    for k in ("max_tokens", "top_k", "seed"):
        if kw[k] is not None and not isinstance(kw[k], int):
            raise RequestError(f"{k} must be an integer")
    return kw


def completion(name: str, events: list[Event]) -> dict:
    """The non-streaming response for a finished turn."""
    msg = {"role": "assistant", "content": "".join(e.text for e in events if e.kind == "content") or None}
    reasoning = "".join(e.text for e in events if e.kind == "reasoning")
    if reasoning:
        msg["reasoning_content"] = reasoning
    calls = [c for e in events if e.kind == "tool_calls" for c in e.tool_calls]
    if calls:
        msg["tool_calls"] = [_call(c) for c in calls]
    done = events[-1]
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": name,
        "choices": [{"index": 0, "message": msg, "finish_reason": done.finish_reason}],
        "usage": done.usage,
    }


def chunks(name: str, events, include_usage: bool = False):
    """The SSE chunks for a turn, as dicts (the caller writes `data: <json>` lines and the final [DONE])."""
    base = {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion.chunk", "created": int(time.time())}
    base["model"] = name

    def chunk(delta, finish=None):
        return dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": finish}])

    yield chunk({"role": "assistant", "content": ""})
    for e in events:
        if e.kind == "content":
            yield chunk({"content": e.text})
        elif e.kind == "reasoning":
            yield chunk({"reasoning_content": e.text})
        elif e.kind == "tool_calls":
            yield chunk({"tool_calls": [dict(_call(c), index=i) for i, c in enumerate(e.tool_calls)]})
        elif e.kind == "done":
            yield chunk({}, e.finish_reason)
            if include_usage:
                yield dict(base, choices=[], usage=e.usage)


def _call(c: dict) -> dict:
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function", "function": dict(c)}


class Handler(BaseHTTPRequestHandler):
    engine: Engine
    lock = threading.Lock()  # one turn at a time on the NPU
    server_version = "xdna1-llm"

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/health":
            e = self.engine
            self._json(200, {"status": "ok", "model": e.name, "device": e.device, "ctx": e.ctx})
        elif path in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [self._model()]})
        elif path in (f"/v1/models/{self.engine.name}", f"/models/{self.engine.name}"):
            self._json(200, self._model())
        else:
            self._error(404, f"no route {self.path}", "not_found_error")

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if path not in ("/v1/chat/completions", "/chat/completions"):
            return self._error(404, f"no route {self.path}", "not_found_error")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"null")
            kw = request_args(body)
        except (ValueError, RequestError) as e:
            return self._error(400, str(e))
        with self.lock:
            try:
                events = self.engine.chat(**kw)
                if body.get("stream"):
                    self._stream(events, bool((body.get("stream_options") or {}).get("include_usage")))
                else:
                    self._json(200, completion(self.engine.name, list(events)))
            except RequestError as e:
                self._error(400, str(e))
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # the client left; closing the generator stopped the turn
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._error(500, f"{type(e).__name__}: {e}", "server_error")

    def _stream(self, events, include_usage):
        first = next(events)  # runs the prompt: a RequestError surfaces here, before any header is sent
        gen = chunks(self.engine.name, itertools.chain([first], events), include_usage)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for c in gen:
                self._sse(c)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        finally:
            gen.close()
            events.close()

    def _sse(self, obj):
        self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
        self.wfile.flush()

    def _model(self):
        return {"id": self.engine.name, "object": "model", "created": 0, "owned_by": "xdna1-llm"}

    def _json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code, message, kind="invalid_request_error"):
        self._json(code, {"error": {"message": message, "type": kind, "code": code}})

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}", flush=True)


def serve(engine: Engine, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"engine": engine, "lock": threading.Lock()})
    return ThreadingHTTPServer((host, port), handler)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="xdna1-llm-server")
    ap.add_argument("--gguf", default=os.environ.get("XDNA1_LLM_GGUF"), help="LFM3 Q4_K_M or Q6_K GGUF")
    ap.add_argument("--host", default="127.0.0.1", help="address to bind (default: localhost only)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--backend", choices=("npu", "cpu"), default="npu")
    ap.add_argument("--kernels", help="directory with built xclbins")
    ap.add_argument("--ctx", type=int, default=4096, help="context length in tokens (prompt + reply)")
    ap.add_argument("--thinking", action="store_true", help="think before answering unless a request says not to")
    ap.add_argument("--no-gemm-prompt", action="store_true", help="no bf16 weight copy; prompts run on the GEMV")
    a = ap.parse_args(argv)
    if not a.gguf:
        ap.error("--gguf (or XDNA1_LLM_GGUF) is required")

    t0 = time.perf_counter()
    eng = Engine.load(a.gguf, a.backend, a.kernels, gemm_prompt=not a.no_gemm_prompt, ctx=a.ctx, thinking=a.thinking)
    srv = serve(eng, a.host, a.port)
    print(f"{eng.name} ready in {time.perf_counter() - t0:.0f} s; serving http://{a.host}:{a.port}/v1", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if a.backend == "npu":
            from .npu.device import hard_exit

            hard_exit(0)


if __name__ == "__main__":
    main()
