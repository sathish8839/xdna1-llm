"""One chat turn, end to end: messages in, a stream of events out (what xdna1_llm.server serves over HTTP).

    eng = Engine.load("LFM3-1.7B-Q4_K_M.gguf")
    for ev in eng.chat([{"role": "user", "content": "Hi"}]):
        ...  # Event("content", "Hello"), ..., Event("done", finish_reason="stop", usage={...})

The model writes its reasoning between <think> and </think> and its tool calls as
<tool_call><function=NAME><parameter=KEY>VALUE</parameter>...</function></tool_call>. These tags are single
tokens, so the output is split by token id: reasoning, content and tool calls come out as separate events, and
tool calls are returned in the OpenAI shape ({"name": ..., "arguments": "<json>"}).
"""

from __future__ import annotations

import codecs
import json
import os
import re
from dataclasses import dataclass, field

from . import chat
from .cpu_linear import CpuLinear
from .gguf_weights import GGUFWeights
from .models.lfm3 import LFM3
from .sampling import Sampler
from .tokenizer import Tokenizer
from .weight_cache import WeightCache


@dataclass
class Event:
    kind: str  # "reasoning", "content", "tool_calls" or "done"
    text: str = ""
    tool_calls: list = field(default_factory=list)
    finish_reason: str | None = None  # with "done": "stop", "length" or "tool_calls"
    usage: dict | None = None  # with "done": prompt_tokens, completion_tokens, total_tokens


SNAPSHOT_EVERY = 32  # decoded tokens between conv snapshots: a later turn re-runs at most this many reply tokens


class RequestError(ValueError):
    """The request cannot be served as given (HTTP 400)."""


def gemm_backend(npu, kdir, cache=None):
    """The bf16 GEMM for batched prompts, sharing the device; None (with a note) if its kernels are not built."""
    from .npu.linear import SUPPORTED_K, NpuLinear

    if not all(os.path.exists(os.path.join(kdir, f"gemm_256x{k}x1024_bf16_f32.xclbin")) for k in SUPPORTED_K):
        print(f"note: no bf16 GEMM kernels in {kdir} (kernels/build_kernels.sh <out> engine); prompt runs on the GEMV")
        return None
    return NpuLinear(npu, kdir, cache=cache)


class Engine:
    def __init__(
        self, model, tok: Tokenizer, name: str, ctx: int = 4096, w=None, thinking: bool = False, device: str = "CPU"
    ):
        self.model, self.tok, self.name, self.ctx, self.w, self.device = model, tok, name, ctx, w, device
        self.thinking = thinking  # default when a request does not say
        ids = tok.ids
        self.think_start, self.think_end = ids.get("<think>"), ids.get("</think>")
        self.call_start, self.call_end = ids.get("<tool_call>"), ids.get("</tool_call>")
        self._cache = None  # (tokens, state) of the last turn, to reuse its shared prefix

    @classmethod
    def load(
        cls,
        gguf: str,
        backend: str = "npu",
        kernels: str | None = None,
        gemm_prompt: bool = True,
        weight_cache: bool = True,
        **kw,
    ):
        """weight_cache: keep the NPU weight layouts on disk (xdna1_llm.weight_cache), so the next load is faster."""
        w = GGUFWeights(gguf)
        tok = Tokenizer.from_gguf(w)
        if backend == "npu":
            from .npu.qlinear import NpuQLinear  # imports pyxrt

            cache = WeightCache.for_model(gguf) if weight_cache else None
            lin = NpuQLinear(kernel_dir=kernels, formats=w.types(), cache=cache)
            model = LFM3(w, lin, gemm_backend(lin.npu, lin.kdir, cache) if gemm_prompt else None)
            kw.setdefault("device", lin.npu.name)
        else:
            model = LFM3(w, CpuLinear())
        return cls(model, tok, os.path.splitext(os.path.basename(gguf))[0], w=w, **kw)

    def sampler(self, **opts) -> Sampler:
        """The GGUF's recommended sampling settings (if any), with the request's values on top."""
        if self.w is not None:
            return Sampler.from_gguf(self.w, **opts)
        return Sampler(**{k: v for k, v in opts.items() if v is not None})

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        thinking: bool | None = None,
        stop: list[str] | None = None,
        **sampling,
    ):
        """Generate one assistant turn; yields Events, ending with a "done" event. Closing the generator early
        (client gone) stops generation."""
        thinking = self.thinking if thinking is None else thinking
        prompt = chat.render(_normalize(messages), tools, enable_thinking=thinking)
        ids = self.tok.encode(prompt)
        room = self.ctx - len(ids)
        if room < 1:
            raise RequestError(f"the prompt has {len(ids)} tokens; the context holds {self.ctx}")
        limit = min(max_tokens or room, room)
        pick = self.sampler(**sampling)
        out = _Output(self, thinking, stop or [], _param_types(tools))

        m = self.model
        st, cached = self._resume(ids)
        logits, _ = m.forward(ids[cached:], st)
        fed = list(ids)  # the tokens in st, kept in step so the next turn can resume from them
        self._cache = (fed, st)
        history, n, finish = list(ids), 0, "length"
        while n < limit:
            t = pick(logits, history)
            history.append(t)
            n += 1
            if t in self.tok.eog:
                finish = "stop"
                break
            yield from out.feed(t)
            if out.stopped:
                finish = "stop"
                break
            self._cache = None  # a step that fails halfway leaves the state unusable
            logits = m.step(t, st)
            fed.append(t)
            self._cache = (fed, st)
            if len(fed) % SNAPSHOT_EVERY == 0:
                st.snapshot()
        yield from out.flush()
        if out.calls:
            finish = "tool_calls"
            yield Event("tool_calls", tool_calls=out.calls)
        usage = {
            "prompt_tokens": len(ids),
            "completion_tokens": n,
            "total_tokens": len(ids) + n,
            "prompt_tokens_details": {"cached_tokens": cached},
        }
        yield Event("done", finish_reason=finish, usage=usage)

    def _resume(self, ids: list[int]):
        """A state to continue `ids` from: the last turn's, rewound to where `ids` departs from it (or to the
        nearest snapshot before that), else a fresh one. Returns (state, number of ids already in it)."""
        cache, self._cache = self._cache, None
        if cache is None:
            return self.model.new_state(self.ctx), 0
        toks, st = cache
        same = 0  # leaves at least one token to run: its logits pick the first reply token
        for a, b in zip(toks, ids[:-1], strict=False):
            if a != b:
                break
            same += 1
        return st, st.rewind(same)


class _Output:
    """Splits generated tokens into reasoning, content and tool calls; applies stop strings to the content."""

    def __init__(self, eng: Engine, thinking: bool, stop: list[str], types: dict):
        self.eng, self.stop, self.types = eng, [s for s in stop if s], types
        self.mode = "reasoning" if thinking else "content"
        self.fresh = True  # at the start of a segment: drop leading whitespace
        self.utf8 = codecs.getincrementaldecoder("utf-8")("replace")
        self.pending = ""  # content held back while it could be the start of a stop string
        self.call = None  # text of the tool call being read
        self.calls, self.stopped = [], False

    def feed(self, t: int):
        e = self.eng
        if t == e.think_start:
            yield from self._flush_content()
            self.mode, self.fresh = "reasoning", True
            return
        if t == e.think_end:
            self.mode, self.fresh = "content", True
            return
        if t == e.call_start:
            yield from self._flush_content()
            self.call = ""
            return
        text = self.utf8.decode(e.tok.token_bytes(t))
        if self.call is not None:
            if t == e.call_end:
                self.calls.append(parse_tool_call(self.call, self.types))
                self.call, self.fresh = None, True
            else:
                self.call += text
            return
        if self.fresh:
            text = text.lstrip()
            if not text:
                return
            self.fresh = False
        if self.mode == "reasoning":
            yield Event("reasoning", text)
            return
        self.pending += text
        for s in self.stop:
            if s in self.pending:
                self.pending = self.pending[: self.pending.index(s)]
                self.stopped = True
                yield from self._flush_content()
                return
        keep = max((len(s) - 1 for s in self.stop), default=0)
        cut = len(self.pending) - keep
        if cut > 0:
            yield Event("content", self.pending[:cut])
            self.pending = self.pending[cut:]

    def _flush_content(self):
        if self.pending:
            yield Event("content", self.pending)
            self.pending = ""

    def flush(self):
        if self.call is not None and self.call.strip():  # generation ended inside a tool call: keep what came
            self.calls.append(parse_tool_call(self.call, self.types))
            self.call = None
        if self.calls:
            self.pending = self.pending.rstrip()
        yield from self._flush_content()


def parse_tool_call(text: str, types: dict | None = None) -> dict:
    """The text between <tool_call> and </tool_call> -> {"name", "arguments": JSON string}. Values are JSON where
    they parse as JSON, except for parameters the tool declares as strings."""
    m = re.search(r"<function=([^>\n]+)>", text)
    name = m.group(1).strip() if m else ""
    args = {}
    for key, raw in re.findall(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", text, re.S):
        key = key.strip()
        if (types or {}).get((name, key)) == "string":
            args[key] = raw
            continue
        try:
            args[key] = json.loads(raw)
        except ValueError:
            args[key] = raw
    return {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}


def _param_types(tools) -> dict:
    """(tool name, parameter) -> JSON schema type, from the request's tools."""
    out = {}
    for t in tools or []:
        fn = t.get("function", t)
        for key, schema in ((fn.get("parameters") or {}).get("properties") or {}).items():
            if isinstance(schema, dict) and isinstance(schema.get("type"), str):
                out[fn.get("name"), key] = schema["type"]
    return out


def _normalize(messages: list[dict]) -> list[dict]:
    """OpenAI messages -> what the chat template expects: tool call arguments as mappings, not JSON strings."""
    out = []
    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg:
            raise RequestError("every message needs a role")
        if msg.get("tool_calls"):
            msg = dict(msg, tool_calls=[_call_args(c) for c in msg["tool_calls"]])
        out.append(msg)
    return out


def _call_args(call: dict) -> dict:
    fn = call.get("function", call)
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError as e:
            raise RequestError(f"tool call arguments are not valid JSON: {e}") from None
    return dict(call, function=dict(fn, arguments=args))
