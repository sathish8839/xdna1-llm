"""The chat engine and the HTTP server, with a fake model that plays back a scripted reply (no weights needed)."""

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from xdna1_llm.engine import Engine, RequestError, parse_tool_call
from xdna1_llm.server import request_args, serve
from xdna1_llm.tokenizer import BYTE_TO_CHAR, CONTROL, NORMAL, USER_DEFINED, Tokenizer

CONTROLS = ["<|startoftext|>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
USERS = [
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
]


def tokenizer() -> Tokenizer:
    tokens = list(BYTE_TO_CHAR.values()) + CONTROLS + USERS
    types = [NORMAL] * 256 + [CONTROL] * len(CONTROLS) + [USER_DEFINED] * len(USERS)
    return Tokenizer(tokens, types, [], bos=256, eos=258, add_bos=False)


class ScriptedModel:
    """Greedy decoding of this model yields `reply` token by token, whatever the prompt."""

    def __init__(self, reply: list[int], vocab: int):
        self.reply, self.vocab, self.prompts, self.runs = reply, vocab, [], []

    def new_state(self, max_len):
        return ScriptedState()

    def _logits(self, st):
        x = np.zeros(self.vocab, np.float32)
        x[self.reply[min(st.n, len(self.reply) - 1)]] = 1.0
        st.n += 1
        return x

    def forward(self, ids, st):
        st.toks += list(ids)
        st.n = 0
        self.prompts.append(list(st.toks))  # the whole prompt, including what the state already held
        self.runs.append(len(ids))  # the tokens actually run
        return self._logits(st), []

    def step(self, t, st):
        st.toks.append(t)
        return self._logits(st)


class ScriptedState:
    """Every position can be returned to."""

    def __init__(self):
        self.toks, self.n = [], 0

    def snapshot(self):
        pass

    def rewind(self, pos):
        del self.toks[pos:]
        return len(self.toks)


def engine(reply_text: str, **kw) -> Engine:
    tok = tokenizer()
    model = ScriptedModel(tok.encode(reply_text), len(tok.tokens))
    return Engine(model, tok, "tiny", ctx=kw.pop("ctx", 512), **kw)


CALL = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>"  # noqa: E501
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}},
        },
    }
]


def run(eng, **kw):
    events = list(eng.chat([{"role": "user", "content": "Hi"}], **kw))
    text = lambda k: "".join(e.text for e in events if e.kind == k)  # noqa: E731
    return (
        text("reasoning"),
        text("content"),
        [c for e in events if e.kind == "tool_calls" for c in e.tool_calls],
        events[-1],
    )


def test_reasoning_and_content_are_split():
    reasoning, content, calls, done = run(engine("They greet me.</think>\n\nHello there!<|im_end|>"), thinking=True)
    assert (reasoning, content, calls) == ("They greet me.", "Hello there!", [])
    assert done.finish_reason == "stop" and done.usage["completion_tokens"] == len(
        tokenizer().encode("They greet me.</think>\n\nHello there!<|im_end|>")
    )


def test_thinking_off_puts_an_empty_think_block_in_the_prompt():
    eng = engine("Hello<|im_end|>")
    run(eng, thinking=False)
    assert eng.tok.decode(eng.model.prompts[-1], special=True).endswith("<|im_start|>assistant\n<think></think>")


def test_tool_calls_are_parsed_with_schema_types():
    reasoning, content, calls, done = run(engine(f"Let me check.{CALL}<|im_end|>"), tools=TOOLS)
    assert content == "Let me check." and done.finish_reason == "tool_calls"
    assert calls == [{"name": "get_weather", "arguments": json.dumps({"city": "Paris", "days": 2})}]
    call = "<function=f>\n<parameter=n>\n7\n</parameter>\n<parameter=s>\n007\n</parameter>\n</function>"
    assert json.loads(parse_tool_call(call)["arguments"]) == {"n": 7, "s": "007"}  # not JSON: kept as text
    assert json.loads(parse_tool_call(call, {("f", "n"): "string"})["arguments"])["n"] == "7"


def test_next_turn_runs_only_what_is_new():
    from xdna1_llm import chat

    eng = engine("Hello there!<|im_end|>")
    first = [{"role": "user", "content": "Hi"}]
    *_, done = eng.chat(first, thinking=False)
    assert done.usage["prompt_tokens_details"]["cached_tokens"] == 0
    second = first + [{"role": "assistant", "content": "Hello there!"}, {"role": "user", "content": "Bye"}]
    *_, done = eng.chat(second, thinking=False)
    want = eng.tok.encode(chat.render(second, enable_thinking=False))
    cached = done.usage["prompt_tokens_details"]["cached_tokens"]
    assert cached >= len(eng.tok.encode(chat.render(first, enable_thinking=False))) + len("Hello there!") - 1
    assert eng.model.prompts[-1] == want and eng.model.runs[-1] == len(want) - cached

    other = [{"role": "user", "content": "Hm"}]  # departs from the cached tokens early on
    *_, done = eng.chat(other, thinking=False)
    want = eng.tok.encode(chat.render(other, enable_thinking=False))
    assert eng.model.prompts[-1] == want and 0 < done.usage["prompt_tokens_details"]["cached_tokens"] < len(want)


def test_a_turn_stopped_early_can_still_be_resumed():
    from xdna1_llm import chat

    eng = engine("one two three<|im_end|>")
    gen = eng.chat([{"role": "user", "content": "Hi"}], thinking=False)
    next(e for e in gen if e.kind == "content")
    gen.close()  # the client went away mid-reply
    msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "o"}, {"role": "user", "content": "?"}]
    *_, done = eng.chat(msgs, thinking=False)
    assert eng.model.prompts[-1] == eng.tok.encode(chat.render(msgs, enable_thinking=False))
    assert done.usage["prompt_tokens_details"]["cached_tokens"] > 0


def test_stop_strings_and_length_limit():
    _, content, _, done = run(engine("one two three<|im_end|>"), stop=["thr"])
    assert content == "one two " and done.finish_reason == "stop"
    _, content, _, done = run(engine("one two three<|im_end|>"), max_tokens=3)
    assert content == "one" and done.finish_reason == "length"


def test_prompt_longer_than_the_context_is_rejected():
    with pytest.raises(RequestError, match="context"):
        run(engine("x", ctx=8))


def test_request_args():
    kw = request_args(
        {
            "messages": [{"role": "user", "content": "x"}],
            "max_completion_tokens": 9,
            "stop": "END",
            "chat_template_kwargs": {"enable_thinking": False},
            "tools": TOOLS,
            "tool_choice": "none",
        }  # fmt: skip
    )
    assert (kw["max_tokens"], kw["stop"], kw["thinking"], kw["tools"]) == (9, ["END"], False, None)
    assert request_args({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "high"})["thinking"]
    with pytest.raises(RequestError):
        request_args({"messages": []})


@pytest.fixture
def server():
    eng = engine(f"Checking.{CALL}<|im_end|>")
    srv = serve(eng, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", eng
    srv.shutdown()


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=10)


def test_http_models_and_completion(server):
    url, eng = server
    with urllib.request.urlopen(url + "/health") as r:
        assert json.loads(r.read()) == {"status": "ok", "model": "tiny", "device": "CPU", "ctx": 512}
    with urllib.request.urlopen(url + "/v1/models") as r:
        assert json.loads(r.read())["data"][0]["id"] == "tiny"
    msgs = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'}}
            ],
        },  # fmt: skip
        {"role": "tool", "tool_call_id": "c", "content": "5 C"},
    ]
    with post(url + "/v1/chat/completions", {"messages": msgs, "tools": TOOLS}) as r:
        out = json.loads(r.read())
    msg = out["choices"][0]["message"]
    assert msg["content"] == "Checking." and out["choices"][0]["finish_reason"] == "tool_calls"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris", "days": 2}
    prompt = eng.tok.decode(eng.model.prompts[-1], special=True)
    assert "<parameter=city>\nOslo\n</parameter>" in prompt and "<tool_response>\n5 C\n</tool_response>" in prompt


def test_http_streaming(server):
    url, _ = server
    body = {"messages": [{"role": "user", "content": "x"}], "stream": True, "stream_options": {"include_usage": True}}
    with post(url + "/v1/chat/completions", body) as r:
        assert r.headers["Content-Type"] == "text/event-stream"
        lines = [ln for ln in r.read().decode().split("\n\n") if ln]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(ln.removeprefix("data: ")) for ln in lines[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    calls = [
        c["choices"][0]["delta"]["tool_calls"]
        for c in chunks
        if c["choices"] and "tool_calls" in c["choices"][0]["delta"]
    ]
    assert text == "Checking." and calls[0][0]["function"]["name"] == "get_weather"
    assert [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]][-1] == "tool_calls"
    assert chunks[-1]["usage"]["prompt_tokens"] > 0


def test_http_errors(server):
    url, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        post(url + "/v1/chat/completions", {"messages": "nope"})
    assert e.value.code == 400 and "messages" in json.loads(e.value.read())["error"]["message"]
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(url + "/v1/nothing")
    assert e.value.code == 404
