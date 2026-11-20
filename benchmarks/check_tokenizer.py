"""Compare xdna1_llm.tokenizer and xdna1_llm.chat with llama.cpp: token ids for tricky strings and a text file,
and rendered prompts for a few conversations (llama.cpp runs the GGUF's Jinja chat template).

    python benchmarks/check_tokenizer.py --gguf LFM3-1.7B-Q4_K_M.gguf --llama-server llama-server.exe

Starts its own llama-server on --server (pick a free port) and stops it at the end. Exits 1 on any difference.
"""

import argparse
import os
import sys

from xdna1_llm import chat
from xdna1_llm.gguf_weights import GGUFWeights
from xdna1_llm.reference import LlamaServer
from xdna1_llm.tokenizer import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))

CASES = [
    "",
    "Hello world",
    "  leading and trailing spaces  ",
    "tabs\tand\nnewlines\r\n\r\nand  \n  mixed \n",
    "I'm sure they'll say it's THEY'RE, we've, you'd, I'M",
    "Numbers: 1 12 123 1234 12345 3.14159 -42 1,000,000 0x1F",
    "unicode: café naïve Straße 日本語のテキスト 한국어 العربية Ελληνικά ١٢٣ ²³ Ⅻ",
    "emoji 😀👍🏽👨‍👩‍👧 and symbols ©®™ → ≤ ≥ ∑",
    "code:\n    def f(x):\n        return x**2  # square\n\nprint(f(3))\n",
    "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>ok</think>Hello!<|im_end|>",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
    "   \n\n\n   x",
    "a b　c​d",  # no-break space, ideographic space, zero-width space
    "!!!??? ... --- ***",
]

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
CALL = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 2}}}
# (messages, tools, template kwargs)
CHATS = [
    ([{"role": "user", "content": "Hi"}], None, {}),
    ([{"role": "user", "content": "Hi"}], None, {"enable_thinking": False}),
    (
        [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": [{"type": "text", "text": "And 3+3?"}]},
        ],
        None,
        {},
    ),
    (
        [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": "", "tool_calls": [CALL]},
            {"role": "tool", "tool_call_id": "c1", "content": "18 C, sunny"},
        ],
        [WEATHER],
        {},
    ),
    (
        [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": "Checking.", "tool_calls": [CALL, CALL]},
            {"role": "tool", "tool_call_id": "c1", "content": "18 C"},
            {"role": "tool", "tool_call_id": "c1", "content": "19 C"},
            {"role": "assistant", "content": "18-19 C."},
            {"role": "user", "content": "Thanks"},
        ],
        [WEATHER],
        {"enable_thinking": False},
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--llama-server", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:18096")
    ap.add_argument("--text", default=os.path.join(HERE, "data", "eval.txt"))
    a = ap.parse_args()

    tok = Tokenizer.from_gguf(GGUFWeights(a.gguf))
    with open(a.text, encoding="utf-8") as f:
        cases = [*CASES, f.read()]
    srv = LlamaServer(a.server).start(a.llama_server, a.gguf)
    bad = 0
    try:
        for s in cases:
            ref = srv.tokenize(s)  # add_special=True: llama.cpp adds BOS only if the GGUF asks for it
            ours = tok.encode(s)
            back = tok.decode(ours, special=True)
            ok = ours == ref and back == s
            bad += not ok
            print(f"{'ok  ' if ok else 'DIFF'} {len(ref):5d} tokens  {s[:50]!r}")
            if ours != ref:
                i = next(
                    (i for i, (x, y) in enumerate(zip(ours, ref, strict=False)) if x != y), min(len(ours), len(ref))
                )
                print(f"     first difference at {i}: ours {ours[i : i + 5]} llama.cpp {ref[i : i + 5]}")
            elif back != s:
                print(f"     decode gives {back[:80]!r}")
        for msgs, tools, kw in CHATS:
            body = {"messages": msgs, "chat_template_kwargs": kw, **({"tools": tools} if tools else {})}
            ref = srv.post("/apply-template", body)["prompt"]
            ours = chat.render(msgs, tools, **kw)
            bad += ours != ref
            print(f"{'ok  ' if ours == ref else 'DIFF'} chat: {len(msgs)} messages, tools {bool(tools)}, {kw}")
            if ours != ref:
                print(f"     ours:      {ours!r}")
                print(f"     llama.cpp: {ref!r}")
    finally:
        srv.stop()
    n = len(cases) + len(CHATS)
    print(f"{n - bad}/{n} identical")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
