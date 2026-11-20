"""LFM3's chat template, in Python: OpenAI-style messages -> the prompt text the model was trained on.

The GGUF ships the template as Jinja (`tokenizer.chat_template`). This is a port of the parts a chat server needs:
system prompt, user/assistant turns, thinking, tools and tool calls, tool results (consecutive results for the same
call are merged). Not ported: images/video, `tool_reference` / `additional_tools` (deferred tool loading).
benchmarks/check_tokenizer.py renders the same conversations through llama.cpp's Jinja engine and compares.

A rendered conversation looks like:

    <|startoftext|><|im_start|>system
    You are helpful.<|im_end|>
    <|im_start|>user
    Hi<|im_end|>
    <|im_start|>assistant
    <think>
"""

from __future__ import annotations

import json

BOS = "<|startoftext|>"


def to_json(v) -> str:
    """Jinja's `tojson` as llama.cpp renders it."""
    return json.dumps(v, ensure_ascii=False)


def render_value(v) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (dict, list)):
        return to_json(v)
    return str(v)


def parse_content(c) -> str:
    """A message's content: a string, or OpenAI content parts ({"type": "text", "text": ...})."""
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        return (c.get("text") or "") if c.get("type") == "text" else to_json(c)
    if isinstance(c, list):
        return "".join(i if isinstance(i, str) else parse_content(i) if isinstance(i, dict) else to_json(i) for i in c)
    return "" if c is None else to_json(c)


def tool_to_json(tool: dict) -> str:
    fn = tool.get("function", tool)
    return "{" + ", ".join(f"{to_json(k)}: {to_json(v)}" for k, v in fn.items() if k != "defer_loading") + "}"


def render_tool_calls(calls: list[dict]) -> str:
    out = []
    for call in calls:
        fn = call.get("function", call)
        args = fn.get("arguments")
        if isinstance(args, str) and args.strip() not in ("", "{}", "null"):
            raise ValueError("tool call arguments must be a mapping: json.loads() them before rendering")
        s = f"<tool_call>\n<function={fn['name']}>\n"
        if isinstance(args, dict):
            s += "".join(f"<parameter={k}>\n{render_value(v)}\n</parameter>\n" for k, v in args.items())
        out.append(s + "</function>\n</tool_call>")
    return "\n\n".join(out)


def render_tool_response(content) -> str:
    if isinstance(content, str) and "<tool_response>" in content:
        return content
    if isinstance(content, str):
        body = content
    elif isinstance(content, dict):
        body = parse_content(content)
    elif isinstance(content, list):
        body = "\n".join(parse_content(i) for i in content)
    else:
        body = "" if content is None else to_json(content)
    return f"<tool_response>\n{body}\n</tool_response>"


def _tool_items(content) -> list:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, dict):
        return [content]
    if isinstance(content, list):
        return list(content)
    return [] if content is None else [content]


_THINKING_KEYS = ("thinking", "reasoning", "reasoning_content")


def render(
    messages: list[dict],
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
    preserve_thinking: bool = False,
) -> str:
    """The prompt for `messages` (dicts with role/content, plus tool_calls / tool_call_id / thinking).

    Thinking from earlier turns is dropped unless preserve_thinking: only assistant turns after the last user
    message keep their <think> block. With add_generation_prompt the text ends in an open assistant turn, starting
    with "<think>" (or an empty "<think></think>" when enable_thinking is False)."""
    out, system = [BOS], ""
    if messages and messages[0]["role"] == "system":
        if messages[0].get("content"):
            system = parse_content(messages[0]["content"])
        messages = messages[1:]
    lines = [tool_to_json(t.get("function", t)) for t in tools or [] if not _deferred(t)]
    if lines:
        system += ("\n\n" if system else "") + "<tools>\n" + "\n".join(lines) + "\n</tools>"
    if system:
        out.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)

    pending = []  # tool results waiting to be merged into one <tool_response>
    for i, m in enumerate(messages):
        role, nxt = m["role"], messages[i + 1] if i + 1 < len(messages) else None
        if role == "tool":
            items = _tool_items(m.get("content"))
            if nxt and nxt["role"] == "tool" and m.get("tool_call_id") and nxt.get("tool_call_id") == m["tool_call_id"]:
                pending += items
                continue
        out.append(f"<|im_start|>{role}\n")
        if role == "assistant":
            keep = preserve_thinking or i > last_user
            raw = m.get("content")
            key = next((k for k in _THINKING_KEYS if k in m), None)
            if key and keep:
                out.append(f"<think>{m[key] or ''}</think>")
            elif keep and "</think>" not in (raw if isinstance(raw, str) else ""):
                out.append("<think></think>")
            content = parse_content(raw) if raw else ""
            if not keep:
                if "</think>" in content:
                    content = content.split("</think>")[-1].strip()
                content = "<think></think>" + content
            out.append(content)
            if m.get("tool_calls"):
                out.append(render_tool_calls(m["tool_calls"]))
        elif role == "tool":
            if pending:
                out.append(render_tool_response(pending + items))
                pending = []
            else:
                out.append(render_tool_response("" if m.get("content") is None else m["content"]))
        elif m.get("content"):
            out.append(parse_content(m["content"]))
        out.append("<|im_end|>\n")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n" + ("<think>" if enable_thinking else "<think></think>"))
    return "".join(out)


def _deferred(tool: dict) -> bool:
    return bool(tool.get("function", tool).get("defer_loading") or tool.get("defer_loading"))
