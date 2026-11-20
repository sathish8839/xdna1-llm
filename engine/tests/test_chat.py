"""The chat template port. Expected strings come from rendering the GGUF's Jinja template with jinja2, as Hugging Face
does (llama.cpp's server rewrites reasoning before rendering, so it is not the reference for thinking)."""

import pytest

from xdna1_llm import chat


def U(c):
    return {"role": "user", "content": c}


B, S, E = "<|startoftext|>", "<|im_start|>", "<|im_end|>"


@pytest.mark.parametrize(
    ("messages", "kw", "expected"),
    [
        ([U("Hi")], {}, f"{B}{S}user\nHi{E}\n{S}assistant\n<think>"),
        ([U("Hi")], {"enable_thinking": False}, f"{B}{S}user\nHi{E}\n{S}assistant\n<think></think>"),
        (  # thinking of turns before the last user message is dropped
            [U("2+2?"), {"role": "assistant", "content": "4", "reasoning_content": "Simple sum."}, U("And 3+3?")],
            {},
            f"{B}{S}user\n2+2?{E}\n{S}assistant\n<think></think>4{E}\n{S}user\nAnd 3+3?{E}\n{S}assistant\n<think>",
        ),
        (
            [U("2+2?"), {"role": "assistant", "content": "<think>Sum.</think>\n4"}, U("3+3?")],
            {},
            f"{B}{S}user\n2+2?{E}\n{S}assistant\n<think></think>4{E}\n{S}user\n3+3?{E}\n{S}assistant\n<think>",
        ),
        (
            [U("2+2?"), {"role": "assistant", "content": "4", "thinking": "Sum."}],
            {"add_generation_prompt": False},
            f"{B}{S}user\n2+2?{E}\n{S}assistant\n<think>Sum.</think>4{E}\n",
        ),
        (
            [U("2+2?"), {"role": "assistant", "content": "4", "reasoning": "Sum."}, U("3+3?")],
            {"preserve_thinking": True},
            f"{B}{S}user\n2+2?{E}\n{S}assistant\n<think>Sum.</think>4{E}\n{S}user\n3+3?{E}\n{S}assistant\n<think>",
        ),
        (
            [
                U("x"),
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": "f", "arguments": {"a": True, "b": None, "c": [1, "é"], "d": 1.5}}}
                    ],
                },
                {"role": "tool", "content": [{"type": "text", "text": "r1"}, {"type": "text", "text": "r2"}]},
            ],
            {},
            f"{B}{S}user\nx{E}\n{S}assistant\n<think></think><tool_call>\n<function=f>\n"
            "<parameter=a>\ntrue\n</parameter>\n<parameter=b>\nnull\n</parameter>\n"
            '<parameter=c>\n[1, "é"]\n</parameter>\n<parameter=d>\n1.5\n</parameter>\n'
            f"</function>\n</tool_call>{E}\n{S}tool\n<tool_response>\nr1\nr2\n</tool_response>{E}\n{S}assistant\n<think>",
        ),
    ],
)
def test_render_matches_jinja(messages, kw, expected):
    assert chat.render(messages, **kw) == expected


def test_system_prompt_and_tools():
    tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
    out = chat.render([{"role": "system", "content": "Be brief."}, U("Hi")], tools=[tool])
    assert out.startswith(
        f'{B}{S}system\nBe brief.\n\n<tools>\n{{"name": "f", "parameters": {{"type": "object"}}}}\n</tools>{E}\n'
    )


def test_string_arguments_are_rejected():
    call = {"function": {"name": "f", "arguments": '{"a": 1}'}}
    with pytest.raises(ValueError, match="mapping"):
        chat.render([U("x"), {"role": "assistant", "content": "", "tool_calls": [call]}])
