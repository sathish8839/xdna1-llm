"""The native tokenizer on a tiny hand-made vocabulary (benchmarks/check_tokenizer.py checks the real one against
llama.cpp)."""

import pytest

from xdna1_llm.tokenizer import BYTE_TO_CHAR, CONTROL, NORMAL, USER_DEFINED, Tokenizer, pretokenize


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("Hello world", ["Hello", " world"]),
        ("it's THEY'RE", ["it", "'s", " THEY", "'RE"]),
        ("12345 3.14", ["123", "45", " ", "3", ".", "14"]),
        ("a  b", ["a", " ", " b"]),
        ("x\n\n  y", ["x", "\n\n", " ", " y"]),
        ("end  ", ["end", "  "]),
        ("f(x):\n", ["f", "(x", "):\n"]),
        ("café 日本", ["café", " 日本"]),
        ("a !!!\n", ["a", " !!!\n"]),
    ],
)
def test_pretokenize(text, words):
    assert pretokenize(text) == words


def test_byte_map_covers_every_byte_once():
    assert sorted(BYTE_TO_CHAR) == list(range(256))
    assert len(set(BYTE_TO_CHAR.values())) == 256
    assert BYTE_TO_CHAR[ord(" ")] == "Ġ" and BYTE_TO_CHAR[ord("\n")] == "Ċ" and BYTE_TO_CHAR[ord("a")] == "a"


def tiny() -> Tokenizer:
    tokens = list(BYTE_TO_CHAR.values()) + ["lo", "low", "Ġlow", "er", "Ġlower", "<s>", "<|im_end|>", "<think>"]
    types = [NORMAL] * 261 + [CONTROL, CONTROL, USER_DEFINED]
    merges = ["l o", "lo w", "Ġ low", "e r"]  # no merge makes "Ġlower": it is only reachable as a whole word
    return Tokenizer(tokens, types, merges, bos=261, eos=262, add_bos=False)


def test_merges_and_whole_words():
    tok = tiny()
    ids = {t: i for i, t in enumerate(tok.tokens)}
    assert tok.encode("low") == [ids["low"]]
    assert tok.encode(" lower") == [ids["Ġlower"]]  # whole word in the vocabulary: no merging needed
    assert tok.encode("lowest") == [ids["low"], ids["e"], ids["s"], ids["t"]]
    assert tok.encode("lower") == [ids["low"], ids["er"]]


def test_special_tokens_and_bos():
    tok = tiny()
    assert tok.encode("<|im_end|>low", add_bos=True) == [261, 262, 257]
    assert tok.encode("<|im_end|>", parse_special=False) == [tok.ids[c] for c in "<|im_end|>"]
    assert tok.encode("<think>", parse_special=False) == [263]  # user-defined tokens are always matched
    assert tok.eog == {262}


def test_decode_round_trip():
    tok = tiny()
    for s in ["low lower", "naïve\n\t😀", "<think>hm<|im_end|>"]:
        assert tok.decode(tok.encode(s), special=True) == s
    assert tok.decode(tok.encode("a<|im_end|>")) == "a"  # control tokens are hidden by default
