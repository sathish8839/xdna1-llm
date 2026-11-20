"""The model's own tokenizer, read from the GGUF: text <-> token ids without llama.cpp.

LFM3 uses GPT-2 style byte-level BPE (`tokenizer.ggml.model = gpt2`, pre-tokenizer `lfm2`). Encoding has three steps:
  1. split out special tokens (`<|im_start|>`, `<think>`, ...), which map straight to their ids;
  2. pre-tokenize the rest into words with the Llama 3 rules (`pretokenize`): letters, up to 3 digits, punctuation
     runs and whitespace each become separate words, so a merge never crosses a word boundary;
  3. turn each word's UTF-8 bytes into printable characters (byte 0x20 -> 'Ġ', ...) and merge neighbouring pieces in
     the order of the GGUF's merge list until no listed pair is left. A word that is already a whole vocabulary entry
     is taken as is (llama.cpp's `ignore_merges` for this pre-tokenizer).
Decoding maps each token's characters back to bytes. llama.cpp (src/llama-vocab.cpp, src/unicode.cpp) is the
reference; benchmarks/check_tokenizer.py compares the two token for token.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

NORMAL, CONTROL, USER_DEFINED = 1, 3, 4  # tokenizer.ggml.token_type values

# Unicode White_Space (what \s means in the pre-tokenizer regex)
_WS = frozenset(
    map(chr, [*range(0x09, 0x0E), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F])
) | {chr(0x3000)}


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte -> character table: printable bytes map to themselves, the rest to code points from 256 up."""
    bs = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs, strict=True)}


BYTE_TO_CHAR = bytes_to_unicode()
CHAR_TO_BYTE = {c: b for b, c in BYTE_TO_CHAR.items()}


def _kind(c: str) -> str:
    if c in _WS:
        return "s"
    cat = unicodedata.category(c)[0]
    return "l" if cat == "L" else "n" if cat == "N" else "o"


def pretokenize(text: str) -> list[str]:
    """Split text into words with the Llama 3 pre-tokenizer regex, written out as a scanner:

        (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\\r\\n\\p{L}\\p{N}]?\\p{L}+ | \\p{N}{1,3} | ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*
        | \\s*[\\r\\n]+ | \\s+(?!\\S) | \\s+

    The first alternative that matches at the current position wins (same structure as llama.cpp's
    unicode_regex_split_custom_llama3)."""
    n = len(text)
    kinds = [_kind(c) for c in text] + [""]
    out, pos = [], 0
    while pos < n:
        c, k, start = text[pos], kinds[pos], pos
        if c == "'" and pos + 1 < n:  # contractions
            c1 = text[pos + 1].lower()
            if c1 in ("s", "t", "m", "d"):
                pos += 2
            elif pos + 2 < n and c1 + text[pos + 2].lower() in ("re", "ve", "ll"):
                pos += 3
        if pos > start:
            pass
        elif c not in "\r\n" and k != "n" and (k == "l" or kinds[pos + 1] == "l"):  # optional prefix + letters
            pos += 1
            while kinds[pos] == "l":
                pos += 1
        elif k == "n":  # 1 to 3 digits
            pos += 1
            while kinds[pos] == "n" and pos - start < 3:
                pos += 1
        elif (kinds[pos + 1] if c == " " else k) == "o":  # optional space + punctuation, then newlines
            pos += 1 + (c == " ")
            while kinds[pos] == "o":
                pos += 1
            while pos < n and text[pos] in "\r\n":
                pos += 1
        else:  # whitespace
            m = 0
            while kinds[pos + m] == "s":
                m += 1
            last_nl = max((j + 1 for j in range(m) if text[pos + j] in "\r\n"), default=0)
            if last_nl:  # \s*[\r\n]+
                pos += last_nl
            elif m > 1 and pos + m < n:  # \s+(?!\S): leave the last space for the next word
                pos += m - 1
            else:
                pos += m
        out.append(text[start:pos])
    return out


class Tokenizer:
    def __init__(self, tokens: list[str], types: list[int], merges: list[str], bos: int, eos: int, add_bos: bool):
        self.tokens, self.types = tokens, types
        self.ids = {t: i for i, t in enumerate(tokens)}
        self.ranks = {tuple(m.split(" ", 1)): r for r, m in enumerate(merges)}
        self.bos, self.eos, self.add_bos = bos, eos, add_bos
        # end-of-generation tokens: <|im_end|> ends a chat turn, <|endoftext|> a document
        self.eog = {eos} | {self.ids[t] for t in ("<|endoftext|>", "<|im_end|>") if t in self.ids}
        specials = sorted(
            (i for i, ty in enumerate(types) if ty in (CONTROL, USER_DEFINED)), key=lambda i: -len(tokens[i])
        )
        self._special_re = {
            parse: re.compile("|".join(re.escape(tokens[i]) for i in specials if parse or types[i] == USER_DEFINED))
            for parse in (True, False)
        }
        self._word = lru_cache(maxsize=1 << 16)(self._encode_word)

    @classmethod
    def from_gguf(cls, w) -> Tokenizer:
        """`w` is a GGUFWeights (or anything with the same metadata accessors)."""
        model = w.string("tokenizer.ggml.model")
        if model != "gpt2":
            raise ValueError(f"only byte-level BPE (gpt2) tokenizers are supported, this GGUF has {model!r}")
        f = w.fields["tokenizer.ggml.merges"]
        merges = [bytes(f.parts[i]).decode("utf-8") for i in f.data]
        add_bos = bool(w.scalar("tokenizer.ggml.add_bos_token")) if "tokenizer.ggml.add_bos_token" in w.fields else True
        return cls(
            w.tokens(),
            [int(t) for t in w.array("tokenizer.ggml.token_type")],
            merges,
            int(w.scalar("tokenizer.ggml.bos_token_id")),
            int(w.scalar("tokenizer.ggml.eos_token_id")),
            add_bos,
        )

    # ---- text -> ids
    def encode(self, text: str, add_bos: bool | None = None, parse_special: bool = True) -> list[int]:
        """Token ids for `text`. Special-token text such as `<|im_start|>` becomes its single id (with
        parse_special=False only user-defined ones like `<think>` do, as in llama.cpp). add_bos defaults to the
        GGUF's tokenizer.ggml.add_bos_token (False for LFM3, whose chat template writes the BOS itself)."""
        ids = [self.bos] if (self.add_bos if add_bos is None else add_bos) else []
        pos = 0
        for m in self._special_re[parse_special].finditer(text):
            ids += self._encode_plain(text[pos : m.start()])
            ids.append(self.ids[m.group()])
            pos = m.end()
        ids += self._encode_plain(text[pos:])
        return ids

    def _encode_plain(self, text: str) -> list[int]:
        return [i for word in pretokenize(text) for i in self._word(word)] if text else []

    def _encode_word(self, word: str) -> tuple[int, ...]:
        word = "".join(BYTE_TO_CHAR[b] for b in word.encode("utf-8"))
        if word in self.ids:  # ignore_merges: a whole word in the vocabulary is one token
            return (self.ids[word],)
        parts = list(word)
        while len(parts) > 1:
            ranked = [(self.ranks.get(p, 1 << 62), i) for i, p in enumerate(zip(parts, parts[1:], strict=False))]
            rank, at = min(ranked)
            if rank == 1 << 62:
                break
            pair, merged, i = (parts[at], parts[at + 1]), [], 0
            while i < len(parts):  # merge every occurrence of the best pair, left to right
                if i + 1 < len(parts) and (parts[i], parts[i + 1]) == pair:
                    merged.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        out = []
        for p in parts:
            if p in self.ids:
                out.append(self.ids[p])
            else:  # not in the vocabulary (should not happen for a byte-level vocab): fall back to single bytes
                out += [self.ids[c] for c in p]
        return tuple(out)

    # ---- ids -> text
    def token_bytes(self, i: int, special: bool = False) -> bytes:
        """The bytes token `i` stands for. Control tokens (<|im_end|>, ...) are empty unless special=True."""
        t, ty = self.tokens[i], self.types[i]
        if ty == NORMAL:
            return bytes(CHAR_TO_BYTE[c] for c in t)
        if ty == USER_DEFINED or (special and ty == CONTROL):
            return t.encode("utf-8")
        return b""

    def decode(self, ids, special: bool = False) -> str:
        return b"".join(self.token_bytes(int(i), special) for i in ids).decode("utf-8", "replace")
