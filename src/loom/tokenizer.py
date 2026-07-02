"""Character-level tokenizer.

loom is deliberately character-level: the vocabulary is every distinct
character in the training corpus, so tokenization adds zero moving parts and
every model behavior stays inspectable down to the byte. For real subword
tokenization (byte-pair encoding), see the companion project mosaic:
https://github.com/Hritikd/mosaic
"""

from __future__ import annotations

import json


class CharTokenizer:
    """Maps characters to integer ids and back.

    The vocabulary is fixed at construction time (sorted unique characters),
    so encode/decode are deterministic and ids are stable across runs.
    """

    def __init__(self, chars: list[str]):
        if len(set(chars)) != len(chars):
            raise ValueError("vocabulary contains duplicate characters")
        self.chars = sorted(chars)
        self._stoi = {ch: i for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str, errors: str = "strict") -> list[int]:
        """Encode text to ids.

        errors="strict" raises on characters outside the vocabulary;
        errors="replace" substitutes the first whitespace-like character in
        the vocabulary (or id 0), which keeps interactive demos forgiving.
        """
        if errors not in ("strict", "replace"):
            raise ValueError(f"unknown errors mode: {errors!r}")
        out = []
        fallback = self._stoi.get(" ", 0)
        for ch in text:
            i = self._stoi.get(ch)
            if i is None:
                if errors == "strict":
                    raise ValueError(f"character {ch!r} not in vocabulary")
                i = fallback
            out.append(i)
        return out

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)

    def to_json(self) -> str:
        return json.dumps({"chars": self.chars})

    @classmethod
    def from_json(cls, s: str) -> CharTokenizer:
        return cls(json.loads(s)["chars"])
