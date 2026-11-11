"""Reversible character codec for subword tokenizers that cannot spell Akkadian.

Helsinki-NLP opus-mt (Marian) SentencePiece vocabularies were trained on modern languages:
characters such as ``š ṣ ṭ ā ī ū`` are often missing from the *source* and/or *target*
vocabulary and silently become ``<unk>``.  Then the model can neither read ``ša-ru-um`` nor
write ``Aššur``/``kārum``, which the chrF++ half of the metric punishes.

:class:`CharCodec` detects those characters for a given tokenizer and maps each to a short
ASCII placeholder (base letter + a marker per diacritic, e.g. ``š -> s~``, ``ṭ -> t^``,
``ā -> a=``).  Targets are encoded before training and decoded after generation, so the
mapping is invisible to the rest of the pipeline.  The fitted mapping is stored next to the
model as ``charmap.json`` and re-loaded at inference.  For byte-level models (byT5) nothing is
ever unknown, so the codec is the identity.
"""
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, Optional

__all__ = ["CharCodec", "CODEC_FILENAME", "placeholder_for"]

LOG = logging.getLogger(__name__)
CODEC_FILENAME = "charmap.json"

# combining mark -> ASCII marker (the markers practically never occur in the references)
_MARKERS = {
    "\u030c": "~",  # caron        š ž č
    "\u0323": "^",  # dot below    ṣ ṭ ḥ
    "\u032e": "#",  # breve below  ḫ
    "\u0331": "_",  # macron below ḏ ṯ
    "\u0304": "=",  # macron       ā ē ī ū
    "\u0302": "+",  # circumflex   â ê î û
    "\u0301": "/",  # acute        á é í ú
    "\u0300": "\\",  # grave      à è ì ù
    "\u0308": "$",  # diaeresis    ä ö ü   (never ":" -> "you:" would decode to "yoü")
    "\u0306": "%",  # breve        ă
    "\u0307": "*",  # dot above    ż
    "\u0303": "@",  # tilde        ñ
}


def placeholder_for(char: str) -> str:
    """Deterministic ASCII placeholder for one character (``š -> s~``, ``Ṣ -> S^``)."""
    decomposed = unicodedata.normalize("NFD", char)
    base, marks = decomposed[:1], decomposed[1:]
    if base.isascii() and base.isalpha() and marks and all(m in _MARKERS for m in marks):
        return base + "".join(_MARKERS[m] for m in marks)
    return f"~{ord(char):x}~"


class CharCodec:
    """Character-level mapping applied to sources/targets around a subword tokenizer."""

    def __init__(self, source_map: Optional[Dict[str, str]] = None, target_map: Optional[Dict[str, str]] = None):
        self.source_map: Dict[str, str] = dict(source_map or {})
        self.target_map: Dict[str, str] = dict(target_map or {})
        self._src_table = str.maketrans(self.source_map) if self.source_map else None
        self._tgt_table = str.maketrans(self.target_map) if self.target_map else None
        inverse = {v: k for k, v in self.target_map.items()}
        if len(inverse) != len(self.target_map):
            raise ValueError("target placeholders must be unique to be reversible")
        self._inverse = inverse
        self._inverse_re = (
            re.compile("|".join(re.escape(p) for p in sorted(inverse, key=len, reverse=True))) if inverse else None
        )

    # ------------------------------------------------------------------ mapping
    @property
    def is_identity(self) -> bool:
        return not self.source_map and not self.target_map

    def encode_source(self, text: str) -> str:
        return text.translate(self._src_table) if self._src_table else text

    def encode_target(self, text: str) -> str:
        return text.translate(self._tgt_table) if self._tgt_table else text

    def decode_target(self, text: str) -> str:
        if not self._inverse_re:
            return text
        return self._inverse_re.sub(lambda m: self._inverse[m.group(0)], text)

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(
        cls,
        tokenizer,
        sources: Iterable[str],
        targets: Iterable[str],
        min_count: int = 1,
    ) -> "CharCodec":
        """Find characters the tokenizer cannot represent and assign placeholders.

        A *source* character is mapped when encoding it yields the unknown token.  A *target*
        character is mapped when it is unknown **or** does not survive an encode/decode round
        trip (the model could never emit it verbatim).
        """
        unk_id = getattr(tokenizer, "unk_token_id", None)
        src_chars = Counter(c for s in sources for c in str(s) if not c.isspace())
        tgt_chars = Counter(c for t in targets for c in str(t) if not c.isspace())

        def _ids(char: str, target: bool):
            kwargs = {"add_special_tokens": False}
            enc = tokenizer(text_target=char, **kwargs) if target else tokenizer(char, **kwargs)
            return list(enc["input_ids"])

        def _bad_source(char: str) -> bool:
            if char.isascii():
                return False
            ids = _ids(char, target=False)
            return not ids or (unk_id is not None and unk_id in ids)

        def _bad_target(char: str) -> bool:
            if char.isascii():
                return False
            ids = _ids(char, target=True)
            if not ids or (unk_id is not None and unk_id in ids):
                return True
            decoded = tokenizer.decode(ids, skip_special_tokens=True).strip()
            return decoded != char

        source_map = {c: placeholder_for(c) for c, n in src_chars.items() if n >= min_count and _bad_source(c)}
        target_map = {c: placeholder_for(c) for c, n in tgt_chars.items() if n >= min_count and _bad_target(c)}

        # placeholders must themselves be representable; fall back to the hex form if not
        for mapping, target in ((source_map, False), (target_map, True)):
            for char, ph in list(mapping.items()):
                ids = _ids(ph, target=target)
                if unk_id is not None and unk_id in ids:
                    mapping[char] = f"~{ord(char):x}~"
        codec = cls(source_map, target_map)
        if not codec.is_identity:
            LOG.info(
                "CharCodec: %d source / %d target characters remapped: src=%s tgt=%s",
                len(source_map), len(target_map), "".join(sorted(source_map)), "".join(sorted(target_map)),
            )
        return codec

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"version": 1, "source": self.source_map, "target": self.target_map}

    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, CODEC_FILENAME)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, directory: str) -> "CharCodec":
        """Load ``charmap.json`` from a model directory (identity codec when absent)."""
        path = os.path.join(directory, CODEC_FILENAME)
        if not os.path.isfile(path):
            return cls()
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(data.get("source"), data.get("target"))

    def __repr__(self) -> str:  # pragma: no cover
        return f"CharCodec(source={self.source_map}, target={self.target_map})"
