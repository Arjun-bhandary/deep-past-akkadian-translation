"""Text normalisation for Old Assyrian transliterations and their English translations.

Implements the formatting suggestions published with the Deep Past Challenge:

* **transliteration (source)** — :func:`normalize_transliteration`: remove modern scribal
  notation (``! ? / : . ˹ ˺ [ ] < >``), turn breaks into the two gap markers ``<gap>``
  (single sign) / ``<big_gap>`` (more than one sign), map ``Ḫ ḫ -> H h`` (the test data only
  has ``H h``), subscript digits to plain digits and superscripts to ``{determinatives}``.
* **translation (target)** — :func:`normalize_translation` / :func:`normalize_gaps`: light
  clean-up and standardisation of the many spellings of missing text (``...``, ``…``,
  ``[...]``, ``x``, ``(broken)``, ``<gap>`` ...) that different models / editions produce.
  This is the "normalize gap tokens" step of the MBR ensemble.
* :func:`restore_glued_determinatives` — OCR-only heuristic used by the AKT pipeline when
  superscript formatting was lost (``dA-šùr -> {d}A-šùr``, ``a-limki -> a-lim{ki}``).

All public functions accept ``None``/NaN, return ``str`` and are idempotent.
"""
from __future__ import annotations

import html
import math
import re
import unicodedata
from typing import Any, List

__all__ = [
    "GAP",
    "BIG_GAP",
    "is_missing",
    "to_text",
    "basic_clean",
    "normalize_transliteration",
    "normalize_translation",
    "normalize_gaps",
    "collapse_gaps",
    "collapse_repeats",
    "postprocess_translation",
    "restore_glued_determinatives",
    "DETERMINATIVES",
]

GAP = "<gap>"
BIG_GAP = "<big_gap>"

# Private-use placeholders protect gap markers while brackets are being stripped.
_GAP_PH = "\ue000"
_BIG_PH = "\ue001"

# Determinatives listed in the competition's dataset instructions (canonical lower case).
DETERMINATIVES = (
    "d", "mul", "ki", "lu2", "e2", "uru", "kur", "mi", "m", "f", "geš", "giš", "tug2",
    "dub", "id2", "mušen", "na4", "kuš", "u2", "munus", "urudu", "dug", "gi", "lú", "é",
)

# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₓ", "0123456789x")
_SUPER_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUPER_DIGIT_MAP = str.maketrans(_SUPER_DIGITS, "0123456789")
_SUPER_LETTERS = {
    "ᵃ": "a", "ᵇ": "b", "ᶜ": "c", "ᵈ": "d", "ᵉ": "e", "ᶠ": "f", "ᵍ": "g", "ʰ": "h",
    "ⁱ": "i", "ʲ": "j", "ᵏ": "k", "ˡ": "l", "ᵐ": "m", "ⁿ": "n", "ᵒ": "o", "ᵖ": "p",
    "ʳ": "r", "ˢ": "s", "ᵗ": "t", "ᵘ": "u", "ᵛ": "v", "ʷ": "w", "ˣ": "x", "ʸ": "y",
    "ᶻ": "z", "ᴬ": "a", "ᴮ": "b", "ᴰ": "d", "ᴱ": "e", "ᴳ": "g", "ᴴ": "h", "ᴵ": "i",
    "ᴶ": "j", "ᴷ": "k", "ᴸ": "l", "ᴹ": "m", "ᴺ": "n", "ᴼ": "o", "ᴾ": "p", "ᴿ": "r",
    "ᵀ": "t", "ᵁ": "u", "ⱽ": "v", "ᵂ": "w",
}
_SUPER_LETTER_CLASS = "".join(_SUPER_LETTERS)
_SUPER_RUN_RE = re.compile(f"[{_SUPER_LETTER_CLASS}]+[{_SUPER_DIGITS}₀-₉]*")
_SUPER_DIGIT_RE = re.compile(f"[{_SUPER_DIGITS}]+")

_SPACES_RE = re.compile("[\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\t\r\n\f\v]")
_ZERO_WIDTH_RE = re.compile("[\u200b-\u200d\u2060\ufeff\u00ad]")
_LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"})
_DASHES = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2212"})
_HALF_BRACKETS = str.maketrans({c: None for c in "˹˺⸢⸣⌈⌉⌊⌋「」"})
_WS_RE = re.compile(r"\s+")

# gap markers that may already be present in any spelling: <gap>, < gap >, <GAP>, <big gap>,
# <big-gap>, <biggap>, &lt;gap&gt; (handled by html.unescape before this runs)
_EXISTING_BIG_RE = re.compile(r"<\s*big[\s_\-]*gap\s*>", re.I)
_EXISTING_GAP_RE = re.compile(r"<\s*gap\s*>", re.I)

_ELLIPSIS = r"(?:\.\s?\.\s?\.+|…)"
_BRACKETED_ELLIPSIS_RE = re.compile(rf"[\[(]\s*{_ELLIPSIS}(?:[\s\-]*{_ELLIPSIS})*\s*[\])]")
_ELLIPSIS_RUN_RE = re.compile(rf"{_ELLIPSIS}(?:[\s\-]*{_ELLIPSIS})*")
_BRACKETED_XS_RE = re.compile(r"\[\s*x(?:[\s.\-]*x)+\s*\]", re.I)
_BRACKETED_X_RE = re.compile(r"\[\s*x\s*\]", re.I)
_BROKEN_COMMENT_RE = re.compile(
    r"[\[(]\s*(?:(?:the\s+)?rest(?:\s+of\s+[\w\s.]+?)?|remainder(?:\s+of\s+[\w\s.]+?)?|"
    r"(?:about\s+|ca\.?\s+|c\.\s+)?(?:\d+|one|two|three|four|five|six|several|many|a\s+few|few)?\s*"
    r"(?:lines?|signs?|words?)|(?:large\s+|small\s+)?(?:break|lacuna|gap))?\s*"
    r"(?:is\s+|are\s+)?(?:broken(?:\s+off|\s+away)?|missing|destroyed|lost|illegible|effaced|damaged)\s*[\])]",
    re.I,
)
_BREAK_WORD_RE = re.compile(r"[\[(]\s*(?:break|lacuna|gap|large\s+break|small\s+break)\s*[\])]", re.I)
_ERASURE_COMMENT_RE = re.compile(
    r"\(\s*(?:erasure|erased|blank|space|uninscribed|vacat|blank\s+space|seal(?:\s+impression)?|traces?)"
    r"(?:\s+[^()]*)?\)",
    re.I,
)
_DOUBLE_ANGLE_RE = re.compile(r"<<[^<>]*>>|«[^«»]*»")
_SINGLE_ANGLE_RE = re.compile(r"<([^<>]*)>")
_XS_RE = re.compile(r"(?<![\w{\u0300-\u036f])x(?:[\s\-]+x)+(?![\w}\u0300-\u036f])")
_X_RE = re.compile(r"(?<![\w{\u0300-\u036f])x(?![\w}\u0300-\u036f])")
_XS_RE_I = re.compile(_XS_RE.pattern, re.I)
_X_RE_I = re.compile(_X_RE.pattern, re.I)
_GAP_RUN_RE = re.compile(f"[{_GAP_PH}{_BIG_PH}](?:[\\s\\-]*[{_GAP_PH}{_BIG_PH}])+")
_DETERMINATIVE_RE = re.compile(r"\{\s*([^{}]*?)\s*\}")
_HTML_SUP_RE = re.compile(r"<\s*sup\s*>(.*?)<\s*/\s*sup\s*>", re.I | re.S)
_HTML_SUB_RE = re.compile(r"<\s*sub\s*>(.*?)<\s*/\s*sub\s*>", re.I | re.S)
_HTML_TAG_RE = re.compile(
    r"<\s*/?\s*(?:i|b|em|strong|span|div|p|br|u|small|font|td|tr|th|table|tbody|thead|sup|sub)\b[^<>]*>",
    re.I,
)
_LATEX_SUP_RE = re.compile(r"\$?\^\{([^{}]*)\}\$?|\$\^([^\s$]+)\$")
_LATEX_SUB_RE = re.compile(r"\$?_\{([^{}]*)\}\$?|\$_([^\s$]+)\$")
_APOSTROPHE_IN_WORD_RE = re.compile(r"(?<=[^\W\d_])['’ʼ‘](?=[^\W\d_])")


def is_missing(value: Any) -> bool:
    """True for ``None``, NaN and empty/whitespace-only strings."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        import pandas as pd  # local import keeps this module dependency-free

        if value is pd.NA or value is pd.NaT:
            return True
    except ImportError:  # pragma: no cover
        pass
    return isinstance(value, str) and not value.strip()


def to_text(value: Any) -> str:
    """``str(value)`` with ``None``/NaN mapped to ``""``."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if is_missing(value):
        return ""
    return str(value)


def basic_clean(text: Any) -> str:
    """Unicode NFC, HTML entities, exotic spaces/zero-width chars, ligatures, whitespace."""
    s = to_text(text)
    if not s:
        return ""
    s = html.unescape(s)
    s = unicodedata.normalize("NFC", s)
    s = _ZERO_WIDTH_RE.sub("", s)
    s = _SPACES_RE.sub(" ", s)
    s = s.translate(_LIGATURES)
    return _WS_RE.sub(" ", s).strip()


def _protect_gaps(s: str) -> str:
    s = _EXISTING_BIG_RE.sub(_BIG_PH, s)
    return _EXISTING_GAP_RE.sub(_GAP_PH, s)


def _collapse_placeholders(s: str) -> str:
    return _GAP_RUN_RE.sub(_BIG_PH, s)


def _restore_gaps(s: str) -> str:
    s = _collapse_placeholders(s)
    s = s.replace(_BIG_PH, f" {BIG_GAP} ").replace(_GAP_PH, f" {GAP} ")
    # a gap glued to a syllable keeps its hyphen: "<gap> -ma" -> "<gap>-ma"
    s = re.sub(r"(<(?:big_)?gap>)\s+-", r"\1-", s)
    s = re.sub(r"-\s+(<(?:big_)?gap>)", r"-\1", s)
    return s


def collapse_gaps(text: Any) -> str:
    """Merge runs of adjacent gap markers: ``<gap> <gap>`` / ``<gap> <big_gap>`` -> ``<big_gap>``."""
    s = _protect_gaps(to_text(text))
    s = _restore_gaps(s)
    return _WS_RE.sub(" ", s).strip()


def _markup_to_plain(s: str, *, superscript_as_determinative: bool) -> str:
    """HTML/LaTeX super/subscripts (from OCR engines) -> ``{det}`` / digits / removed."""

    def sup(m: re.Match) -> str:
        inner = (m.group(1) if m.group(1) is not None else m.group(2) or "").strip()
        if not inner or re.fullmatch(r"[\d*†‡§,\s]+", inner):  # footnote marker
            return ""
        if not superscript_as_determinative:
            return ""
        return "{" + inner + "}"

    def sub(m: re.Match) -> str:
        return (m.group(1) if m.group(1) is not None else m.group(2) or "").strip()

    s = _HTML_SUP_RE.sub(sup, s)
    s = _HTML_SUB_RE.sub(sub, s)
    s = _LATEX_SUP_RE.sub(sup, s)
    s = _LATEX_SUB_RE.sub(sub, s)
    s = _HTML_TAG_RE.sub(" ", s)
    return s.replace("$", "")


def _unicode_superscripts(s: str, *, superscript_as_determinative: bool) -> str:
    def run(m: re.Match) -> str:
        if not superscript_as_determinative:
            return ""
        letters = "".join(_SUPER_LETTERS.get(c, c) for c in m.group(0))
        return "{" + letters.translate(_SUPER_DIGIT_MAP).translate(_SUBSCRIPTS) + "}"

    s = _SUPER_RUN_RE.sub(run, s)
    return _SUPER_DIGIT_RE.sub("", s)  # remaining superscript digits are footnote markers


# --------------------------------------------------------------------------------------
# source side
# --------------------------------------------------------------------------------------
def normalize_transliteration(
    text: Any,
    *,
    h_to_plain: bool = True,
    subscripts_to_digits: bool = True,
    x_to_gap: bool = True,
    lowercase_determinatives: bool = True,
) -> str:
    """Normalise an Akkadian transliteration to the competition's suggested format.

    >>> normalize_transliteration("[u]m-ma kà-ru-um ˹Kà˺-ni-iš ša-qí-il₅ [x] ... a-lim<sup>ki</sup>!")
    'um-ma kà-ru-um Kà-ni-iš ša-qí-il5 <big_gap> a-lim{ki}'
    """
    s = basic_clean(text)
    if not s:
        return ""
    s = _protect_gaps(s)
    s = _markup_to_plain(s, superscript_as_determinative=True)
    s = _unicode_superscripts(s, superscript_as_determinative=True)
    if subscripts_to_digits:
        s = s.translate(_SUBSCRIPTS)
    if h_to_plain:
        s = s.replace("ḫ", "h").replace("Ḫ", "H")
    s = s.translate(_DASHES)

    # --- breaks -> gap placeholders (before brackets are stripped) ---
    s = _BROKEN_COMMENT_RE.sub(_BIG_PH, s)
    s = _BREAK_WORD_RE.sub(_BIG_PH, s)
    s = _BRACKETED_ELLIPSIS_RE.sub(_BIG_PH, s)
    s = _ELLIPSIS_RUN_RE.sub(_BIG_PH, s)
    s = _BRACKETED_XS_RE.sub(_BIG_PH, s)
    s = _BRACKETED_X_RE.sub(_GAP_PH, s)

    # --- modern scribal notation ---
    s = _DOUBLE_ANGLE_RE.sub(" ", s)  # << >> errant signs: drop with content
    s = _SINGLE_ANGLE_RE.sub(r"\1", s)  # < > scribal insertions: keep the text
    s = s.replace("<", "").replace(">", "")
    s = s.translate(_HALF_BRACKETS)
    s = s.replace("[", "").replace("]", "")
    s = _ERASURE_COMMENT_RE.sub(" ", s)
    s = s.replace("(", "").replace(")", "")
    if x_to_gap:
        s = _XS_RE_I.sub(_BIG_PH, s)
        s = _X_RE_I.sub(_GAP_PH, s)
    s = s.replace("!", "").replace("?", "")
    s = s.replace(":", " ").replace("/", " ")
    s = re.sub(r"(?<!\S)\.+(?!\S)", " ", s)  # stand-alone "." word divider (keeps KÙ.BABBAR)

    # --- determinatives ---
    def det(m: re.Match) -> str:
        inner = m.group(1).replace(" ", "")
        if not inner:
            return ""
        return "{" + (inner.lower() if lowercase_determinatives else inner) + "}"

    s = _DETERMINATIVE_RE.sub(det, s)
    s = _APOSTROPHE_IN_WORD_RE.sub("ʾ", s)
    s = _restore_gaps(s)
    s = re.sub(r"\s*-\s*-+\s*", "-", s)  # "--" left behind by removed signs
    return _WS_RE.sub(" ", s).strip()


# --------------------------------------------------------------------------------------
# target side
# --------------------------------------------------------------------------------------
def normalize_gaps(text: Any, *, x_to_gap: bool = True) -> str:
    """Standardise missing-text markers in an English translation.

    ``...``, ``…``, ``[...]``, ``[x x]``, ``x x``, ``(broken)``, ``(rest missing)``,
    ``<big gap>`` -> ``<big_gap>``; ``[x]``, ``x``, ``< gap >`` -> ``<gap>``; adjacent markers
    are merged into one ``<big_gap>``.  With ``x_to_gap`` (default, as in the ensemble) a
    stand-alone ``x`` (an unread number/sign, e.g. "x shekels") also becomes ``<gap>``.
    """
    s = to_text(text)
    if not s:
        return ""
    s = _protect_gaps(s)
    s = _BROKEN_COMMENT_RE.sub(_BIG_PH, s)
    s = _BREAK_WORD_RE.sub(_BIG_PH, s)
    s = _BRACKETED_ELLIPSIS_RE.sub(_BIG_PH, s)
    s = _ELLIPSIS_RUN_RE.sub(_BIG_PH, s)
    s = _BRACKETED_XS_RE.sub(_BIG_PH, s)
    s = _BRACKETED_X_RE.sub(_GAP_PH, s)
    if x_to_gap:
        s = _XS_RE.sub(_BIG_PH, s)
        s = _X_RE.sub(_GAP_PH, s)
    s = _restore_gaps(s)
    s = _WS_RE.sub(" ", s).strip()
    # "<gap> ." / "<big_gap> ," -> attach punctuation like any other word
    return re.sub(r"(<(?:big_)?gap>) ([,.;:!?])", r"\1\2", s)


def normalize_translation(text: Any, *, x_to_gap: bool = True, ascii_quotes: bool = False) -> str:
    """Light, idempotent clean-up of an English translation (training target / model output)."""
    s = basic_clean(text)
    if not s:
        return ""
    s = _protect_gaps(s)
    s = _markup_to_plain(s, superscript_as_determinative=False)  # <sup>12</sup> footnotes etc.
    s = _unicode_superscripts(s, superscript_as_determinative=False)
    if ascii_quotes:
        s = s.translate(str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "‚": "'"}))
    s = s.replace(_GAP_PH, f" {GAP} ").replace(_BIG_PH, f" {BIG_GAP} ")
    s = normalize_gaps(s, x_to_gap=x_to_gap)
    return _WS_RE.sub(" ", s).strip()


def collapse_repeats(text: Any, max_n: int = 8, min_repeats: int = 3) -> str:
    """Collapse degenerate generation loops: an n-gram (n <= ``max_n``) repeated
    ``min_repeats``+ times back-to-back is kept once ("of the of the of the" -> "of the")."""
    toks: List[str] = to_text(text).split()
    if len(toks) < min_repeats:
        return " ".join(toks)
    changed = True
    while changed:
        changed = False
        for n in range(1, max_n + 1):
            out: List[str] = []
            i = 0
            while i < len(toks):
                unit = toks[i : i + n]
                if len(unit) < n:
                    out.extend(toks[i:])
                    break
                reps = 1
                while toks[i + reps * n : i + (reps + 1) * n] == unit:
                    reps += 1
                if reps >= min_repeats:
                    out.extend(unit)
                    i += reps * n
                    changed = True
                else:
                    out.append(toks[i])
                    i += 1
            toks = out
    return " ".join(toks)


_ECHO_PREFIX_RE = re.compile(r"^\s*(?:translation|english|translate[^:]{0,60})\s*:\s*", re.I)


def postprocess_translation(text: Any, *, x_to_gap: bool = True) -> str:
    """Final clean-up applied to every generated candidate before MBR / submission."""
    s = _ECHO_PREFIX_RE.sub("", to_text(text))
    s = normalize_translation(s, x_to_gap=x_to_gap)
    return collapse_repeats(s)


# --------------------------------------------------------------------------------------
# OCR heuristics (AKT pipeline only)
# --------------------------------------------------------------------------------------
_UPPER = "A-ZÀ-ÝŠṢṬḪĀĒĪŪ"
_GLUED_D_RE = re.compile(rf"(?<![\w{{}}\-])d(?=[{_UPPER}])")
_PAREN_DET_RE = re.compile(
    r"\((" + "|".join(sorted((re.escape(d) for d in DETERMINATIVES), key=len, reverse=True)) + r")\)(?=\S)|"
    r"(?<=\S)\((" + "|".join(sorted((re.escape(d) for d in DETERMINATIVES), key=len, reverse=True)) + r")\)",
    re.I,
)


_VOWELS = set("aeiouáàéèíìóòúùāēīōūâêîôû")


def restore_glued_determinatives(text: Any) -> str:
    """Re-insert determinative braces that OCR flattened into the running text.

    * ``dA-šùr`` / ``dUTU`` -> ``{d}A-šùr`` / ``{d}UTU`` (a lower-case ``d`` glued to an
      upper-case sign at the start of a word is never a syllable);
    * ``a-limki`` / ``Kà-ni-iški`` -> ``a-lim{ki}`` / ``Kà-ni-iš{ki}`` (``ki`` glued to the end
      of a hyphen-segment of 2+ letters cannot be a CV syllable);
    * ``(d)A-šùr`` / ``a-lim(ki)`` -> ``{d}A-šùr`` / ``a-lim{ki}``.

    Only use this on OCR output; competition data already carries ``{...}``.
    """
    s = to_text(text)
    if not s:
        return ""
    s = _PAREN_DET_RE.sub(lambda m: "{" + (m.group(1) or m.group(2)).lower() + "}", s)
    s = _GLUED_D_RE.sub("{d}", s)

    def ki(word: str) -> str:
        if "{" in word or "<" in word:
            return word
        parts = word.split("-")
        last = parts[-1]
        core = re.sub(r"[^\w]", "", last)
        stem = core[:-2]
        if (
            len(core) >= 4
            and core.endswith("ki")
            and stem.isalpha()
            and core.islower()
            and stem[-1] not in _VOWELS
        ):
            parts[-1] = last[: last.rfind("ki")] + "{ki}" + last[last.rfind("ki") + 2 :]
        return "-".join(parts)

    return " ".join(ki(w) for w in s.split())
