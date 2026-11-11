"""Competition metric: geometric mean of corpus BLEU and corpus chrF++.

The Deep Past Challenge scores a submission with ``sqrt(BLEU * chrF++)`` where both
scores are *micro-averaged* over the corpus (sufficient statistics are summed over all
sentences before the score is computed), exactly like

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    score = (bleu * chrf) ** 0.5

This module is a dependency-free re-implementation of sacrebleu 2.x (13a tokenizer, ``exp``
smoothing, chrF with ``char_order=6, word_order=2, beta=2``) that is verified against
sacrebleu in ``tests/test_metric.py``.  On top of the corpus metric it exposes cached
per-sentence statistics (:class:`SentenceStats`) so that MBR decoding can score every
ordered candidate pair of a 32-candidate pool quickly: n-gram match counts are symmetric,
so each unordered pair is intersected once and both directions are derived from it.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

__all__ = [
    "BLEU_ORDER",
    "CHRF_CHAR_ORDER",
    "CHRF_WORD_ORDER",
    "CHRF_BETA",
    "tokenize_13a",
    "sentence_bleu",
    "sentence_chrf",
    "sentence_score",
    "corpus_bleu",
    "corpus_chrf",
    "corpus_score",
    "competition_score",
    "SentenceStats",
    "sentence_stats",
    "pair_scores",
    "pairwise_matrix",
]

BLEU_ORDER = 4
CHRF_CHAR_ORDER = 6
CHRF_WORD_ORDER = 2
CHRF_BETA = 2
_CHRF_ORDERS = CHRF_CHAR_ORDER + CHRF_WORD_ORDER
_EPS = 1e-16

# --------------------------------------------------------------------------------------
# Tokenisation (sacrebleu Tokenizer13a + TokenizerRegexp)
# --------------------------------------------------------------------------------------
_13A_RULES = (
    # language-dependent part (assuming Western languages)
    (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),
    # tokenize period and comma unless preceded by a digit
    (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),
    # tokenize period and comma unless followed by a digit
    (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),
    # tokenize dash when preceded by a digit
    (re.compile(r"([0-9])(-)"), r"\1 \2 "),
)

# chrF++ "primitive tokenisation" punctuation set (== string.punctuation)
_PUNCTS = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


@lru_cache(maxsize=2**16)
def tokenize_13a(line: str) -> Tuple[str, ...]:
    """Tokenise ``line`` exactly like sacrebleu's default ``13a`` tokenizer used by BLEU."""
    line = line.rstrip()  # BLEU._preprocess_segment strips trailing whitespace first
    line = line.replace("<skipped>", "")
    line = line.replace("-\n", "")
    line = line.replace("\n", " ")
    if "&" in line:
        line = line.replace("&quot;", '"')
        line = line.replace("&amp;", "&")
        line = line.replace("&lt;", "<")
        line = line.replace("&gt;", ">")
    line = f" {line} "
    for regex, repl in _13A_RULES:
        line = regex.sub(repl, line)
    return tuple(line.split())


def _chrf_words(sent: str) -> List[str]:
    """chrF++ word tokenisation: split on whitespace, peel one leading/trailing punct."""
    words: List[str] = []
    for w in sent.split():
        if len(w) == 1:
            words.append(w)
        elif w[-1] in _PUNCTS:
            words += [w[:-1], w[-1]]
        elif w[0] in _PUNCTS:
            words += [w[0], w[1:]]
        else:
            words.append(w)
    return words


# --------------------------------------------------------------------------------------
# Cached per-sentence statistics
# --------------------------------------------------------------------------------------
class SentenceStats:
    """Pre-computed n-gram statistics of one sentence (usable as hypothesis or reference)."""

    __slots__ = ("text", "bleu_len", "bleu_ngrams", "bleu_totals", "chrf_ngrams", "chrf_counts")

    def __init__(self, text: str):
        self.text = text
        toks = tokenize_13a(text)
        n_tok = len(toks)
        self.bleu_len = n_tok
        self.bleu_ngrams: List[Counter] = [
            Counter(toks[i : i + n] for i in range(n_tok - n + 1)) for n in range(1, BLEU_ORDER + 1)
        ]
        self.bleu_totals: List[int] = [max(0, n_tok - n + 1) for n in range(1, BLEU_ORDER + 1)]

        chars = "".join(text.split())
        n_ch = len(chars)
        grams: List[Counter] = [
            Counter(chars[i : i + n] for i in range(n_ch - n + 1)) for n in range(1, CHRF_CHAR_ORDER + 1)
        ]
        words = _chrf_words(text)
        n_w = len(words)
        grams += [
            Counter(" ".join(words[i : i + n]) for i in range(n_w - n + 1)) for n in range(1, CHRF_WORD_ORDER + 1)
        ]
        self.chrf_ngrams = grams
        self.chrf_counts: List[int] = [sum(c.values()) for c in grams]

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"SentenceStats({self.text!r})"


@lru_cache(maxsize=2**15)
def sentence_stats(text: str) -> SentenceStats:
    """Return (cached) :class:`SentenceStats` for ``text``."""
    return SentenceStats(text)


def _match(a: Counter, b: Counter) -> int:
    """Clipped n-gram intersection size (symmetric)."""
    if len(a) > len(b):
        a, b = b, a
    total = 0
    for gram, count in a.items():
        other = b.get(gram)
        if other:
            total += count if count < other else other
    return total


# --------------------------------------------------------------------------------------
# Score formulas (identical to sacrebleu.BLEU.compute_bleu / CHRF._compute_f_score)
# --------------------------------------------------------------------------------------
def _bleu_from_stats(
    correct: Sequence[float],
    total: Sequence[float],
    sys_len: int,
    ref_len: int,
    effective_order: bool,
) -> float:
    """BLEU with ``exp`` smoothing (sacrebleu default)."""
    bp = 1.0
    if sys_len < ref_len:
        bp = math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0
    if not any(correct):
        return 0.0
    precisions = [0.0] * BLEU_ORDER
    smooth_mteval = 1.0
    eff_order = BLEU_ORDER
    for n in range(1, BLEU_ORDER + 1):
        if total[n - 1] == 0:
            break
        if effective_order:
            eff_order = n
        if correct[n - 1] == 0:
            smooth_mteval *= 2
            precisions[n - 1] = 100.0 / (smooth_mteval * total[n - 1])
        else:
            precisions[n - 1] = 100.0 * correct[n - 1] / total[n - 1]
    log_sum = 0.0
    for p in precisions[:eff_order]:
        log_sum += math.log(p) if p > 0 else -9999999999
    return bp * math.exp(log_sum / eff_order)


def _chrf_from_stats(stats: Sequence[float], beta: int = CHRF_BETA) -> float:
    """chrF from flattened ``[hyp, ref, match] * order`` statistics (no eps smoothing)."""
    factor = beta**2
    avg_prec = avg_rec = 0.0
    eff = 0
    for i in range(len(stats) // 3):
        n_hyp, n_ref, n_match = stats[3 * i], stats[3 * i + 1], stats[3 * i + 2]
        if n_hyp > 0 and n_ref > 0:
            avg_prec += n_match / n_hyp
            avg_rec += n_match / n_ref
            eff += 1
    if eff == 0:
        return 0.0
    avg_prec /= eff
    avg_rec /= eff
    if avg_prec + avg_rec:
        return 100.0 * (1 + factor) * avg_prec * avg_rec / (factor * avg_prec + avg_rec)
    return 0.0


def _geo(bleu: float, chrf: float) -> float:
    return math.sqrt(max(bleu, 0.0) * max(chrf, 0.0))


# --------------------------------------------------------------------------------------
# Segment / pair scoring
# --------------------------------------------------------------------------------------
def _bleu_segment(h: SentenceStats, r: SentenceStats) -> Tuple[List[int], List[int]]:
    correct = [_match(h.bleu_ngrams[n], r.bleu_ngrams[n]) for n in range(BLEU_ORDER)]
    return correct, list(h.bleu_totals)


def _chrf_segment(h: SentenceStats, r: SentenceStats, matches: Sequence[int] | None = None) -> List[int]:
    if matches is None:
        matches = [_match(h.chrf_ngrams[k], r.chrf_ngrams[k]) for k in range(_CHRF_ORDERS)]
    stats: List[int] = []
    for k in range(_CHRF_ORDERS):
        ref_count = r.chrf_counts[k]
        # sacrebleu: "Don't count hits if no reference exists for that n-gram"
        stats += [h.chrf_counts[k] if ref_count else 0, ref_count, matches[k]]
    return stats


def sentence_bleu(hyp: str, ref: str) -> float:
    """== ``sacrebleu.sentence_bleu(hyp, [ref]).score`` (effective order, exp smoothing)."""
    h, r = sentence_stats(hyp), sentence_stats(ref)
    correct, total = _bleu_segment(h, r)
    return _bleu_from_stats(correct, total, h.bleu_len, r.bleu_len, effective_order=True)


def sentence_chrf(hyp: str, ref: str) -> float:
    """== ``sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score``."""
    return _chrf_from_stats(_chrf_segment(sentence_stats(hyp), sentence_stats(ref)))


def sentence_score(hyp: str, ref: str) -> float:
    """Sentence-level geometric mean of BLEU and chrF++ (the MBR utility)."""
    return _geo(sentence_bleu(hyp, ref), sentence_chrf(hyp, ref))


def pair_scores(a: SentenceStats, b: SentenceStats, utility: str = "geo") -> Tuple[float, float]:
    """Return ``(u(hyp=a, ref=b), u(hyp=b, ref=a))`` sharing one n-gram intersection.

    ``utility`` is one of ``"geo"`` (sqrt(BLEU*chrF++), default), ``"chrf"`` or ``"bleu"``.
    """
    need_bleu = utility in ("geo", "bleu")
    need_chrf = utility in ("geo", "chrf")
    if utility not in ("geo", "bleu", "chrf"):
        raise ValueError(f"unknown utility {utility!r}")
    bleu_ab = bleu_ba = chrf_ab = chrf_ba = 0.0
    if need_bleu:
        correct = [_match(a.bleu_ngrams[n], b.bleu_ngrams[n]) for n in range(BLEU_ORDER)]
        bleu_ab = _bleu_from_stats(correct, a.bleu_totals, a.bleu_len, b.bleu_len, effective_order=True)
        bleu_ba = _bleu_from_stats(correct, b.bleu_totals, b.bleu_len, a.bleu_len, effective_order=True)
    if need_chrf:
        matches = [_match(a.chrf_ngrams[k], b.chrf_ngrams[k]) for k in range(_CHRF_ORDERS)]
        chrf_ab = _chrf_from_stats(_chrf_segment(a, b, matches))
        chrf_ba = _chrf_from_stats(_chrf_segment(b, a, matches))
    if utility == "bleu":
        return bleu_ab, bleu_ba
    if utility == "chrf":
        return chrf_ab, chrf_ba
    return _geo(bleu_ab, chrf_ab), _geo(bleu_ba, chrf_ba)


def pairwise_matrix(texts: Sequence[str], utility: str = "geo") -> List[List[float]]:
    """``M[i][j] = utility(hyp=texts[i], ref=texts[j])``; the diagonal is the self-score.

    The self-score is 100 for any non-empty string and 0 for an empty one.
    """
    stats = [sentence_stats(t) for t in texts]
    n = len(stats)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        mat[i][i] = 100.0 if texts[i].strip() else 0.0
        for j in range(i + 1, n):
            mat[i][j], mat[j][i] = pair_scores(stats[i], stats[j], utility)
    return mat


# --------------------------------------------------------------------------------------
# Corpus level (the competition metric)
# --------------------------------------------------------------------------------------
def _check_lengths(hyps: Sequence[str], refs: Sequence[str]) -> None:
    if len(hyps) != len(refs):
        raise ValueError(f"{len(hyps)} hypotheses vs {len(refs)} references")


def corpus_bleu(hyps: Sequence[str], refs: Sequence[str]) -> float:
    """== ``sacrebleu.corpus_bleu(hyps, [refs]).score``."""
    _check_lengths(hyps, refs)
    correct = [0] * BLEU_ORDER
    total = [0] * BLEU_ORDER
    sys_len = ref_len = 0
    for hyp, ref in zip(hyps, refs):
        h, r = sentence_stats(hyp), sentence_stats(ref)
        c, t = _bleu_segment(h, r)
        for n in range(BLEU_ORDER):
            correct[n] += c[n]
            total[n] += t[n]
        sys_len += h.bleu_len
        ref_len += r.bleu_len
    return _bleu_from_stats(correct, total, sys_len, ref_len, effective_order=False)


def corpus_chrf(hyps: Sequence[str], refs: Sequence[str]) -> float:
    """== ``sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score``."""
    _check_lengths(hyps, refs)
    agg = [0] * (3 * _CHRF_ORDERS)
    for hyp, ref in zip(hyps, refs):
        seg = _chrf_segment(sentence_stats(hyp), sentence_stats(ref))
        for k, v in enumerate(seg):
            agg[k] += v
    return _chrf_from_stats(agg)


def corpus_score(hyps: Sequence[str], refs: Sequence[str]) -> Dict[str, float]:
    """Return ``{"bleu", "chrf", "geo_mean"}`` — ``geo_mean`` is the leaderboard metric."""
    bleu = corpus_bleu(hyps, refs)
    chrf = corpus_chrf(hyps, refs)
    return {"bleu": bleu, "chrf": chrf, "geo_mean": _geo(bleu, chrf)}


def competition_score(hyps: Sequence[str], refs: Sequence[str]) -> float:
    """The Deep Past leaderboard metric: ``sqrt(corpus BLEU * corpus chrF++)``."""
    return corpus_score(hyps, refs)["geo_mean"]


if __name__ == "__main__":  # pragma: no cover - tiny CLI: score a submission against references
    import argparse

    import pandas as pd

    ap = argparse.ArgumentParser(description="Score predictions with the competition metric.")
    ap.add_argument("predictions", help="CSV with columns id,translation")
    ap.add_argument("references", help="CSV with columns id,translation (or --ref_col)")
    ap.add_argument("--ref_col", default="translation")
    ap.add_argument("--pred_col", default="translation")
    args = ap.parse_args()
    pred = pd.read_csv(args.predictions)
    ref = pd.read_csv(args.references)
    merged = ref.merge(pred, on="id", suffixes=("_ref", "_pred"), how="left")
    rc = args.ref_col + "_ref" if args.ref_col == args.pred_col else args.ref_col
    pc = args.pred_col + "_pred" if args.ref_col == args.pred_col else args.pred_col
    hyps = merged[pc].fillna("").astype(str).tolist()
    refs = merged[rc].fillna("").astype(str).tolist()
    print(corpus_score(hyps, refs))
