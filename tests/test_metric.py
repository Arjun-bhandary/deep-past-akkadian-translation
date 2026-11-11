"""The pure-python metric must reproduce sacrebleu exactly (sentence and corpus level)."""
import math
import random

import pytest

from deep_past import metric

sacrebleu = pytest.importorskip("sacrebleu")

SENTS = [
    "Thus Kanesh, say to the dātu-payers, our messenger, every single colony,",
    "Thus kārum Kanesh, say to the dātu-payers: our messenger and every colony.",
    "In the letter of the City (it is written): 'From this day on, whoever buys meteoric iron,",
    "(the City of) Assur is not entitled to part of the profit made, the tithe on it",
    "kārum Kanesh will collect.",
    "<gap> shekels of silver <big_gap> to Aššur-idī",
    "x minas of copper, 5 1/2 shekels of tin; 1,000 donkeys!",
    "",
    "   ",
    "a",
    "Puzur-Aššur son of Aššur-malik owes 10 minas of silver &amp; 2 textiles.",
    "said: \"Give me the tablet\" (and) he did not (give it).",
    "1-2-3 4.5 6,7 a.b c,d",
]


def _random_sent(rng: random.Random) -> str:
    vocab = ["the", "silver", "of", "Aššur", "kārum", "minas", ",", ".", "'s", "(", ")", "x",
             "<gap>", "5", "1/2", "tin", "textiles", "Šalim-ahum", "said:", "\"", "-", "a-na"]
    return " ".join(rng.choice(vocab) for _ in range(rng.randint(0, 25)))


@pytest.mark.parametrize("hyp", SENTS)
@pytest.mark.parametrize("ref", SENTS)
def test_sentence_scores_match_sacrebleu(hyp, ref):
    exp_bleu = sacrebleu.sentence_bleu(hyp, [ref]).score
    exp_chrf = sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score
    assert metric.sentence_bleu(hyp, ref) == pytest.approx(exp_bleu, abs=1e-9)
    assert metric.sentence_chrf(hyp, ref) == pytest.approx(exp_chrf, abs=1e-9)


def test_tokenizer_matches_sacrebleu():
    from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a

    tok = Tokenizer13a()
    rng = random.Random(0)
    for s in SENTS + [_random_sent(rng) for _ in range(200)]:
        assert " ".join(metric.tokenize_13a(s)) == tok(s.rstrip())


def test_corpus_scores_match_sacrebleu():
    rng = random.Random(1)
    hyps = [_random_sent(rng) for _ in range(300)] + SENTS
    refs = [_random_sent(rng) for _ in range(300)] + SENTS[::-1]
    exp_bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    exp_chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    got = metric.corpus_score(hyps, refs)
    assert got["bleu"] == pytest.approx(exp_bleu, abs=1e-9)
    assert got["chrf"] == pytest.approx(exp_chrf, abs=1e-9)
    assert got["geo_mean"] == pytest.approx(math.sqrt(exp_bleu * exp_chrf), abs=1e-9)
    assert metric.competition_score(hyps, refs) == pytest.approx(got["geo_mean"])


def test_pairwise_matrix_matches_directional_scores():
    rng = random.Random(2)
    texts = [_random_sent(rng) for _ in range(12)] + SENTS[:4]
    mat = metric.pairwise_matrix(texts)
    for i, a in enumerate(texts):
        for j, b in enumerate(texts):
            if i == j:
                assert mat[i][i] == (100.0 if a.strip() else 0.0)
            else:
                assert mat[i][j] == pytest.approx(metric.sentence_score(a, b), abs=1e-9)
    for util, fn in (("bleu", metric.sentence_bleu), ("chrf", metric.sentence_chrf)):
        m = metric.pairwise_matrix(texts[:6], utility=util)
        assert m[0][1] == pytest.approx(fn(texts[0], texts[1]), abs=1e-9)


def test_identical_is_perfect_and_bounds():
    s = "Thus kārum Kanesh, say to the dātu-payers"
    assert metric.sentence_score(s, s) == pytest.approx(100.0)
    assert metric.corpus_score([s], [s])["geo_mean"] == pytest.approx(100.0)
    assert metric.sentence_score("", s) == 0.0
    with pytest.raises(ValueError):
        metric.corpus_score(["a"], [])
