import math

import pytest

from deep_past.normalize import (
    BIG_GAP,
    GAP,
    collapse_gaps,
    collapse_repeats,
    is_missing,
    normalize_gaps,
    normalize_translation,
    normalize_transliteration,
    postprocess_translation,
    restore_glued_determinatives,
)

TRANSLIT_CASES = [
    # (raw, expected)
    ("[u]m-ma kà-ru-um ˹Kà˺-ni-iš ša-qí-il₅ [x] ... a-lim<sup>ki</sup>!",
     "um-ma kà-ru-um Kà-ni-iš ša-qí-il5 <big_gap> a-lim{ki}"),
    ("a-nim : ma-ma-an KÙ.AN", "a-nim ma-ma-an KÙ.AN"),
    ("ú-lá e-WA : iš-ra-tí-šu", "ú-lá e-WA iš-ra-tí-šu"),
    ("né-mì-lim ᵈA-šùr", "né-mì-lim {d}A-šùr"),
    ("a-limᵏⁱ iš-tù u₄-mì-im", "a-lim{ki} iš-tù u4-mì-im"),
    ("ḫa-ra-an Ḫu-ra-ṣa-nim", "ha-ra-an Hu-ra-ṣa-nim"),
    ("[x] KÙ.BABBAR [x x] a-na", "<gap> KÙ.BABBAR <big_gap> a-na"),
    ("[… …] i-dí-in", "<big_gap> i-dí-in"),
    ("x-ma a-na x x x", "<gap>-ma a-na <big_gap>"),
    ("a-na <<a-na>> <be>-lí-a", "a-na be-lí-a"),
    ("i-qí-bi₄-ma (erasure) a-na / be-lí", "i-qí-bi4-ma a-na be-lí"),
    ("lu ta-dí-in! a-wi-lum?", "lu ta-dí-in a-wi-lum"),
    ("(rest broken)", "<big_gap>"),
    ("<gap> <gap> a-na <big_gap> <gap>", "<big_gap> a-na <big_gap>"),
    ("&lt;gap&gt; KÙ.BABBAR", "<gap> KÙ.BABBAR"),
    ("{D}UTU {KI} { ki }", "{d}UTU {ki} {ki}"),
    ("A-šur-i-dí DUMU Šu-ku-bi₄-im", "A-šur-i-dí DUMU Šu-ku-bi4-im"),
    ("ša–ra–tim    a-na​", "ša-ra-tim a-na"),
    ("KAxU ŠEₓ", "KAxU ŠEx"),
    ("", ""),
]


@pytest.mark.parametrize("raw,expected", TRANSLIT_CASES)
def test_normalize_transliteration(raw, expected):
    assert normalize_transliteration(raw) == expected


@pytest.mark.parametrize("raw,_", TRANSLIT_CASES)
def test_transliteration_idempotent(raw, _):
    once = normalize_transliteration(raw)
    assert normalize_transliteration(once) == once


def test_competition_format_is_stable():
    # already-normalised competition text must not change
    s = "um-ma kà-ru-um kà-ni-iš-ma a-na da-tim <gap> KÙ.BABBAR {d}UTU a-lim{ki} <big_gap> ší-mì-im"
    assert normalize_transliteration(s) == s


GAP_CASES = [
    ("Thus kārum Kanesh, say to the ... payers", f"Thus kārum Kanesh, say to the {BIG_GAP} payers"),
    ("give him […] silver", f"give him {BIG_GAP} silver"),
    ("x shekels of silver", f"{GAP} shekels of silver"),
    ("x x minas", f"{BIG_GAP} minas"),
    ("[x] textiles (broken) and", f"{GAP} textiles {BIG_GAP} and"),
    ("< gap > < big gap > <GAP>", BIG_GAP),
    ("the rest is (rest broken).", f"the rest is {BIG_GAP}."),
    ("from Aššur . . . to Kanesh", f"from Aššur {BIG_GAP} to Kanesh"),
    ("exactly six textiles", "exactly six textiles"),
    ("Tax and excise", "Tax and excise"),
]


@pytest.mark.parametrize("raw,expected", GAP_CASES)
def test_normalize_gaps(raw, expected):
    assert normalize_gaps(raw) == expected
    assert normalize_gaps(expected) == expected


def test_normalize_translation_cleanup():
    s = "  “The meteoric iron belongs to<sup>12</sup>\n a man ²with an account”  "
    assert normalize_translation(s) == "“The meteoric iron belongs to a man with an account”"
    assert normalize_translation(s, ascii_quotes=True) == '"The meteoric iron belongs to a man with an account"'
    assert normalize_translation("ﬁne silver") == "fine silver"
    assert normalize_translation(None) == ""
    assert normalize_translation(float("nan")) == ""


def test_collapse_and_postprocess():
    assert collapse_repeats("of the of the of the silver") == "of the silver"
    assert collapse_repeats("one by one") == "one by one"
    assert collapse_repeats("a a a a b") == "a b"
    assert postprocess_translation("Translation: x shekels ... silver") == f"{GAP} shekels {BIG_GAP} silver"
    assert collapse_gaps("<gap> <gap> a") == f"{BIG_GAP} a"


def test_restore_glued_determinatives():
    assert restore_glued_determinatives("dA-šùr a-limki dUTU") == "{d}A-šùr a-lim{ki} {d}UTU"
    assert restore_glued_determinatives("Kà-ni-iški a-ki ša-ki") == "Kà-ni-iš{ki} a-ki ša-ki"
    assert restore_glued_determinatives("(d)A-šur a-lim(ki)") == "{d}A-šur a-lim{ki}"
    assert restore_glued_determinatives("DUMU da-tim") == "DUMU da-tim"


def test_is_missing():
    assert is_missing(None) and is_missing(float("nan")) and is_missing("  ")
    assert not is_missing("a") and not is_missing(0)
    assert math.isnan(float("nan"))
