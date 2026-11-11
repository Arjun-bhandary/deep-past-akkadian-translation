import pytest

from deep_past.codec import CharCodec, placeholder_for


def test_placeholders_are_unique_and_ascii():
    chars = "šŠṣṢṭṬḫḪḥāĀēĒīĪūŪâêîûáàéèíìúùäöüăżñ"
    phs = [placeholder_for(c) for c in chars]
    assert len(set(phs)) == len(phs)
    assert all(p.isascii() for p in phs)
    assert placeholder_for("š") == "s~" and placeholder_for("Ṣ") == "S^" and placeholder_for("ā") == "a="
    assert placeholder_for("ʾ").startswith("~")


def test_roundtrip_and_persistence(tmp_path):
    codec = CharCodec({"š": "sz"}, {c: placeholder_for(c) for c in "šŠṣṭāīū"})
    text = "Aššur-idī said to Šalim-ahum: the kārum of Kaneš, ṭuppum, ṣuhārum"
    enc = codec.encode_target(text)
    assert enc.isascii()
    assert codec.decode_target(enc) == text
    assert codec.encode_source("ša-ru-um") == "sza-ru-um"
    codec.save(tmp_path)
    again = CharCodec.load(tmp_path)
    assert again.decode_target(enc) == text
    assert CharCodec.load(tmp_path / "missing").is_identity


def test_duplicate_target_placeholders_rejected():
    with pytest.raises(ValueError):
        CharCodec(None, {"a": "x~", "b": "x~"})


class _FakeTok:
    """ASCII-only tokenizer: anything non-ASCII is <unk> (id 0)."""
    unk_token_id = 0

    def __call__(self, text=None, text_target=None, add_special_tokens=False):
        s = text_target if text_target is not None else text
        return {"input_ids": [ord(c) if c.isascii() else 0 for c in s]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i)


def test_fit_detects_unknown_chars():
    codec = CharCodec.fit(_FakeTok(), ["ša-ru-um a-na"], ["Aššur and the kārum"])
    assert set(codec.source_map) == {"š"}
    assert set(codec.target_map) == {"š", "ā"}
    assert codec.decode_target(codec.encode_target("Aššur kārum")) == "Aššur kārum"


@pytest.mark.network
def test_fit_on_real_marian_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ROMANCE-en")
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip(f"hub unavailable: {exc}")
    srcs = ["ša-ru-um a-na ṭup-pí-im ṣa-ba-tim ší-mì-im ú-lá", "{d}UTU KÙ.BABBAR"]
    tgts = ["Aššur-idī said to Šalim-ahum: the kārum of Kanesh will collect the ṭuppum"]
    codec = CharCodec.fit(tok, srcs, tgts)
    for t in tgts:
        enc = codec.encode_target(t)
        ids = tok(text_target=enc, add_special_tokens=False)["input_ids"]
        assert tok.unk_token_id not in ids
        assert codec.decode_target(tok.decode(ids, skip_special_tokens=True)) == t
