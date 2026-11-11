import pytest

torch = pytest.importorskip("torch")

from deep_past.codec import CharCodec  # noqa: E402
from deep_past.modeling import detect_family, generate, load_seq2seq, pick_dtype, prepare_sources  # noqa: E402


def test_detect_family_and_dtype():
    assert detect_family("marian") == "marian"
    assert detect_family("t5") == "t5"
    assert detect_family("bart") == "auto"
    assert pick_dtype("auto", "t5", "cpu") == torch.float32
    assert pick_dtype("fp16", "t5", "cpu") == torch.float16
    with pytest.raises(ValueError):
        pick_dtype("int4", "t5", "cpu")


def test_prepare_sources():
    codec = CharCodec({"š": "sz"}, None)
    out = prepare_sources(["[x] ša-ru-um ḫa!"], prefix="translate Akkadian to English: ", codec=codec)
    assert out == ["translate Akkadian to English: <gap> sza-ru-um ha"]
    assert prepare_sources(["  a  b "], normalization="none") == ["  a  b "]
    assert prepare_sources(["  a  b "], normalization="light") == ["a b"]


def test_generate_beams_shape_and_order(tiny_byt5_dir):
    model, tok, codec, fam = load_seq2seq(tiny_byt5_dir, device="cpu")
    assert fam == "t5" and codec.is_identity
    srcs = ["a-na", "um-ma A-šur-i-dí-ma a-na Šu-ku-bi-im qí-bi-ma", "KÙ.BABBAR"]
    cands = generate(model, tok, srcs, num_beams=3, num_return_sequences=3, max_new_tokens=8, batch_size=2)
    assert len(cands) == 3
    for row in cands:
        assert len(row) == 3
        assert [c["rank"] for c in row] == [0, 1, 2]
        assert all(isinstance(c["text"], str) for c in row)
        assert all(c["score"] is not None for c in row)
        assert row[0]["score"] >= row[-1]["score"]
    greedy = generate(model, tok, srcs, num_beams=1, num_return_sequences=1, max_new_tokens=8)
    assert all(len(r) == 1 and r[0]["score"] is None for r in greedy)
