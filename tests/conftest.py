"""Shared fixtures: a tiny, randomly initialised byT5 model built fully offline."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="session")
def tiny_byt5_dir(tmp_path_factory):
    """Directory with a ~0.2M-parameter random T5 + ByT5Tokenizer (no network needed)."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from transformers import ByT5Tokenizer, T5Config, T5ForConditionalGeneration

    out = tmp_path_factory.mktemp("tiny_byt5")
    tok = ByT5Tokenizer()
    cfg = T5Config(
        vocab_size=384, d_model=32, d_kv=8, d_ff=64, num_layers=1, num_decoder_layers=1, num_heads=2,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1, feed_forward_proj="gated-gelu",
        tie_word_embeddings=False,
    )
    model = T5ForConditionalGeneration(cfg)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return str(out)
