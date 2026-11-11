"""Seq2seq model loading, precision selection and batched beam-search generation.

Shared by training-time evaluation (:mod:`deep_past.train`) and inference
(:mod:`deep_past.infer`) so both see byte-identical pre/post-processing:

    model, tok, codec, family = load_seq2seq(path, dtype="auto", device="cuda")
    srcs = prepare_sources(texts, prefix=..., normalization="competition", codec=codec)
    cands = generate(model, tok, srcs, num_beams=4, num_return_sequences=4, codec=codec)
    # cands[i] == [{"text": ..., "score": ..., "rank": 0}, ...]  (len == num_return_sequences)

Works with transformers 4.4x and 5.x (weights are loaded in the checkpoint dtype and cast
afterwards instead of relying on the renamed ``torch_dtype``/``dtype`` kwarg).
"""
from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .codec import CharCodec
from .io_utils import resolve_path
from .normalize import basic_clean, normalize_transliteration, postprocess_translation, to_text

__all__ = [
    "FAMILIES",
    "detect_family",
    "pick_dtype",
    "load_seq2seq",
    "prepare_sources",
    "generate",
]

LOG = logging.getLogger(__name__)
FAMILIES = ("marian", "t5", "auto")
_T5_TYPES = {"t5", "mt5", "umt5", "longt5", "byt5"}


def detect_family(config: Any) -> str:
    """``marian`` / ``t5`` / ``auto`` from a transformers config (or its ``model_type``)."""
    model_type = getattr(config, "model_type", config if isinstance(config, str) else "")
    model_type = str(model_type).lower()
    if model_type == "marian":
        return "marian"
    if model_type in _T5_TYPES:
        return "t5"
    return "auto"


def pick_dtype(requested: str, family: str, device: Any):
    """Resolve ``auto|fp32|fp16|bf16`` into a torch dtype.

    ``auto``: fp32 on CPU; on CUDA fp16 for Marian, and for T5-family models bf16 on
    Ampere+ (compute capability >= 8) else fp32 — T5/byT5 overflow in fp16 and the T4/P100
    GPUs on Kaggle have no native bf16.
    """
    import torch

    requested = (requested or "auto").lower()
    table = {"fp32": torch.float32, "float32": torch.float32, "fp16": torch.float16,
             "float16": torch.float16, "half": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}
    if requested in table:
        return table[requested]
    if requested != "auto":
        raise ValueError(f"unknown dtype {requested!r}")
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type != "cuda":
        return torch.float32
    if family == "marian":
        return torch.float16
    major, _ = torch.cuda.get_device_capability(dev)
    return torch.bfloat16 if major >= 8 else torch.float32


def load_seq2seq(
    path: str,
    *,
    family: str = "auto",
    dtype: str = "auto",
    device: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
) -> Tuple[Any, Any, CharCodec, str]:
    """Load ``(model, tokenizer, codec, family)`` from a local dir or Hub id (``eval()`` mode)."""
    import torch
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

    path = resolve_path(path)
    tok_path = resolve_path(tokenizer_path) if tokenizer_path else path
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(path)
    fam = family if family and family != "auto" else detect_family(config)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
    except (OSError, ValueError, TypeError) as exc:
        if fam == "t5" and "byt5" in str(getattr(config, "_name_or_path", "")).lower() + str(path).lower():
            from transformers import ByT5Tokenizer  # byte-level: no vocabulary files needed

            LOG.warning("no tokenizer files in %s (%s); using ByT5Tokenizer()", tok_path, exc)
            tokenizer = ByT5Tokenizer()
        else:
            raise
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    torch_dtype = pick_dtype(dtype, fam, device)
    model.to(device=device, dtype=torch_dtype)
    model.eval()
    codec = CharCodec.load(path) if os.path.isdir(path) else CharCodec()
    LOG.info("loaded %s (%s, %s, %s, %.0fM params)", path, fam, str(torch_dtype).replace("torch.", ""), device,
             sum(p.numel() for p in model.parameters()) / 1e6)
    return model, tokenizer, codec, fam


def prepare_sources(
    texts: Sequence[Any],
    *,
    prefix: str = "",
    normalization: str = "competition",
    codec: Optional[CharCodec] = None,
) -> List[str]:
    """Apply source normalisation (``competition`` | ``light`` | ``none``), codec and prefix."""
    out: List[str] = []
    for t in texts:
        if normalization == "competition":
            s = normalize_transliteration(t)
        elif normalization == "light":
            s = basic_clean(t)
        elif normalization == "none":
            s = to_text(t)
        else:
            raise ValueError(f"unknown normalization {normalization!r}")
        if codec is not None:
            s = codec.encode_source(s)
        out.append(f"{prefix}{s}" if prefix else s)
    return out


def _is_oom(exc: BaseException) -> bool:
    import torch

    oom_type = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    return isinstance(exc, oom_type) or "out of memory" in str(exc).lower()


def generate(
    model: Any,
    tokenizer: Any,
    sources: Sequence[str],
    *,
    num_beams: int = 4,
    num_return_sequences: int = 4,
    max_source_length: int = 512,
    max_new_tokens: int = 512,
    batch_size: int = 16,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    repetition_penalty: float = 1.0,
    codec: Optional[CharCodec] = None,
    postprocess: bool = True,
    x_to_gap: bool = True,
    log_every: int = 20,
) -> List[List[Dict[str, Any]]]:
    """Beam-search ``num_return_sequences`` candidates for every source string.

    Inputs are processed longest-first (less padding, early OOM); a CUDA OOM halves the batch
    size and retries.  Returns, in input order, ``[{"text", "score", "rank"}, ...]`` per source
    where ``score`` is the length-normalised beam log-probability (``None`` for greedy).
    """
    import torch

    num_return_sequences = max(1, min(num_return_sequences, num_beams))
    device = next(model.parameters()).device
    lengths = [len(tokenizer(s, add_special_tokens=True)["input_ids"]) for s in sources]
    order = sorted(range(len(sources)), key=lambda i: -lengths[i])
    results: List[Optional[List[Dict[str, Any]]]] = [None] * len(sources)
    gen_kwargs: Dict[str, Any] = dict(
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=num_beams > 1,
    )
    if num_beams > 1:
        gen_kwargs.update(early_stopping=True, length_penalty=length_penalty)
    if no_repeat_ngram_size:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    if repetition_penalty and repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    start, done, bs, pos, n_batches = time.perf_counter(), 0, max(1, batch_size), 0, 0
    while pos < len(order):
        idx = order[pos : pos + bs]
        batch = [sources[i] for i in idx]
        try:
            enc = tokenizer(batch, max_length=max_source_length, truncation=True, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
            with torch.inference_mode():
                out = model.generate(**enc, **gen_kwargs)
        except RuntimeError as exc:  # includes torch.cuda.OutOfMemoryError
            if _is_oom(exc) and bs > 1:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                LOG.warning("CUDA OOM - retrying with batch_size=%d", bs)
                continue
            raise
        seqs = out.sequences if hasattr(out, "sequences") else out
        scores = getattr(out, "sequences_scores", None)
        texts = tokenizer.batch_decode(seqs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for j, src_index in enumerate(idx):
            cands = []
            for r in range(num_return_sequences):
                k = j * num_return_sequences + r
                text = texts[k]
                if codec is not None:
                    text = codec.decode_target(text)
                if postprocess:
                    text = postprocess_translation(text, x_to_gap=x_to_gap)
                score = float(scores[k]) if scores is not None else None
                if score is not None and not math.isfinite(score):
                    score = None
                cands.append({"text": text, "score": score, "rank": r})
            results[src_index] = cands
        pos += len(idx)
        done += len(idx)
        n_batches += 1
        if log_every and n_batches % log_every == 0:
            rate = done / max(time.perf_counter() - start, 1e-9)
            LOG.info("generated %d/%d (%.1f sent/s, batch_size=%d)", done, len(order), rate, bs)
    return [r if r is not None else [] for r in results]
