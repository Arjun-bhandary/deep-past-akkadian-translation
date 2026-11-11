"""Small I/O, logging and environment helpers shared by every stage."""
from __future__ import annotations

import contextlib
import glob
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

__all__ = [
    "setup_logging",
    "read_table",
    "write_table",
    "read_jsonl",
    "write_jsonl",
    "find_column",
    "seed_everything",
    "is_kaggle",
    "resolve_path",
    "ensure_dir",
    "timer",
]

PathLike = Union[str, os.PathLike]
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", log_file: Optional[PathLike] = None) -> logging.Logger:
    """Configure root logging once (stdout + optional file) and return the package logger."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    fmt = logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S")
    if not any(getattr(h, "_deep_past", False) for h in root.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh._deep_past = True  # type: ignore[attr-defined]
        root.addHandler(sh)
    if log_file is not None:
        ensure_dir(Path(log_file).parent)
        target = os.path.abspath(log_file)
        if not any(getattr(h, "baseFilename", None) == target for h in root.handlers):
            fh = logging.FileHandler(target, encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)
    return logging.getLogger("deep_past")


def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_table(path: PathLike, **kwargs: Any):
    """Read ``.csv/.tsv/.parquet/.jsonl/.json`` into a DataFrame (strings kept as-is)."""
    import pandas as pd

    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".gz"):
        return pd.read_csv(path, keep_default_na=True, **kwargs)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t", **kwargs)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path, **kwargs)
    if ext == ".jsonl":
        return pd.read_json(path, lines=True, **kwargs)
    if ext == ".json":
        return pd.read_json(path, **kwargs)
    raise ValueError(f"unsupported table format: {path}")


def write_table(df, path: PathLike) -> str:
    path = str(path)
    ensure_dir(Path(path).parent)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext == ".tsv":
        df.to_csv(path, sep="\t", index=False)
    elif ext in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    elif ext == ".jsonl":
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"unsupported table format: {path}")
    return path


def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(records: Iterable[Dict[str, Any]], path: PathLike) -> str:
    path = str(path)
    ensure_dir(Path(path).parent)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)  # atomic: readers never see half-written candidate files
    return path


def find_column(columns_or_df: Any, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    """Return the first column whose name matches one of ``candidates`` (case-insensitive)."""
    columns = list(getattr(columns_or_df, "columns", columns_or_df))
    lower = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise KeyError(f"none of {list(candidates)} found in columns {columns}")
    return None


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def is_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.isdir("/kaggle/input")


def resolve_path(path: Optional[PathLike]) -> Optional[str]:
    """Expand ``~``/``$VARS`` and glob patterns (``/kaggle/input/*/final-byt5``).

    Returns the first sorted glob match, or the expanded string unchanged if nothing matches
    (so callers can still pass Hugging Face Hub ids such as ``google/byt5-base``).
    """
    if path is None:
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    if any(ch in expanded for ch in "*?["):
        matches = sorted(glob.glob(expanded))
        if matches:
            return matches[0]
    return expanded


@contextlib.contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None) -> Iterator[Dict[str, float]]:
    """``with timer("generate") as t: ...`` -> logs and stores elapsed seconds in ``t["s"]``."""
    out: Dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield out
    finally:
        out["s"] = time.perf_counter() - start
        (logger or logging.getLogger("deep_past")).info("%s took %.1fs", name, out["s"])
