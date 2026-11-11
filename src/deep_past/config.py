"""YAML/JSON configuration loading with ``defaults:`` inheritance and CLI overrides.

    cfg = load_config("configs/train/helsinki_final.yaml", ["lr=3e-4", "data.max_rows=100"])

* ``defaults: base.yaml`` (or a list) is resolved relative to the file and deep-merged first;
* every override ``dotted.key=value`` is parsed with YAML (``3e-4`` -> float, ``[5]`` -> list,
  ``null`` -> None) and assigned into the nested dict;
* string values are expanded for ``~`` and ``${ENV_VARS}``.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, Iterable, Optional

import yaml

__all__ = ["load_config", "deep_merge", "apply_overrides", "add_config_args"]


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (lists are replaced)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        if path.endswith(".json"):
            data = json.load(fh)
        else:
            data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must be a mapping, got {type(data).__name__}")
    return data


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value)) if ("$" in value or value.startswith("~")) else value
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Apply ``a.b.c=value`` overrides (values parsed as YAML scalars/lists/dicts)."""
    out = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
    return out


def load_config(path: str, overrides: Optional[Iterable[str]] = None, _seen: Optional[set] = None) -> Dict[str, Any]:
    """Load a config file, resolving ``defaults`` (inheritance) and applying overrides."""
    path = os.path.abspath(path)
    seen = set(_seen or ())
    if path in seen:
        raise ValueError(f"circular config defaults at {path}")
    seen.add(path)
    data = _read(path)
    parents = data.pop("defaults", None) or []
    if isinstance(parents, str):
        parents = [parents]
    merged: Dict[str, Any] = {}
    for parent in parents:
        parent_path = parent if os.path.isabs(parent) else os.path.join(os.path.dirname(path), parent)
        merged = deep_merge(merged, load_config(parent_path, None, seen))
    merged = deep_merge(merged, data)
    merged = apply_overrides(merged, overrides)
    return _expand(merged)


def add_config_args(parser) -> None:
    """Add the standard ``--config`` / ``--set key=value`` arguments to an argparse parser."""
    parser.add_argument("--config", required=True, help="YAML/JSON config file")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="override a config value (dotted keys, YAML values); repeatable",
    )
