"""Configuration loader and helpers for logseq-chat.

This module centralises default settings and exposes a convenient helper for
loading ``config.yaml`` (or preset overrides) as a nested namespace.  Keeping
all configuration semantics in one place makes it easier to experiment with the
RAG pipeline while ensuring scripts like ``ingest.py`` and ``chat.py`` stay in
sync.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, MutableMapping, Optional

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "logseq_root": "",
    "include_dirs": ["journals", "pages"],
    "exclude_globs": ["**/.git/**", "**/.DS_Store", "**/assets/**"],
    "file_exts": [".md"],
    "runtime": {
        "request_timeout": 180,
    },
    "chunk": {
        "chunk_size": 900,
        "chunk_overlap": 120,
    },
    "retrieval": {
        "top_k": 6,
        "mmr": {"enabled": True},
        "query_expansion": {
            "enabled": True,
            "max_expansions": 6,
            "synonyms": {
                "sailing": ["sloop", "schooner", "boat"],
                "boat": ["vessel", "ship"],
            },
        },
    },
    "models": {
        "llm": {
            "name": "llama3.1",
            "temperature": 0.1,
        },
        "embedding": {
            "name": "nomic-embed-text",
        },
    },
    "storage": {
        "chroma_path": ".rag/chroma",
        "collection_name": "logseq_rag",
        "clear_before_ingest": False,
    },
    "evaluation": {
        "dataset": "evaluations/datasets/baseline.yaml",
        "configurations_file": "evaluations/configurations.yaml",
        "max_queries": None,
        "scoring": {
            "accuracy_weight": 0.35,
            "coverage_weight": 0.2,
            "relevance_weight": 0.2,
            "hallucination_weight": 0.15,
            "speed_weight": 0.1,
        },
    },
}


def _deep_merge(base: MutableMapping[str, Any], updates: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge ``updates`` into ``base`` and return ``base``.

    Lists are replaced wholesale to keep intent explicit.  Dictionaries are
    merged key-by-key.  ``base`` is mutated in-place, so callers should pass a
    copy when they need to preserve the original.
    """

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


class ConfigNamespace(dict):
    """Dict subclass that exposes attribute access for convenience."""

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - trivial proxy
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - guard for clarity
            raise AttributeError(item) from exc
        return _wrap(value)

    __setattr__ = dict.__setitem__  # type: ignore
    __delattr__ = dict.__delitem__  # type: ignore

    def to_dict(self) -> Dict[str, Any]:
        """Return a deep Python ``dict`` copy of the namespace."""

        return _unwrap(self)


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return ConfigNamespace(value)
    if isinstance(value, list):
        return [
            _wrap(item)
            for item in value
        ]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, ConfigNamespace):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(item) for item in value]
    return copy.deepcopy(value)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            "Create one by copying config.yaml.sample and adjusting the values."
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top of {path}, got {type(data)!r}")
    return data


def load_app_config(
    path: Optional[Path] = None,
    *,
    overrides: Optional[Iterable[MutableMapping[str, Any]]] = None,
) -> ConfigNamespace:
    """Load the application configuration and apply optional overrides."""

    config_path = path or Path("config.yaml")
    base = copy.deepcopy(DEFAULT_CONFIG)
    file_values = _load_yaml(config_path)
    _deep_merge(base, file_values)

    if overrides:
        for override in overrides:
            _deep_merge(base, copy.deepcopy(override))

    return ConfigNamespace(base)


def apply_overrides(base: ConfigNamespace, *overrides: MutableMapping[str, Any]) -> ConfigNamespace:
    """Return a new ``ConfigNamespace`` with overrides applied to ``base``."""

    merged = base.to_dict()
    for override in overrides:
        _deep_merge(merged, copy.deepcopy(override))
    return ConfigNamespace(merged)


__all__ = ["ConfigNamespace", "DEFAULT_CONFIG", "apply_overrides", "load_app_config"]
