"""Utilities for expanding user queries before retrieval.

The heuristic implemented here is intentionally lightweight: it simply appends
pre-defined synonyms to the natural language question so that vector search can
match conceptually related terms (e.g. *sloop* when the user asks about
"sailing").  The synonym lists live in ``config.yaml`` so advanced users can
fine-tune the behaviour without modifying code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from app_config import ConfigNamespace


_WORD = re.compile(r"[\w'-]+")


@dataclass
class ExpandedQuery:
    original: str
    expanded: str
    added_terms: List[str]

    @property
    def changed(self) -> bool:
        return bool(self.added_terms)


def expand_query(question: str, config: ConfigNamespace) -> ExpandedQuery:
    """Expand ``question`` with synonym hints defined in ``config``."""

    expansion_cfg = getattr(config.retrieval, "query_expansion", ConfigNamespace({}))
    if not getattr(expansion_cfg, "enabled", False):
        return ExpandedQuery(question, question, [])

    synonyms = getattr(expansion_cfg, "synonyms", {}) or {}
    if not isinstance(synonyms, dict) or not synonyms:
        return ExpandedQuery(question, question, [])

    normalized_map = {
        key.lower(): [term.lower() for term in values]
        for key, values in synonyms.items()
    }

    tokens = [token.lower() for token in _WORD.findall(question)]

    added: List[str] = []
    for token in tokens:
        if token in normalized_map:
            added.extend(normalized_map[token])
        else:
            for root, related in normalized_map.items():
                if token in related:
                    added.append(root)

    deduped: List[str] = []
    seen = set(tokens)
    for term in added:
        if term not in seen and term not in deduped:
            deduped.append(term)

    max_terms = getattr(expansion_cfg, "max_expansions", None)
    if isinstance(max_terms, int) and max_terms >= 0:
        deduped = deduped[:max_terms]

    if not deduped:
        return ExpandedQuery(question, question, [])

    expanded = f"{question} " + " ".join(deduped)
    return ExpandedQuery(question, expanded, deduped)


__all__ = ["ExpandedQuery", "expand_query"]
