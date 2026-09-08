"""Canonical logical-query formatting for evaluation protocol clean_v2."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


QUERY_FORMATTER_VERSION = "clean_v2"
_WHITESPACE_RE = re.compile(r"\s+")
_DUPLICATE_TODAY_RE = re.compile(
    r"^(Today is\s+[^.]+\.\s*)Today is\s+[^.]+\.\s*", re.IGNORECASE
)
_MALFORMED_TODAY_RE = re.compile(r"^Today is\s*\.\s*", re.IGNORECASE)


def _get(record: Any, key: str, default: Any = "") -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def normalize_stored_question(question: Any) -> str:
    """Normalize transport whitespace without rewriting the question body."""
    if question is None:
        return ""
    return _WHITESPACE_RE.sub(" ", str(question)).strip()


def format_query(record: Any) -> str:
    """Return the one logical query used by training and every evaluation stage.

    Clean v2 treats the stored question as authoritative. StreamingQA already
    contains its natural-language temporal prefix, while TimeQA has no query
    timestamp. No generic timestamp concatenation is permitted.
    """
    question = _get(record, "question", "")
    if not question:
        question = _get(record, "query", "")
    return normalize_stored_question(question)


def validate_formatted_query(query: str) -> None:
    """Raise when a query contains a known legacy timestamp corruption."""
    if _MALFORMED_TODAY_RE.search(query):
        raise ValueError("Malformed temporal prefix: 'Today is .'")
    if _DUPLICATE_TODAY_RE.search(query):
        raise ValueError("Duplicate 'Today is' temporal prefix")


def format_and_validate_query(record: Any) -> str:
    query = format_query(record)
    validate_formatted_query(query)
    return query

