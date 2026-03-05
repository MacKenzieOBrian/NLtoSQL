"""
SQL text cleanup helpers.

Turns raw model output into one executable SELECT statement by stripping
markdown fences, normalising spaced keywords, and rejecting non-SELECT output.
"""

from __future__ import annotations

import re
from typing import Optional

from .llm import extract_first_select as _extract_first_select
from .query_runner import DEFAULT_FORBIDDEN_TOKENS


def _normalize_spaced_keywords(text: str) -> str:
    """Fix tokenized keywords like 'S E L E C T' produced by some decoders."""
    keywords = [
        "select",
        "from",
        "where",
        "group",
        "by",
        "order",
        "limit",
        "join",
        "inner",
        "left",
        "right",
        "on",
        "having",
        "distinct",
    ]
    out = text or ""
    for kw in keywords:
        pattern = r"\b" + r"\s*".join(list(kw)) + r"\b"
        out = re.sub(pattern, kw.upper(), out, flags=re.I)
    return out


# extension path: guardrails are optional and excluded from primary model-only claims.
def clean_candidate_with_reason(raw: str) -> tuple[Optional[str], str]:
    """
    Extract a single safe SELECT statement.

    Returns:
      (sql, "ok") on success
      (None, reason) on rejection
    """
    if not raw or not raw.strip():
        return None, "empty"

    text = _normalize_spaced_keywords(raw)

    # Chat models often wrap output in fenced blocks; strip before extraction.
    text = re.sub(r"```(.*?)```", r"\1", text, flags=re.S).strip()

    sql = _extract_first_select(text)
    if not sql:
        return None, "no_select"

    sql = sql.strip()
    if ";" in sql:
        sql = sql.split(";", 1)[0].strip() + ";"
    else:
        sql = sql.rstrip(";") + ";"

    # DEFAULT_FORBIDDEN_TOKENS tokens have trailing spaces ("drop "); strip for matching.
    low = sql.lower()
    if any(tok.strip() in low for tok in DEFAULT_FORBIDDEN_TOKENS):
        return None, "forbidden_sql"

    return sql, "ok"
