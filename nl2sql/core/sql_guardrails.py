"""
SQL text cleanup helpers.

Turns raw model output into one executable SELECT statement by stripping
markdown fences, normalising spaced keywords, and rejecting non-SELECT output.
"""

from __future__ import annotations

import re
from typing import Optional


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


_FORBIDDEN_SQL = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "truncate",
    "create",
    "grant",
    "revoke",
)


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

    # remove markdown fences commonly returned by chat models.
    text = text.replace("```sql", "```").replace("```json", "```")
    text = re.sub(r"```(.*?)```", r"\1", text, flags=re.S).strip()

    # reuse shared extraction helper so behavior matches other code paths.
    from .llm import extract_first_select as _extract_first_select

    # extract_first_select guarantees SELECT…FROM structure; None means no valid SQL found.
    sql = _extract_first_select(text)
    if not sql:
        return None, "no_select"

    sql = sql.strip()
    # keep only the first statement.
    if ";" in sql:
        sql = sql.split(";", 1)[0].strip() + ";"
    else:
        sql = sql.rstrip(";") + ";"

    low = sql.lower()
    if any(tok in low for tok in _FORBIDDEN_SQL):
        return None, "forbidden_sql"

    return sql, "ok"
