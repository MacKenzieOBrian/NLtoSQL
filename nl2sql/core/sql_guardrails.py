"""
Lightweight SQL text guardrails for candidate cleanup.

Goal: turn raw model text into one executable SELECT statement.

References (project anchors):
- `REFERENCES.md#ref-scholak2021-picard`
- `REFERENCES.md#ref-wang2018-eg-decoding`
- `REFERENCES.md#ref-pourreza2023-dinsql`

Implementation docs:
- Python regex docs: https://docs.python.org/3/library/re.html
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

    sql = _extract_first_select(text)
    if not sql:
        return None, "no_select"

    sql = sql.strip()
    if not sql.lower().startswith("select"):
        return None, "no_select"
    if " from " not in f" {sql.lower()} ":
        return None, "no_from"

    # keep only the first statement.
    if ";" in sql:
        sql = sql.split(";", 1)[0].strip() + ";"
    else:
        sql = sql.rstrip(";") + ";"

    low = sql.lower()
    if any(tok in low for tok in _FORBIDDEN_SQL):
        return None, "forbidden_sql"

    return sql, "ok"
