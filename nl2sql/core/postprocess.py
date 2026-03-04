"""
Small SQL cleanup helpers used by eval and notebook demos.

1) Keep the first SELECT statement
2) Drop ORDER BY / LIMIT when the NLQ does not ask for ranking
3) Normalize SQL text for EM comparison
"""

from __future__ import annotations

import re


SQL_RE = re.compile(r"(?is)\bselect\b.*?(;|\Z)")
RANKING_HINT_RE = re.compile(
    r"\b(top|highest|lowest|most|least|largest|smallest|first|last|max|min|order|sort|rank)\b",
    re.IGNORECASE,
)


def normalize_sql(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(";")
    return s.lower()


def first_select_only(text: str) -> str:
    """Return the first SELECT statement if one exists."""
    m = SQL_RE.search((text or "").strip())
    if not m:
        return text
    sql = m.group(0).strip()
    if sql and not sql.endswith(";"):
        sql += ";"
    return sql


def _strip_order_by_limit(sql: str, nlq: str) -> str:
    if RANKING_HINT_RE.search(nlq or ""):
        return sql
    out = re.sub(r"(?is)\sorder\s+by\s+[^;]+", "", sql)
    out = re.sub(r"(?is)\slimit\s+\d+\s*", "", out)
    return out


def guarded_postprocess(sql: str, nlq: str) -> str:
    """Apply conservative post-processing: extract first SELECT, strip ranking clauses."""
    current = first_select_only(sql or "")
    current = _strip_order_by_limit(current, nlq)
    return current
