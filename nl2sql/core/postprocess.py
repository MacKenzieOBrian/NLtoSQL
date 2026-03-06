"""
Small SQL cleanup helpers used by eval and notebook demos.

1) Keep the first SELECT statement
2) Drop ORDER BY / LIMIT when the NLQ does not ask for ranking
3) Normalize SQL text for EM comparison
"""

from __future__ import annotations

import re

from .llm import extract_first_select as _extract_first_select

# "first"/"last" omitted — both appear in ClassicModels column names (contactFirstName etc.).
RANKING_HINT_RE = re.compile(
    r"\b(top|highest|lowest|most|least|largest|smallest|max|min|order|sort|rank)\b",
    re.IGNORECASE,
)


def normalize_sql(s: str) -> str:
    """Normalize SQL for exact-match string comparison."""
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(";")
    return s.lower()


def first_select_only(text: str) -> str:
    """Return the first SELECT found, or original text as fallback."""
    result = _extract_first_select(text or "")
    return result if result is not None else (text or "")


def _strip_order_by_limit(sql: str, nlq: str) -> str:
    """Remove ORDER BY/LIMIT unless NLQ requests ranking — prevents spurious EX failures [22]."""
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
