"""Small SQL clean-up helpers used before scoring."""

from __future__ import annotations

import re

from .llm import extract_first_select as _extract_first_select

# "first"/"last" omitted — both appear in ClassicModels column names (contactFirstName etc.).
# ai note copilot: "regex for ranking keyword detection in NLQ"
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


# ai note copilot: "regex substitution to strip ORDER BY/LIMIT conditionally"
def _strip_order_by_limit(sql: str, nlq: str) -> str:
    """Remove ORDER BY/LIMIT unless the question clearly asks for ranking."""
    if RANKING_HINT_RE.search(nlq or ""):
        return sql
    out = re.sub(r"(?is)\sorder\s+by\s+[^;]+", "", sql)
    out = re.sub(r"(?is)\slimit\s+\d+\s*", "", out)
    return out

# Keep this conservative: clean obvious formatting noise, but do not guess a new query.
def guarded_postprocess(sql: str, nlq: str) -> str:
    """Apply conservative post-processing: extract first SELECT, strip ranking clauses."""
    current = first_select_only(sql or "")
    current = _strip_order_by_limit(current, nlq)
    return current
