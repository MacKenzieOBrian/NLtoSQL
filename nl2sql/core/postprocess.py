"""
Small SQL cleanup helpers used by eval and notebook demos.

This module stays intentionally simple:
1) keep the first SELECT statement
2) drop ORDER BY / LIMIT when the NLQ does not ask for ranking
3) optionally trim projection to a provided field list
4) normalize SQL text for EM comparison
"""

from __future__ import annotations

import re
from typing import Any, Iterable


SELECT_LIST_RE = re.compile(r"(?is)^\s*select\s+(.*?)\s+from\s+", re.DOTALL)
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


def _split_select_list(select_part: str) -> list[str]:
    parts: list[str] = []
    curr: list[str] = []
    depth = 0
    for ch in select_part:
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        if ch == "," and depth == 0:
            piece = "".join(curr).strip()
            if piece:
                parts.append(piece)
            curr = []
            continue
        curr.append(ch)
    tail = "".join(curr).strip()
    if tail:
        parts.append(tail)
    return parts


def _rebuild_select(sql: str, new_items: Iterable[str]) -> str:
    return re.sub(SELECT_LIST_RE, lambda _: f"SELECT {', '.join(new_items)} FROM ", sql, count=1)


def _select_item_matches_field(item: str, field: str) -> bool:
    if not item or not field:
        return False
    return re.search(rf"\b{re.escape(field)}\b", item, re.IGNORECASE) is not None


def _trim_projection(sql: str, fields: Iterable[str] | None) -> str:
    if not fields:
        return sql
    m = SELECT_LIST_RE.search(sql or "")
    if not m:
        return sql
    select_part = m.group(1).strip()
    if "*" in select_part:
        return sql
    cols = _split_select_list(select_part)
    if len(cols) < 2:
        return sql

    trimmed: list[str] = []
    used: set[int] = set()
    for field in fields:
        found = False
        for idx, col in enumerate(cols):
            if idx in used:
                continue
            if _select_item_matches_field(col, str(field)):
                trimmed.append(col)
                used.add(idx)
                found = True
                break
        if not found:
            return sql

    if not trimmed or trimmed == cols:
        return sql
    return _rebuild_select(sql, trimmed)


def guarded_postprocess(
    sql: str,
    nlq: str,
    *,
    explicit_fields: Iterable[str] | None = None,
    required_fields: Iterable[str] | None = None,
) -> str:
    return debug_guarded_postprocess(
        sql=sql,
        nlq=nlq,
        explicit_fields=explicit_fields,
        required_fields=required_fields,
    )["final_sql"]


def debug_guarded_postprocess(
    sql: str,
    nlq: str,
    *,
    explicit_fields: Iterable[str] | None = None,
    required_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    explicit_list = list(explicit_fields) if explicit_fields is not None else None
    required_list = list(required_fields) if required_fields is not None else None
    target_fields = explicit_list or required_list

    steps: list[dict[str, Any]] = []

    def _record(stage: str, before: str, after: str, note: str) -> str:
        steps.append(
            {
                "stage": stage,
                "before": before,
                "after": after,
                "changed": before != after,
                "note": note,
            }
        )
        return after

    current = sql or ""
    current = _record(
        "first_select_only",
        current,
        first_select_only(current),
        "extract the first SQL SELECT statement",
    )
    current = _record(
        "strip_order_by_limit",
        current,
        _strip_order_by_limit(current, nlq),
        "remove ranking clauses unless the question asks for ranking",
    )
    current = _record(
        "trim_projection",
        current,
        _trim_projection(current, target_fields),
        "trim the SELECT list to the provided field hints when possible",
    )

    return {
        "input_sql": sql,
        "nlq": nlq,
        "explicit_fields": explicit_list,
        "required_fields": required_list,
        "steps": steps,
        "final_sql": current,
    }
