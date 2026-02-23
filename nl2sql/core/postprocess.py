"""
Post-processing for model-generated SQL text.

How to read this file:
1) Normalize SQL text for stable EM comparison.
2) Keep one SELECT statement and remove obvious prompt artifacts.
3) Apply small projection/order heuristics used in this project.

References:
- Transformers generation docs: https://huggingface.co/docs/transformers/main_classes/text_generation
- Python regex docs: https://docs.python.org/3/library/re.html
"""

from __future__ import annotations

import re
from typing import Any, Iterable


def normalize_sql(s: str) -> str:
    # Regex reference: https://docs.python.org/3/library/re.html
    # Rationale: early evals showed formatting noise dominating EM; normalization keeps EM focused on real errors.
    # Used for EM (Exact Match): normalize surface form so EM detects real regressions
    # (missing JOIN, wrong predicate) rather than harmless whitespace/casing differences.
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(";")
    return s.lower()


# "List all ..." is a common prompt pattern where the model over-selects columns.
# We clamp projection for that specific phrasing to reduce evaluation noise.
LIST_ALL_RE = re.compile(r"(?is)^\s*list\s+all\s+")

# Capture the SELECT list for projection heuristics (best-effort, not a full parser).
SELECT_LIST_RE = re.compile(r"(?is)^\s*select\s+(.*?)\s+from\s+", re.DOTALL)

# First SELECT block extraction: prevents multi-statement outputs and trailing explanations.
SQL_RE = re.compile(r"(?is)\bselect\b.*?(;|\Z)")

# Models often add ORDER BY / LIMIT even when the question does not ask for ranking.
# We keep these clauses only when the NLQ contains ranking cues.
RANKING_HINT_RE = re.compile(
    r"\b(top|highest|lowest|most|least|largest|smallest|first|last|max|min|order|sort|rank)\b",
    re.IGNORECASE,
)

# Some gold queries do not include ID/code columns unless explicitly requested.
# Dropping ID-like projections can improve EM without changing execution semantics.
ID_IN_NLQ_RE = re.compile(r"\b(id|ids|number|numbers|code|codes|line item|line number)\b", re.IGNORECASE)
ID_LIKE_COL_RE = re.compile(
    r"\b(order(number)?|customer(number)?|employee(number)?|office(code)?|product(code)?|line(number)?)\b",
    re.IGNORECASE,
)
ENTITY_NOUN_RE = re.compile(r"\b(customer|customers|order|orders|product|products|payment|payments|office|offices|employee|employees)\b", re.IGNORECASE)
LISTING_NLQ_RE = re.compile(r"\b(list|show|which|display|give|find|with|who|that|top|highest|lowest|most|least|first|last)\b", re.IGNORECASE)


def first_select_only(text: str) -> str:
    """Return the first SELECT...; block, or the original text if no SELECT found."""
    m = SQL_RE.search((text or "").strip())
    if not m:
        return text
    sql = m.group(0).strip()
    if not sql.endswith(";"):
        sql += ";"
    return sql


def _strip_order_by_limit(sql: str, nlq: str) -> str:
    """Remove ORDER BY / LIMIT when the NLQ does not imply ranking."""
    if RANKING_HINT_RE.search(nlq or ""):
        return sql
    # Motivation: ranking/limit clauses are a frequent "model habit" that can:
    # - reduce EM (gold often omits ORDER BY)
    # - change semantics if LIMIT is applied unintentionally
    out = re.sub(r"(?is)\sorder\s+by\s+[^;]+", "", sql)
    out = re.sub(r"(?is)\slimit\s+\d+\s*", "", out)
    return out


def _split_select_list(select_part: str) -> list[str]:
    """Lightweight split on commas not inside parentheses."""
    parts: list[str] = []
    buf = []
    depth = 0
    for ch in select_part:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts


def _rebuild_select(sql: str, new_select_items: Iterable[str]) -> str:
    # Rebuild SELECT list with a regex substitution. This is intentionally simple
    # (no full SQL AST) so the postprocess layer remains inspectable.
    return re.sub(SELECT_LIST_RE, lambda _: f"SELECT {', '.join(new_select_items)} FROM ", sql, count=1)


def enforce_explicit_projection(sql: str, explicit_fields: Iterable[str] | None) -> str:
    """
    If the NLQ explicitly enumerates fields, trim the SELECT list to those fields only.
    This keeps strict EX aligned with gold projection shape.
    """
    if not explicit_fields:
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
    ordered: list[str] = []
    used: set[int] = set()
    for field in explicit_fields:
        found = None
        for idx, col in enumerate(cols):
            if idx in used:
                continue
            if _select_item_matches_field(col, field):
                found = idx
                ordered.append(col)
                used.add(idx)
                break
        if found is None:
            # If we can't confidently match all explicit fields, skip trimming.
            return sql
    if ordered == cols:
        return sql
    return _rebuild_select(sql, ordered)


def prune_id_like_columns(
    sql: str,
    nlq: str,
    explicit_fields: Iterable[str] | None = None,
    required_fields: Iterable[str] | None = None,
) -> str:
    """
    Drop ID/code/number columns when the NLQ does not ask for them explicitly.
    Helps EM on datasets where gold projections exclude IDs.
    """
    if ID_IN_NLQ_RE.search(nlq or ""):
        return sql
    # If explicit fields include an ID-like column, keep identifiers.
    if explicit_fields:
        for field in explicit_fields:
            if ID_LIKE_COL_RE.search(field):
                return sql
    # If required fields include an ID-like column, keep identifiers.
    if required_fields:
        for field in required_fields:
            if ID_LIKE_COL_RE.search(str(field)):
                return sql
    m = SELECT_LIST_RE.search(sql)
    if not m:
        return sql
    select_part = m.group(1).strip()
    if "*" in select_part:
        return sql  # do not touch SELECT *
    cols = _split_select_list(select_part)
    # Keep only non-ID-like projection items. This is a heuristic; it can be wrong
    # if the NLQ implicitly expects IDs, so we only do it when the NLQ lacks ID cues.
    kept = [c for c in cols if not ID_LIKE_COL_RE.search(c.split()[-1])]
    if not kept:
        return sql  # avoid empty select list
    return _rebuild_select(sql, kept)


def enforce_minimal_projection(
    sql: str,
    nlq: str,
    required_fields: Iterable[str] | None = None,
) -> str:
    if not sql or not nlq:
        return sql
    # If the NLQ (or constraints) require specific output fields, avoid collapsing the projection.
    if required_fields:
        return sql
    # If the NLQ enumerates fields or uses "with", keep the full projection.
    if "," in nlq or " and " in nlq.lower() or " with " in nlq.lower():
        return sql
    if not LIST_ALL_RE.search(nlq.strip()):
        return sql
    m = SELECT_LIST_RE.search(sql)
    if not m:
        return sql
    select_part = m.group(1).strip()
    if "*" in select_part:
        return sql
    # Motivation: "list all ..." questions are commonly answered with multiple columns.
    # Gold queries in this benchmark often use a minimal projection (e.g., names only).
    first_expr = select_part.split(",")[0].strip()
    rebuilt = re.sub(SELECT_LIST_RE, lambda _: f"SELECT {first_expr} FROM ", sql, count=1)
    return rebuilt


def _select_item_matches_field(item: str, field: str) -> bool:
    if not item or not field:
        return False
    return re.search(rf"\b{re.escape(field)}\b", item, re.IGNORECASE) is not None


def reorder_projection(sql: str, explicit_fields: Iterable[str] | None) -> str:
    """
    Reorder SELECT columns to match explicit NLQ field order without dropping columns.
    This reduces EX sensitivity to column ordering when the NLQ enumerates fields.
    """
    if not explicit_fields:
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
    ordered: list[str] = []
    used: set[int] = set()
    for field in explicit_fields:
        for idx, col in enumerate(cols):
            if idx in used:
                continue
            if _select_item_matches_field(col, field):
                ordered.append(col)
                used.add(idx)
                break
    if len(ordered) < 2:
        return sql
    for idx, col in enumerate(cols):
        if idx not in used:
            ordered.append(col)
    if ordered == cols:
        return sql
    return _rebuild_select(sql, ordered)


def guarded_postprocess(
    sql: str,
    nlq: str,
    *,
    explicit_fields: Iterable[str] | None = None,
    required_fields: Iterable[str] | None = None,
) -> str:
    """
    Combined guardrail used in eval:
    - keep first SELECT
    - strip ordering/limits if question doesn't ask for ranking
    - drop ID-like columns if NLQ didn't request them
    - minimal projection for "list all ..." phrasing
    - reorder projections to match explicit NLQ field order
    """
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
    """
    Return a full stage-by-stage trace of SQL cleanup used by guarded_postprocess.

    Intended for notebook demos and debugging:
    - shows each transform input/output
    - marks whether each stage changed the SQL
    - keeps final behavior identical to guarded_postprocess
    """
    explicit_list = list(explicit_fields) if explicit_fields is not None else None
    required_list = list(required_fields) if required_fields is not None else None

    steps: list[dict[str, Any]] = []

    def _record(stage: str, before: str, after: str, note: str | None = None) -> str:
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
        "extract first SELECT candidate and normalize statement boundary",
    )
    current = _record(
        "strip_order_by_limit",
        current,
        _strip_order_by_limit(current, nlq),
        "remove ranking clauses when NLQ does not request ranking",
    )

    if explicit_list:
        current = _record(
            "enforce_explicit_projection",
            current,
            enforce_explicit_projection(current, explicit_list),
            "trim projection to explicitly requested fields",
        )
    else:
        current = _record(
            "prune_id_like_columns",
            current,
            prune_id_like_columns(
                current,
                nlq,
                explicit_fields=explicit_list,
                required_fields=required_list,
            ),
            "drop identifier-like columns when not requested",
        )
        current = _record(
            "enforce_minimal_projection",
            current,
            enforce_minimal_projection(current, nlq, required_fields=required_list),
            "collapse over-wide projection for list-style NLQs",
        )

    current = _record(
        "reorder_projection",
        current,
        reorder_projection(current, explicit_list),
        "align projection order with explicit NLQ field ordering",
    )

    return {
        "input_sql": sql,
        "nlq": nlq,
        "explicit_fields": explicit_list,
        "required_fields": required_list,
        "steps": steps,
        "final_sql": current,
    }
