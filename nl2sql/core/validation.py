"""
Validation helpers for generated SQL.

Parses schema text from the prompt context and checks that table names,
column names, and basic SQL shape constraints are satisfied before execution.
"""

from __future__ import annotations

from typing import Optional

import re

from .sql_guardrails import clean_candidate_with_reason


def parse_schema_text(schema_text: str) -> tuple[set[str], dict[str, set[str]]]:
    tables: set[str] = set()
    table_cols: dict[str, set[str]] = {}
    if not schema_text:
        return tables, table_cols
    for line in schema_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(?is)^([a-zA-Z_][\w$]*)\s*\((.*)\)\s*$", line)
        if not m:
            continue
        table = m.group(1).strip().lower()
        cols_raw = m.group(2)
        cols = [c.strip().lower() for c in cols_raw.split(",") if c.strip()]
        tables.add(table)
        table_cols[table] = set(cols)
    return tables, table_cols


_SELECT_CLAUSE_RE = re.compile(r"(?is)\bselect\b(.*?)\bfrom\b")
_AGG_EXPR_RE = re.compile(r"(?is)\b(count|sum|avg|min|max)\s*\(")
_SELECT_STAR_RE = re.compile(r"(?is)\bselect\s+([a-zA-Z_][\w$]*\.)?\*")
_SELECT_STAR_ALLOW_RE = re.compile(
    r"\b(all columns|all fields|full details|full row|entire row|all details|every column)\b",
    re.IGNORECASE,
)


def _extract_select_clause(sql_low: str) -> str:
    m = _SELECT_CLAUSE_RE.search(sql_low or "")
    return m.group(1) if m else ""


def _select_has_star(sql_low: str) -> bool:
    return re.search(r"\bselect\s+([a-zA-Z_][\w$]*\.)?\*\b", sql_low or "") is not None


def _select_has_field(select_clause: str, field: str) -> bool:
    if not select_clause or not field:
        return False
    return re.search(rf"\b{re.escape(field.lower())}\b", select_clause) is not None


def _split_select_expressions(select_clause: str) -> list[str]:
    """Split a SELECT clause on commas, respecting parentheses depth."""
    parts, curr, depth = [], [], 0
    for ch in (select_clause or ""):
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        if ch == "," and depth == 0:
            piece = "".join(curr).strip()
            if piece:
                parts.append(piece)
            curr = []
        else:
            curr.append(ch)
    tail = "".join(curr).strip()
    if tail:
        parts.append(tail)
    return parts


def _is_agg_expression(expr: str) -> bool:
    return bool(_AGG_EXPR_RE.search(expr or ""))


def schema_validate(
    *,
    sql: str,
    schema_index: tuple[set[str], dict[str, set[str]]],
) -> tuple[bool, str, dict]:
    """Check that every table name in FROM/JOIN exists in the schema."""
    tables, _ = schema_index
    if not tables:
        return True, "no_schema", {}

    sql_low = (sql or "").lower()
    for m in re.finditer(r"(?is)\b(from|join)\s+([a-zA-Z_][\w$]*)", sql_low):
        table = m.group(2)
        after = sql_low[m.end(): m.end() + 1]
        if after == "(" or table in tables:
            continue
        return False, f"unknown_table:{table}", {}

    return True, "ok", {}


def validate_sql(
    sql: str,
    schema_text: Optional[str] = None,
    *,
    nlq: Optional[str] = None,
) -> dict:
    """Validate SQL formatting + schema references without executing."""
    # catch obvious formatting/schema errors before hitting the database.
    if not sql or not sql.strip():
        return {"valid": False, "reason": "empty_sql"}

    cleaned, reason = clean_candidate_with_reason(sql)
    if not cleaned:
        return {"valid": False, "reason": f"clean_reject:{reason}"}

    if _SELECT_STAR_RE.search(cleaned) and not _SELECT_STAR_ALLOW_RE.search(nlq or ""):
        return {"valid": False, "reason": "select_star_forbidden"}

    if not schema_text:
        return {"valid": False, "reason": "schema_missing"}

    tables, table_cols = parse_schema_text(schema_text)
    if not tables:
        return {"valid": False, "reason": "schema_missing"}

    ok, why, detail = schema_validate(
        sql=cleaned,
        schema_index=(tables, table_cols),
    )
    if not ok:
        out = {"valid": False, "reason": why}
        if detail:
            out.update(detail)
        return out

    return {"valid": True, "reason": "schema_ok"}


def validate_constraints(sql: str, constraints: Optional[dict], *, schema_text: Optional[str] = None) -> dict:
    """Validate SQL against a very small set of optional shape hints."""
    if not constraints:
        return {"valid": True, "reason": "no_constraints"}
    if not isinstance(constraints, dict):
        return {"valid": True, "reason": "no_constraints"}
    if not sql or not sql.strip():
        return {"valid": False, "reason": "empty_sql"}

    sql_low = sql.lower()
    select_clause = _extract_select_clause(sql_low)
    has_select_star = _select_has_star(sql_low)

    agg = constraints.get("agg")
    if constraints.get("distinct") and "select distinct" not in sql_low:
        return {"valid": False, "reason": "missing_distinct"}

    if agg:
        agg_low = agg.lower()
        has_agg = re.search(rf"\b{re.escape(agg_low)}\s*\(", sql_low) is not None
        if not has_agg and agg in {"MAX", "MIN"}:
            has_agg = "order by" in sql_low and re.search(r"\blimit\s+1\b", sql_low) is not None
        if not has_agg:
            return {"valid": False, "reason": f"missing_agg:{agg}"}

    if constraints.get("needs_group_by") and "group by" not in sql_low:
        return {"valid": False, "reason": "missing_group_by"}
    if constraints.get("needs_group_by") and not has_select_star:
        exprs = _split_select_expressions(select_clause)
        non_agg_exprs = [e for e in exprs if not _is_agg_expression(e)]
        if not non_agg_exprs:
            return {"valid": False, "reason": "missing_group_dimension_projection"}

    if constraints.get("needs_order_by") and "order by" not in sql_low:
        return {"valid": False, "reason": "missing_order_by"}

    limit = constraints.get("limit")
    if limit is not None:
        if re.search(rf"\blimit\s+{limit}\b", sql_low) is None:
            return {"valid": False, "reason": f"missing_limit:{limit}"}

    value_hints = constraints.get("value_hints") or []
    if value_hints and not any(v in sql_low for v in value_hints):
        return {"valid": False, "reason": "missing_value_hint"}

    required_output_fields = constraints.get("required_output_fields") or []
    if required_output_fields and not has_select_star:
        missing_required = [f for f in required_output_fields if not _select_has_field(select_clause, f)]
        if missing_required:
            return {
                "valid": False,
                "reason": "missing_required_output_field",
                "missing_fields": missing_required,
            }
    if constraints.get("strict_required_output_fields") and required_output_fields and not has_select_star:
        exprs = _split_select_expressions(select_clause)
        unexpected = [
            expr for expr in exprs
            if not any(_select_has_field(expr, f) for f in required_output_fields)
        ]
        if unexpected:
            return {
                "valid": False,
                "reason": "unexpected_output_field",
                "unexpected_fields": unexpected[:3],
            }

    return {"valid": True, "reason": "ok"}
