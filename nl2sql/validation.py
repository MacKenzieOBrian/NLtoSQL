"""
Shared validation utilities used by both the tool-driven and class-based agents.
Centralizes schema parsing, join-key validation, and constraint checks.
"""

from __future__ import annotations

from typing import Optional

import re

from .agent_schema_linking import _tables_connected, validate_join_hints
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


def _extract_select_clause(sql_low: str) -> str:
    if not sql_low:
        return ""
    m = _SELECT_CLAUSE_RE.search(sql_low)
    if not m:
        return ""
    return m.group(1)


def _select_has_star(sql_low: str) -> bool:
    if not sql_low:
        return False
    return re.search(r"\bselect\s+([a-zA-Z_][\w$]*\.)?\*\b", sql_low) is not None


def _select_has_field(select_clause: str, field: str) -> bool:
    if not select_clause or not field:
        return False
    return re.search(rf"\b{re.escape(field.lower())}\b", select_clause) is not None


def _split_select_expressions(select_clause: str) -> list[str]:
    if not select_clause:
        return []
    parts: list[str] = []
    curr: list[str] = []
    depth = 0
    for ch in select_clause:
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


def _is_agg_expression(expr: str) -> bool:
    if not expr:
        return False
    return _AGG_EXPR_RE.search(expr) is not None


def _tables_in_query(sql_low: str) -> set[str]:
    tables: set[str] = set()
    if not sql_low:
        return tables
    for m in re.finditer(r"(?is)\b(from|join)\s+([a-zA-Z_][\w$]*)", sql_low):
        table = m.group(2)
        after = sql_low[m.end() : m.end() + 1]
        if after == "(":
            continue
        tables.add(table)
    return tables


def schema_validate(
    *,
    sql: str,
    schema_index: tuple[set[str], dict[str, set[str]]],
    enforce_join_hints: bool = True,
) -> tuple[bool, str, dict]:
    tables, table_cols = schema_index
    if not tables:
        return True, "no_schema", {}

    sql_low = (sql or "").lower()

    # Validate explicit table names in FROM/JOIN (skip subqueries).
    for m in re.finditer(r"(?is)\b(from|join)\s+([a-zA-Z_][\w$]*)", sql_low):
        table = m.group(2)
        after = sql_low[m.end() : m.end() + 1]
        if after == "(":
            continue
        if table not in tables:
            return False, f"unknown_table:{table}", {}

    # Validate qualified columns table.column when table is known.
    for m in re.finditer(r"(?is)\b([a-zA-Z_][\w$]*)\.([a-zA-Z_][\w$]*)\b", sql_low):
        table = m.group(1)
        col = m.group(2)
        if table in table_cols and col not in table_cols[table]:
            return False, f"unknown_column:{table}.{col}", {}

    if enforce_join_hints:
        ok_join, why_join, detail = validate_join_hints(sql)
        if not ok_join:
            return False, why_join, detail

    return True, "ok", {}


def validate_sql(sql: str, schema_text: Optional[str] = None, *, enforce_join_hints: bool = True) -> dict:
    """Validate SQL formatting + schema references without executing."""
    # Rationale: catch obvious formatting/schema errors before hitting the database.
    if not sql or not sql.strip():
        return {"valid": False, "reason": "empty_sql"}

    cleaned, reason = clean_candidate_with_reason(sql)
    if not cleaned:
        return {"valid": False, "reason": f"clean_reject:{reason}"}

    if not schema_text:
        return {"valid": False, "reason": "schema_missing"}

    tables, table_cols = parse_schema_text(schema_text)
    if not tables:
        return {"valid": False, "reason": "schema_missing"}

    ok, why, detail = schema_validate(
        sql=cleaned,
        schema_index=(tables, table_cols),
        enforce_join_hints=enforce_join_hints,
    )
    if not ok:
        out = {"valid": False, "reason": why}
        if detail:
            out.update(detail)
        return out

    return {"valid": True, "reason": "schema_ok"}


def validate_constraints(sql: str, constraints: Optional[dict], *, schema_text: Optional[str] = None) -> dict:
    """Validate SQL structure against extracted constraints."""
    # Rationale: prevents "runs but wrong shape" (e.g., missing GROUP BY or LIMIT).
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

    # Entity identifiers are treated as soft hints at generation/rerank time.
    # Strict projection checks are handled via required_output_fields above.

    if constraints.get("needs_location"):
        tables_in_query = _tables_in_query(sql_low)
        location_tables = set(constraints.get("location_tables") or [])
        if location_tables and not (tables_in_query & location_tables):
            return {
                "valid": False,
                "reason": "missing_location_table",
                "location_tables": sorted(location_tables),
            }
        if re.search(r"\b(city|country|state|territory|region)\b", sql_low) is None:
            return {"valid": False, "reason": "missing_location_column"}

    value_columns = constraints.get("value_columns") or []
    if constraints.get("needs_location") and value_columns:
        location_cols = {"city", "country", "state", "territory", "region"}
        if all(str(c).lower() in location_cols for c in value_columns):
            value_columns = []
    if schema_text and value_columns:
        _, table_cols = parse_schema_text(schema_text)
        tables_in_query = _tables_in_query(sql_low)
        present_value_cols: list[str] = []
        value_col_tables: dict[str, list[str]] = {}
        for col in value_columns:
            col_low = str(col).lower()
            if not col_low:
                continue
            tables_with_col = [t for t in tables_in_query if col_low in (table_cols.get(t) or set())]
            if tables_with_col:
                present_value_cols.append(col)
                value_col_tables[col] = tables_with_col
        if not present_value_cols:
            return {
                "valid": False,
                "reason": "missing_value_column_table",
                "missing_value_columns": value_columns,
            }
        # If present value columns span multiple tables, require a connected join path.
        tables_for_values = set()
        for tables in value_col_tables.values():
            tables_for_values.update(tables)
        if len(tables_for_values) > 1:
            if not _tables_connected(schema_text, tables_for_values):
                return {
                    "valid": False,
                    "reason": "missing_join_path",
                    "tables": sorted(tables_for_values),
                }

    required_tables = constraints.get("required_tables") or []
    if required_tables:
        tables_in_query = _tables_in_query(sql_low)
        require_all = bool(constraints.get("required_tables_all"))
        if require_all:
            missing = [t for t in required_tables if t not in tables_in_query]
            if missing:
                return {
                    "valid": False,
                    "reason": "missing_required_table",
                    "missing_tables": missing,
                }
        else:
            if not any(t in tables_in_query for t in required_tables):
                return {
                    "valid": False,
                    "reason": "missing_required_table",
                    "missing_tables": required_tables,
                }

    if constraints.get("needs_self_join") and constraints.get("self_join_table"):
        table = str(constraints.get("self_join_table") or "").lower()
        if table:
            # Require a self-join pattern: table appears in FROM and JOIN.
            has_from = re.search(rf"\bfrom\s+{re.escape(table)}\b", sql_low) is not None
            has_join = re.search(rf"\bjoin\s+{re.escape(table)}\b", sql_low) is not None
            if not (has_from and has_join):
                return {
                    "valid": False,
                    "reason": "missing_self_join",
                    "table": table,
                }

    return {"valid": True, "reason": "ok"}
