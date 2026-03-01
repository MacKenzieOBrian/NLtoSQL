"""
Shared validation utilities used by both the tool-driven and class-based agents.

How to read this file:
1) Parse schema text into tables/columns.
2) Validate SQL against basic schema references.
3) Optionally validate a small set of simple SQL-shape hints.

References (project anchors):
- `REFERENCES.md#ref-wang2020-ratsql`
- `REFERENCES.md#ref-lin2020-bridge`
- `REFERENCES.md#ref-li2023-resdsql`

Implementation docs:
- Python regex docs: https://docs.python.org/3/library/re.html
- SQL SELECT syntax (GROUP BY / ORDER BY / LIMIT): https://dev.mysql.com/doc/refman/8.0/en/select.html
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
_STRING_LITERAL_RE = re.compile(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"")
_TABLE_ALIAS_RE = re.compile(
    r"(?is)\b(?:from|join)\s+([a-zA-Z_][\w$]*)"
    r"(?:\s+(?:as\s+)?(?!(?:on|where|group|order|having|limit|join|left|right|inner|outer|using)\b)([a-zA-Z_][\w$]*))?"
)
_OUTPUT_ALIAS_RE = re.compile(r"(?is)\bas\s+([a-zA-Z_][\w$]*)\b")
_IDENT_RE = re.compile(r"(?<!\.)\b([a-zA-Z_][\w$]*)\b")
_SELECT_STAR_RE = re.compile(r"(?is)\bselect\s+([a-zA-Z_][\w$]*\.)?\*")
_SELECT_STAR_ALLOW_RE = re.compile(
    r"\b(all columns|all fields|full details|full row|entire row|all details|every column)\b",
    re.IGNORECASE,
)
_SQL_KEYWORDS = {
    "select", "from", "where", "join", "left", "right", "inner", "outer", "on", "as",
    "group", "by", "order", "having", "limit", "distinct", "and", "or", "not", "in",
    "is", "null", "like", "between", "case", "when", "then", "else", "end", "desc", "asc",
    "using", "union", "all", "exists", "count", "sum", "avg", "min", "max", "round",
    "cast", "coalesce", "if", "date", "year", "month", "day", "true", "false", "interval",
}


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


def _query_tables_and_aliases(sql_low: str) -> tuple[set[str], dict[str, str], set[str]]:
    tables: set[str] = set()
    alias_map: dict[str, str] = {}
    output_aliases: set[str] = set()
    for m in _TABLE_ALIAS_RE.finditer(sql_low or ""):
        table = (m.group(1) or "").lower()
        alias = (m.group(2) or "").lower()
        if table:
            tables.add(table)
        if alias and alias not in _SQL_KEYWORDS:
            alias_map[alias] = table
    for m in _OUTPUT_ALIAS_RE.finditer(sql_low or ""):
        alias = (m.group(1) or "").lower()
        if alias and alias not in _SQL_KEYWORDS:
            output_aliases.add(alias)
    return tables, alias_map, output_aliases


def _allows_select_star(nlq: Optional[str]) -> bool:
    return bool(_SELECT_STAR_ALLOW_RE.search(nlq or ""))


def _is_function_identifier(scrubbed_sql: str, end_pos: int) -> bool:
    tail = scrubbed_sql[end_pos:]
    m = re.match(r"\s*\(", tail)
    return m is not None


def _visible_column_matches(column: str, query_tables: set[str], table_cols: dict[str, set[str]]) -> set[str]:
    if not query_tables:
        return set()
    return {table for table in query_tables if column in table_cols.get(table, set())}


def _unknown_or_ambiguous_column(
    sql_low: str,
    *,
    query_tables: set[str],
    alias_map: dict[str, str],
    output_aliases: set[str],
    table_cols: dict[str, set[str]],
) -> Optional[str]:
    scrubbed = _STRING_LITERAL_RE.sub(" ", sql_low or "")
    for match in _IDENT_RE.finditer(scrubbed):
        tok = (match.group(1) or "").lower()
        if (
            tok in output_aliases
            or tok in alias_map
            or tok in query_tables
            or tok in _SQL_KEYWORDS
        ):
            continue
        if tok.isdigit():
            continue
        if _is_function_identifier(scrubbed, match.end()):
            continue
        matched_tables = _visible_column_matches(tok, query_tables, table_cols)
        if not matched_tables:
            return f"unknown_column:{tok}"
        if len(matched_tables) > 1:
            return f"ambiguous_column:{tok}"
    return None


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
    query_tables, alias_map, output_aliases = _query_tables_and_aliases(sql_low)

    # validate explicit table names in from/join (skip subqueries).
    for m in re.finditer(r"(?is)\b(from|join)\s+([a-zA-Z_][\w$]*)", sql_low):
        table = m.group(2)
        after = sql_low[m.end() : m.end() + 1]
        if after == "(":
            continue
        if table not in tables:
            return False, f"unknown_table:{table}", {}

    # validate qualified columns table.column when table is known.
    for m in re.finditer(r"(?is)\b([a-zA-Z_][\w$]*)\.([a-zA-Z_][\w$]*)\b", sql_low):
        qualifier = m.group(1)
        col = m.group(2)
        table = alias_map.get(qualifier, qualifier)
        if table in table_cols and col not in table_cols[table]:
            return False, f"unknown_column:{qualifier}.{col}", {}

    col_issue = _unknown_or_ambiguous_column(
        sql_low,
        query_tables=query_tables,
        alias_map=alias_map,
        output_aliases=output_aliases,
        table_cols=table_cols,
    )
    if col_issue:
        return False, col_issue, {}

    return True, "ok", {}


def validate_sql(
    sql: str,
    schema_text: Optional[str] = None,
    *,
    enforce_join_hints: bool = True,
    nlq: Optional[str] = None,
) -> dict:
    """Validate SQL formatting + schema references without executing."""
    # rationale: catch obvious formatting/schema errors before hitting the database.
    if not sql or not sql.strip():
        return {"valid": False, "reason": "empty_sql"}

    cleaned, reason = clean_candidate_with_reason(sql)
    if not cleaned:
        return {"valid": False, "reason": f"clean_reject:{reason}"}

    if _SELECT_STAR_RE.search(cleaned) and not _allows_select_star(nlq):
        return {"valid": False, "reason": "select_star_forbidden"}

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

    return {"valid": True, "reason": "ok"}
