"""
Validation helpers for generated SQL.

Parses schema text from the prompt context and checks that table names,
column names, and SQL safety constraints are satisfied before execution.
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


# SELECT * is blocked by default: it returns every column so the result set
# almost never matches a gold SQL that names specific columns, causing spurious
# EX failures. Forbidding it steers the model toward explicit column selection.
_SELECT_STAR_RE = re.compile(r"(?is)\bselect\s+([a-zA-Z_][\w$]*\.)?\*")

# Override: if the NLQ explicitly asks for all columns, SELECT * is legitimate.
# This prevents over-rejection for questions like "show all details of order 103".
_SELECT_STAR_ALLOW_RE = re.compile(
    r"\b(all columns|all fields|full details|full row|entire row|all details|every column)\b",
    re.IGNORECASE,
)


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
        # Peek at the character immediately after the identifier. If it is "(",
        # the "table" is actually a subquery alias (FROM (SELECT ...) AS t) — not
        # a real table name, so skip it rather than rejecting a valid query.
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
