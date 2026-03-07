"""Pre-execution SQL validation."""

from __future__ import annotations

from typing import Optional

import re

from .sql_guardrails import clean_candidate_with_reason


def parse_schema_text(schema_text: str) -> tuple[set[str], dict[str, set[str]]]:
    """Parse compact schema text into table and column lookup structures."""
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


# Block SELECT * so predictions stay close to the expected column set.
_SELECT_STAR_RE = re.compile(r"(?is)\bselect\s+([a-zA-Z_][\w$]*\.)?\*")


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
        # Skip subqueries like FROM (SELECT ...) AS t.
        after = sql_low[m.end(): m.end() + 1]
        if after == "(" or table in tables:
            continue
        return False, f"unknown_table:{table}", {}

    return True, "ok", {}


def validate_sql(
    sql: str,
    schema_text: Optional[str] = None,
) -> dict:
    """Validate SQL formatting + schema references without executing."""
    if not sql or not sql.strip():
        return {"valid": False, "reason": "empty_sql"}

    cleaned, reason = clean_candidate_with_reason(sql)
    if not cleaned:
        return {"valid": False, "reason": f"clean_reject:{reason}"}

    if _SELECT_STAR_RE.search(cleaned):
        return {"valid": False, "reason": "select_star_forbidden"}

    if not schema_text:
        return {"valid": False, "reason": "schema_missing"}

    # Only check table names here. Column checks are intentionally omitted.
    tables, _ = parse_schema_text(schema_text)
    if not tables:
        return {"valid": False, "reason": "schema_missing"}

    ok, why, detail = schema_validate(
        sql=cleaned,
        schema_index=(tables, {}),
    )
    if not ok:
        out = {"valid": False, "reason": why}
        if detail:
            out.update(detail)
        return out

    return {"valid": True, "reason": "schema_ok"}
