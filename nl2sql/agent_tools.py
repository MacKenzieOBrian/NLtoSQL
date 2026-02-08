"""
Agent tool interface for a ReAct-style NL->SQL loop.
The LLM selects actions; these functions execute them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import json

import re
from sqlalchemy import text

from .db import safe_connection
from .schema import list_tables, get_table_columns
from .llm import generate_sql_from_messages
from .prompting import SYSTEM_INSTRUCTIONS
from .query_runner import QueryRunner
from .agent_utils import (
    build_schema_subset,
    _extract_value_hints,
    _extract_required_columns,
    _projection_hints,
)
from .validation import parse_schema_text, validate_sql as _validate_sql, validate_constraints as _validate_constraints


@dataclass
class AgentContext:
    engine: Any
    db_name: str
    model: Any
    tok: Any
    runner: QueryRunner
    schema_cache: Optional[dict] = None
    schema_text_cache: Optional[str] = None
    max_new_tokens: int = 128


_CTX: Optional[AgentContext] = None


def set_agent_context(ctx: AgentContext) -> None:
    global _CTX
    _CTX = ctx


def _require_ctx() -> AgentContext:
    if _CTX is None:
        raise RuntimeError("Agent tools not initialized. Call set_agent_context(...) first.")
    return _CTX


def schema_to_text(schema: dict) -> str:
    """Render a structured schema to prompt-friendly text."""
    lines: list[str] = []
    for table in schema.get("tables", []):
        cols = [c["name"] for c in table.get("columns", [])]
        lines.append(f"{table['name']}({', '.join(cols)})")
    return "\n".join(lines)


def link_schema(nlq: str, schema_text: Optional[str] = None, max_tables: int = 6) -> dict:
    """Return a pruned schema text + join hints for the NLQ."""
    if not schema_text:
        schema_text = _require_ctx().schema_text_cache or schema_to_text(get_schema())
    subset, debug = build_schema_subset(
        schema_text, nlq, max_tables=max_tables, return_debug=True
    )
    return {
        "schema_text": subset,
        "changed": subset != schema_text,
        "link_debug": debug,
    }


def extract_constraints(nlq: str) -> dict:
    """Lightweight, deterministic constraint extraction from NLQ."""
    # Regex reference: https://docs.python.org/3/library/re.html
    # Rationale: structural cues (COUNT, ORDER BY, LIMIT) were a common source of EX failures.
    nl = (nlq or "").lower()

    agg = None
    if re.search(r"\bcount\b|how many|number of|total number of|count of", nl):
        agg = "COUNT"
    elif re.search(r"\baverage\b|\bavg\b|mean", nl):
        agg = "AVG"
    elif re.search(r"\btotal\b|\bsum\b|how much", nl):
        agg = "SUM"
    elif re.search(r"\bmaximum\b|\bmax\b|highest|most", nl):
        agg = "MAX"
    elif re.search(r"\bminimum\b|\bmin\b|lowest|least", nl):
        agg = "MIN"

    needs_group_by = bool(agg and re.search(r"\bper\b|\bby\b|for each|each", nl))
    needs_order_by = bool(re.search(r"\btop\b|\bhighest\b|\blowest\b|\bmost\b|\bleast\b|sorted|ranked|order by", nl))

    limit = None
    m = re.search(r"\b(top|first|last)\s+(\d+)\b", nl)
    if m:
        try:
            limit = int(m.group(2))
        except ValueError:
            limit = None

    distinct = bool(re.search(r"\b(unique|distinct|different)\b", nl))

    value_hints = _extract_value_hints(nlq)
    explicit_fields = _extract_required_columns(nlq)
    projection_hints = _projection_hints(nlq)
    needs_location = bool(
        value_hints and re.search(r"\b(in|from|located|based|office)\b", nl)
    )

    schema_text = _require_ctx().schema_text_cache or schema_to_text(get_schema())
    _, table_cols = parse_schema_text(schema_text)
    location_cols = {"city", "country", "state", "territory", "region"}
    location_tables = sorted(
        [t for t, cols in table_cols.items() if cols & location_cols]
    )

    return {
        "agg": agg,
        "needs_group_by": needs_group_by,
        "needs_order_by": needs_order_by,
        "limit": limit,
        "distinct": distinct,
        "value_hints": value_hints,
        "explicit_fields": explicit_fields,
        "projection_hints": projection_hints,
        "needs_location": needs_location,
        "location_tables": location_tables,
    }


def validate_constraints(sql: str, constraints: Optional[dict]) -> dict:
    """Validate SQL structure against extracted constraints."""
    return _validate_constraints(sql, constraints)


def get_schema() -> dict:
    """Return structured schema: tables, columns, PK/FK."""
    ctx = _require_ctx()
    if ctx.schema_cache is not None:
        return ctx.schema_cache

    engine = ctx.engine
    db_name = ctx.db_name
    tables = list_tables(engine)

    schema: dict[str, Any] = {"tables": [], "foreign_keys": []}
    for table in tables:
        cols_df = get_table_columns(engine, db_name=db_name, table_name=table)
        columns = []
        pk_cols = []
        for _, row in cols_df.iterrows():
            col = {
                "name": str(row["COLUMN_NAME"]),
                "type": str(row["DATA_TYPE"]),
                "nullable": str(row["IS_NULLABLE"]).upper() == "YES",
                "key": str(row["COLUMN_KEY"] or ""),
            }
            if col["key"] == "PRI":
                pk_cols.append(col["name"])
            columns.append(col)
        schema["tables"].append({"name": table, "columns": columns, "primary_keys": pk_cols})

    fk_query = text(
        """
        SELECT TABLE_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = :db AND REFERENCED_TABLE_NAME IS NOT NULL
        """
    )
    with safe_connection(engine) as conn:
        fk_rows = conn.execute(fk_query, {"db": db_name}).fetchall()
    for row in fk_rows:
        schema["foreign_keys"].append(
            {
                "table": row[0],
                "column": row[1],
                "ref_table": row[2],
                "ref_column": row[3],
            }
        )

    ctx.schema_cache = schema
    ctx.schema_text_cache = schema_to_text(schema)
    return schema


def validate_sql(sql: str, schema_text: Optional[str] = None) -> dict:
    """Validate SQL formatting + schema references without executing."""
    # Rationale: catch obvious formatting/schema errors before hitting the database.
    if schema_text is None:
        schema_text = _require_ctx().schema_text_cache or schema_to_text(get_schema())
    if not schema_text:
        return {"valid": False, "reason": "schema_missing"}
    return _validate_sql(sql, schema_text, enforce_join_hints=True)


def get_table_samples(table: str, n: int = 3) -> list[dict]:
    """Return example rows to ground column usage."""
    ctx = _require_ctx()
    if not table:
        return [{"_error": "Missing table name"}]
    try:
        with safe_connection(ctx.engine) as conn:
            res = conn.execute(text(f"SELECT * FROM {table} LIMIT :n"), {"n": n})
            cols = list(res.keys())
            rows = res.fetchmany(n)
        return [dict(zip(cols, row)) for row in rows]
    except Exception as e:
        return [{"_error": str(e)}]


def _call_llm(messages: list[dict[str, str]], *, max_new_tokens: Optional[int] = None) -> str:
    ctx = _require_ctx()
    return generate_sql_from_messages(
        model=ctx.model,
        tokenizer=ctx.tok,
        messages=messages,
        max_new_tokens=max_new_tokens or ctx.max_new_tokens,
    )


def generate_sql(nlq: str, schema_text: str, constraints: dict) -> str:
    """LLM call that generates a single SQL candidate."""
    constraint_text = json.dumps(constraints or {}, ensure_ascii=False)
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": f"Schema:\n{schema_text}"},
        {"role": "user", "content": f"NLQ: {nlq}\nConstraints: {constraint_text}\nReturn a single SQL SELECT."},
    ]
    return _call_llm(messages, max_new_tokens=128)


def repair_sql(nlq: str, bad_sql: str, error: str, schema_text: str) -> str:
    """LLM call that revises SQL using execution feedback."""
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": f"Schema:\n{schema_text}"},
        {
            "role": "user",
            "content": (
                "The previous SQL failed. Fix it.\n"
                f"NLQ: {nlq}\n"
                f"Bad SQL: {bad_sql}\n"
                f"Error: {error}\n"
                "Return a corrected single SQL SELECT."
            ),
        },
    ]
    return _call_llm(messages, max_new_tokens=128)


def run_sql(sql: str) -> dict:
    """Execute SQL. Return rows OR error string."""
    ctx = _require_ctx()
    meta = ctx.runner.run(sql, capture_df=True)
    if not meta.success:
        return {"success": False, "error": meta.error}
    rows = []
    if meta.result_preview is not None:
        rows = meta.result_preview.to_dict(orient="records")
    return {
        "success": True,
        "rows": rows,
        "rowcount": meta.rowcount,
        "truncated": meta.truncated,
        "columns": meta.columns,
    }


def finish(answer: str, sql: str, provenance: dict) -> dict:
    """Final output container."""
    return {"answer": answer, "sql": sql, "provenance": provenance}
