"""
Agent tool interface for a ReAct-style NL->SQL loop.

How to read this file:
1) `AgentContext` stores model, DB, and runner handles.
2) Tool functions expose schema, constraints, generate/repair/validate/run.
3) `react_pipeline.py` calls these tools in a Thought->Action->Observation loop.

References:
- ReAct paper: https://arxiv.org/abs/2210.03629
- SQLAlchemy execute docs: https://docs.sqlalchemy.org/en/20/core/connections.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import json

from sqlalchemy import text

from .db import safe_connection
from .schema import list_tables, get_table_columns
from .llm import generate_sql_from_messages
from .prompting import SYSTEM_INSTRUCTIONS
from .query_runner import QueryRunner
from .agent_schema_linking import build_schema_subset, format_join_hints
from .constraint_policy import build_constraints
from .validation import validate_sql as _validate_sql, validate_constraints as _validate_constraints


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


def get_agent_context() -> AgentContext:
    """Public accessor used by module-level ReAct orchestration."""
    return _require_ctx()


def schema_to_text(schema: dict) -> str:
    """Render a structured schema to prompt-friendly text."""
    lines: list[str] = []
    tables: set[str] = set()
    for table in schema.get("tables", []):
        tables.add(table["name"])
        cols = [c["name"] for c in table.get("columns", [])]
        lines.append(f"{table['name']}({', '.join(cols)})")
    if lines:
        lines.append(format_join_hints(tables))
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
    schema_text = _require_ctx().schema_text_cache or schema_to_text(get_schema())
    return build_constraints(nlq, schema_text)


def validate_constraints(sql: str, constraints: Optional[dict]) -> dict:
    """Validate SQL structure against extracted constraints."""
    ctx = _require_ctx()
    schema_text = ctx.schema_text_cache or schema_to_text(get_schema())
    return _validate_constraints(sql, constraints, schema_text=schema_text)


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


def _call_llm(
    messages: list[dict[str, str]],
    *,
    max_new_tokens: Optional[int] = None,
    num_return_sequences: int = 1,
    do_sample: bool = False,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str | list[str]:
    ctx = _require_ctx()
    return generate_sql_from_messages(
        model=ctx.model,
        tokenizer=ctx.tok,
        messages=messages,
        max_new_tokens=max_new_tokens or ctx.max_new_tokens,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )


def generate_sql(
    nlq: str,
    schema_text: str,
    constraints: dict,
    *,
    num_cands: int = 1,
    do_sample: bool = False,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str | list[str]:
    """LLM call that generates SQL candidate(s)."""
    constraint_text = json.dumps(constraints or {}, ensure_ascii=False)
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": f"Schema:\n{schema_text}"},
        {"role": "user", "content": f"NLQ: {nlq}\nConstraints: {constraint_text}\nReturn a single SQL SELECT."},
    ]
    return _call_llm(
        messages,
        max_new_tokens=128,
        num_return_sequences=num_cands,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )


def repair_sql(
    nlq: str,
    bad_sql: str,
    error: str,
    schema_text: str,
) -> str:
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
