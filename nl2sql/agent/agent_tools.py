"""
Shared runtime context for ReAct-style notebook workflows.

References (project anchors):
- `REFERENCES.md#ref-yao2023-react`
- `REFERENCES.md#ref-zhu2024-survey`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy.engine import Engine

from ..core.schema import build_schema_summary


@dataclass
class AgentContext:
    engine: Engine
    db_name: str
    model: Any
    tok: Any
    runner: Any
    max_new_tokens: int = 128
    schema_cache: Optional[dict[str, Any]] = None
    schema_text_cache: Optional[str] = None
    exemplar_pool: Optional[list[dict[str, Any]]] = None


_AGENT_CONTEXT: AgentContext | None = None


def set_agent_context(ctx: AgentContext) -> None:
    global _AGENT_CONTEXT
    _AGENT_CONTEXT = ctx


def get_agent_context() -> AgentContext:
    if _AGENT_CONTEXT is None:
        raise RuntimeError(
            "Agent context is not set. Call set_agent_context(AgentContext(...)) first."
        )
    return _AGENT_CONTEXT


def schema_to_text(schema_cache: dict[str, Any] | None) -> str:
    """
    Convert structured schema cache into prompt text:
    table(col1, col2, ...)
    Join hints: t1.c1 = t2.c2; ...
    """
    if not isinstance(schema_cache, dict):
        return ""

    lines: list[str] = []
    tables = schema_cache.get("tables") or []
    for t in tables:
        if not isinstance(t, dict):
            continue
        table_name = str(t.get("name") or "").strip()
        if not table_name:
            continue
        cols = t.get("columns") or []
        col_names: list[str] = []
        for c in cols:
            if isinstance(c, dict):
                n = str(c.get("name") or "").strip()
            else:
                n = str(c or "").strip()
            if n:
                col_names.append(n)
        lines.append(f"{table_name}({', '.join(col_names)})")

    hints: list[str] = []
    for fk in schema_cache.get("foreign_keys") or []:
        if not isinstance(fk, dict):
            continue
        t = str(fk.get("table") or "").strip()
        c = str(fk.get("column") or "").strip()
        rt = str(fk.get("ref_table") or "").strip()
        rc = str(fk.get("ref_column") or "").strip()
        if t and c and rt and rc:
            hints.append(f"{t}.{c} = {rt}.{rc}")

    if hints:
        lines.append("Join hints: " + "; ".join(hints))
    return "\n".join(lines)


def ensure_schema_text(ctx: AgentContext) -> str:
    """
    Resolve schema text in this order:
    1) cached schema text
    2) structured schema cache converted to text
    3) live DB summary via build_schema_summary
    """
    if isinstance(ctx.schema_text_cache, str) and ctx.schema_text_cache.strip():
        return ctx.schema_text_cache

    text_from_cache = schema_to_text(ctx.schema_cache)
    if text_from_cache.strip():
        ctx.schema_text_cache = text_from_cache
        return text_from_cache

    schema_text = build_schema_summary(ctx.engine, db_name=ctx.db_name)
    ctx.schema_text_cache = schema_text
    return schema_text


__all__ = [
    "AgentContext",
    "set_agent_context",
    "get_agent_context",
    "schema_to_text",
    "ensure_schema_text",
]

