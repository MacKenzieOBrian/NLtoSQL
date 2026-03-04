"""
Shared runtime context for ReAct-style notebook workflows.

Related literature: ReAct-style reasoning and acting [19] and LLM text-to-SQL
survey coverage [9, 12].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy.engine import Engine

from ..core.schema import build_schema_summary


def _safe_str(obj: Any, key: str) -> str:
    """Get a stripped string field from a dict; returns '' for missing/non-dict input."""
    return str(obj.get(key) or "").strip() if isinstance(obj, dict) else ""


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
    for t in schema_cache.get("tables") or []:
        table_name = _safe_str(t, "name")
        if not table_name:
            continue
        col_names: list[str] = []
        for c in (t.get("columns") or []):
            n = _safe_str(c, "name") if isinstance(c, dict) else str(c or "").strip()
            if n:
                col_names.append(n)
        lines.append(f"{table_name}({', '.join(col_names)})")

    hints: list[str] = []
    for fk in schema_cache.get("foreign_keys") or []:
        t, c = _safe_str(fk, "table"), _safe_str(fk, "column")
        rt, rc = _safe_str(fk, "ref_table"), _safe_str(fk, "ref_column")
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
