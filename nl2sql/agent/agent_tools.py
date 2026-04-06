"""Shared runtime context for the agent notebook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy.engine import Engine

from ..core.schema import build_schema_summary


def _safe_str(obj: Any, key: str) -> str:
    """Return an empty string instead of failing on bad schema entries."""
    return str(obj.get(key) or "").strip() if isinstance(obj, dict) else ""


@dataclass
class AgentContext:
    """Shared runtime objects needed by the ReAct notebook and loop.

    This bundles the live DB connection, model, tokenizer, runner, and cached
    schema/exemplar data so the agent helpers do not need long argument lists.
    """

    engine: Engine
    db_name: str
    model: Any
    tok: Any
    runner: Any
    schema_cache: Optional[dict[str, Any]] = None
    schema_text_cache: Optional[str] = None
    exemplar_pool: Optional[list[dict[str, Any]]] = None


_AGENT_CONTEXT: AgentContext | None = None


def set_agent_context(ctx: AgentContext) -> None:
    """Store the current agent context for later helper calls."""
    global _AGENT_CONTEXT
    _AGENT_CONTEXT = ctx


def get_agent_context() -> AgentContext:
    """Return the current agent context or fail loudly if it was never set."""
    if _AGENT_CONTEXT is None:
        raise RuntimeError(
            "Agent context is not set. Call set_agent_context(AgentContext(...)) first."
        )
    return _AGENT_CONTEXT


# ai note copilot: "dict traversal to compact schema+FK hint text"
def schema_to_text(schema_cache: dict[str, Any] | None) -> str:
    """Turn a schema dict into simple prompt text."""
    if not isinstance(schema_cache, dict):
        return ""

    lines: list[str] = []
    for t in schema_cache.get("tables") or []:
        table_name = _safe_str(t, "name")
        if not table_name:
            continue

        # Collect the column names for th table so 
        # one compact line like customers(id, name, country).
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

        # Foreign key hints make likely join paths explicit in the prompt.
        if t and c and rt and rc:
            hints.append(f"{t}.{c} = {rt}.{rc}")

    if hints:
        lines.append("Join hints: " + "; ".join(hints))
    return "\n".join(lines)


def ensure_schema_text(ctx: AgentContext) -> str:
    """Return schema text, using cached data first when available."""
    # Reuse cached schema text if we already built it earlier.
    if isinstance(ctx.schema_text_cache, str) and ctx.schema_text_cache.strip():
        return ctx.schema_text_cache

    # If structured schema data is cached, convert that into prompt text.
    text_from_cache = schema_to_text(ctx.schema_cache)
    if text_from_cache.strip():
        ctx.schema_text_cache = text_from_cache
        return text_from_cache

    # Fallback: rebuild the schema summary from the live database.
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
