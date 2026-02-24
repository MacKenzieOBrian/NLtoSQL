"""
Shared helpers for notebook experiment orchestration.

These helpers keep baseline and QLoRA notebooks aligned so behavior does not
drift when one notebook is edited.
"""

from __future__ import annotations

import re
from typing import Any


def model_alias_from_id(model_id: str) -> str:
    """Convert a model id into a filesystem-safe alias."""
    tail = (model_id or "model").split("/")[-1]
    alias = re.sub(r"[^a-z0-9]+", "_", tail.lower()).strip("_")
    return alias or "model"


def make_prompt_variants(default_system_instructions: str) -> dict[str, str]:
    """Return controlled prompt variants for ablation runs."""
    return {
        "default": default_system_instructions,
        "schema_only_minimal": """You are an expert data analyst writing MySQL queries.
Given the database schema and a natural language question, write a single SQL SELECT query.

Rules:
- Output ONLY SQL (no explanation, no markdown).
- Output exactly ONE statement, starting with SELECT.
- Use only tables/columns listed in the schema.
""",
        "no_routing_hints": default_system_instructions.split("- Routing hints:")[0].rstrip(),
    }


def schema_variant_text(schema_text: str, variant: str) -> str:
    """Apply schema truncation variants used in prompt ablations."""
    lines = schema_text.splitlines()
    if variant == "full":
        return schema_text
    if variant == "first_80_lines":
        return "\n".join(lines[:80])
    if variant == "first_40_lines":
        return "\n".join(lines[:40])
    raise ValueError(f"Unknown SCHEMA_VARIANT: {variant}")


def exemplar_pool_for_strategy(items: list[dict[str, Any]], strategy: str) -> list[dict[str, Any]]:
    """Apply few-shot exemplar pool strategies for ablation runs."""
    if strategy == "all":
        return list(items)

    def _sql(x: dict[str, Any]) -> str:
        return str(x.get("sql", "")).strip()

    def _is_join(sql: str) -> bool:
        s = sql.lower()
        return " join " in f" {s} "

    def _is_agg(sql: str) -> bool:
        return bool(re.search(r"\b(sum|avg|count|min|max)\s*\(", sql.lower()))

    if strategy == "brief_sql":
        ranked = sorted(items, key=lambda x: len(_sql(x)))
        keep = max(50, int(0.4 * len(ranked)))
        pool = ranked[:keep]
    elif strategy == "join_heavy":
        pool = [x for x in items if _is_join(_sql(x))]
    elif strategy == "agg_heavy":
        pool = [x for x in items if _is_agg(_sql(x))]
    else:
        raise ValueError(f"Unknown EXEMPLAR_STRATEGY: {strategy}")

    # fallback keeps runs stable if a strategy gets too small
    return pool if len(pool) >= 10 else list(items)


__all__ = [
    "exemplar_pool_for_strategy",
    "make_prompt_variants",
    "model_alias_from_id",
    "schema_variant_text",
]

