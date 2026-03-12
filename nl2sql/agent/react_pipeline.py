"""ReAct loop for NL-to-SQL.

Each step the model generates a Thought + Action; the controller executes the
action, appends the Observation, and calls the model again.  This follows the
Yao et al. (2022) ReAct pattern: model-generated reasoning traces interleaved
with external tool observations fed back into the context window.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

from ..core.llm import extract_first_select, generate_sql_from_messages
from ..core.postprocess import guarded_postprocess, normalize_sql
from ..core.prompting import make_few_shot_messages
from ..core.validation import validate_sql
from .agent_tools import ensure_schema_text, get_agent_context
from .prompts import REACT_SYSTEM_PROMPT


@dataclass(frozen=True)
class ReactAblationConfig:
    """Fixed configuration for one ReAct evaluation recipe."""

    name: str = "react_core"
    max_steps: int = 6
    few_shot_k: int = 3
    few_shot_seed: int = 7
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.2
    top_p: float = 0.9


def core_react_config(name: str = "react_core") -> ReactAblationConfig:
    """Return the default ReAct configuration used in the dissertation."""
    return ReactAblationConfig(name=name)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_react_output(raw: str) -> tuple[str, str, str]:
    """Extract (thought, action_type, sql) from a Thought+Action model response.

    action_type is 'finish' or 'query'.  Falls back to extracting the first
    SELECT if the format is not followed exactly.
    """
    thought = ""
    action = "query"
    sql = ""

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("Thought:"):
            thought = line[len("Thought:"):].strip()
        elif line.lower().startswith("action:"):
            # Accept both the strict prompt format:
            #   Action: query[SELECT ...]
            # and the looser form some models emit:
            #   Action: query SELECT ...
            lower = line.lower()
            if lower.startswith("action: finish["):
                action = "finish"
                sql = line[line.index("[") + 1 : line.rindex("]")].strip()
            elif lower.startswith("action: query["):
                action = "query"
                sql = line[line.index("[") + 1 : line.rindex("]")].strip()
            else:
                m = re.match(r"(?is)^action:\s*(query|finish)\s+(.*)$", line)
                if m:
                    action = m.group(1).lower()
                    sql = m.group(2).strip()

    # Fallback: model did not follow the format — grab the first SELECT.
    if not sql:
        sql = extract_first_select(raw) or ""
    if sql and not sql.endswith(";"):
        sql += ";"

    return thought, action, sql


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

def _observe(sql: str, ctx: Any, schema_text: str) -> dict[str, Any]:
    """Validate then execute sql; return an observation dict."""
    check = validate_sql(sql, schema_text)
    if not check.get("valid"):
        return {
            "success": False,
            "text": f"Validation error: {check.get('reason')}",
        }
    meta = ctx.runner.run(sql)
    if meta.success:
        return {"success": True, "text": f"Success: {meta.rowcount} rows returned."}
    return {"success": False, "text": f"Execution error: {meta.error}"}


# ---------------------------------------------------------------------------
# Initial message construction
# ---------------------------------------------------------------------------

def _build_initial_messages(
    nlq: str,
    schema_text: str,
    cfg: ReactAblationConfig,
) -> list[dict[str, str]]:
    """Build the opening message list: system prompt + few-shot + schema + question."""
    ctx = get_agent_context()
    exemplars: list[dict[str, Any]] = []
    pool = list(ctx.exemplar_pool or [])

    if cfg.few_shot_k > 0 and pool:
        pool = [ex for ex in pool if ex.get("nlq") != nlq]
        if pool:
            rng = random.Random(f"{cfg.few_shot_seed}:{normalize_sql(nlq)}")
            exemplars = rng.sample(pool, min(cfg.few_shot_k, len(pool)))

    messages = make_few_shot_messages(schema=schema_text, exemplars=exemplars, nlq=nlq)
    messages[0] = {"role": "system", "content": REACT_SYSTEM_PROMPT}
    return messages


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_react_pipeline(
    *,
    nlq: str,
    config: ReactAblationConfig | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Run the ReAct loop for one question; return (final_sql, trace).

    Loop structure (faithful to Yao et al. 2022):
      model outputs Thought+Action  →  controller observes  →  append Observation
      →  call model again  →  repeat until finish or max_steps.
    """
    cfg = config or core_react_config()
    ctx = get_agent_context()
    schema_text = ensure_schema_text(ctx)

    messages = _build_initial_messages(nlq, schema_text, cfg)
    trace: list[dict[str, Any]] = []
    current_sql = ""

    for step in range(cfg.max_steps):
        raw = generate_sql_from_messages(
            model=ctx.model,
            tokenizer=ctx.tok,
            messages=messages,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
        thought, action, sql = _parse_react_output(str(raw))
        sql = guarded_postprocess(sql, nlq)
        current_sql = sql or current_sql

        entry: dict[str, Any] = {
            "step": step,
            "thought": thought,
            "action": action,
            "sql": sql,
        }
        # Append the model's own output to history so future steps see the trace.
        messages.append({"role": "assistant", "content": str(raw)})

        if action == "finish":
            entry["observation"] = {"success": True, "text": "Agent finished."}
            trace.append(entry)
            break

        obs = _observe(sql, ctx, schema_text)
        entry["observation"] = obs
        trace.append(entry)
        messages.append({"role": "user", "content": f"Observation: {obs['text']}"})

        if obs["success"]:
            break

    return current_sql, trace


__all__ = [
    "ReactAblationConfig",
    "core_react_config",
    "run_react_pipeline",
]
