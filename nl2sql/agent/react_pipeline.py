"""ReAct-style loop for NL-to-SQL.

This controller follows the same high-level ``reason -> act -> observe ->
continue`` pattern as ReAct (Yao et al., dissertation ref [17]) and the
official ``ysymyth/ReAct`` repository. The project-specific adaptation is that
the growing trace is stored as chat-format messages for instruct models rather
than as one flat prompt string, and the action space is narrowed to
SQL-specific ``query[...]`` and ``finish[...]`` actions.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

from ..core.llm import extract_first_select, generate_sql_from_messages
from ..core.postprocess import guarded_postprocess, normalize_sql
from ..core.query_runner import QueryRunner
from ..core.validation import validate_sql
from .agent_tools import ensure_schema_text, get_agent_context
from .prompts import REACT_SYSTEM_PROMPT


@dataclass(frozen=True)
class ReactAblationConfig:
    """Fixed configuration for one ReAct evaluation recipe."""

    name: str = "react_core"
    max_steps: int = 7
    few_shot_k: int = 3
    few_shot_seed: int = 7
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.2
    top_p: float = 0.9
    max_success_refinements: int = 1


def core_react_config(name: str = "react_core") -> ReactAblationConfig:
    """Return the default ReAct configuration used in the dissertation."""
    return ReactAblationConfig(name=name)


# --- Parsing

# ai note copilot: "regex to parse Thought/Action lines with missing-bracket fallback"
def _parse_react_output(raw: str) -> tuple[str, str, str]:
    """Extract (thought, action_type, sql) from a Thought+Action model response.

    action_type is 'finish' or 'query'.  Falls back to extracting the first
    SELECT if the format is not followed exactly.
    """
    def _extract_action_sql(line: str) -> str:
        """Return SQL inside action brackets, tolerating missing closing bracket."""
        open_idx = line.find("[")
        if open_idx < 0:
            return ""
        close_idx = line.rfind("]")
        if close_idx <= open_idx:
            # Some model outputs omit the trailing `]`; keep the remainder.
            return line[open_idx + 1 :].strip()
        return line[open_idx + 1 : close_idx].strip()

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
                sql = _extract_action_sql(line)
            elif lower.startswith("action: query["):
                action = "query"
                sql = _extract_action_sql(line)
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


# --- Observation
# Keep observation text short so the growing trace stays usable in small models.
def _compact_success_text(meta: Any) -> str:
    """Return a short success observation string for the ReAct loop."""
    cols = ", ".join(meta.columns or []) if meta.columns else "none"
    preview = repr(meta.preview_rows or [])
    return (
        f"Success: rows={meta.rowcount}; "
        f"columns={cols}; "
        f"truncated={bool(meta.truncated)}; "
        f"preview={preview}"
    )


# ai note copilot: "map MySQL error strings to short observation tokens"
def _compact_execution_error(error: str | None) -> str:
    """Map noisy DB errors to short, reusable observation labels when possible."""
    err = (error or "").strip()
    low = err.lower()

    if "unknown column" in low:
        m = re.search(r"Unknown column '([^']+)'", err, flags=re.IGNORECASE)
        return f"unknown_column:{m.group(1)}" if m else "unknown_column"
    if "ambiguous column" in low:
        return "ambiguous_column"
    if "doesn't exist" in low and "table" in low:
        return "unknown_table"
    if "you have an error in your sql syntax" in low or "syntax" in low:
        return "syntax_error"
    if "operand should contain" in low:
        return "bad_subquery_shape"
    if "invalid use of group function" in low:
        return "invalid_group_function"
    return err or "execution_failed"


def _observe_with_runner(sql: str, runner: QueryRunner, schema_text: str) -> dict[str, Any]:
    """Validate then execute sql with the provided runner; return an observation dict."""
    check = validate_sql(sql, schema_text)
    if not check.get("valid"):
        return {
            "success": False,
            "text": f"Validation error: {check.get('reason')}",
        }
    meta = runner.run(sql)
    if meta.success:
        return {
            "success": True,
            "text": _compact_success_text(meta),
            "rowcount": meta.rowcount,
            "columns": meta.columns,
            "truncated": meta.truncated,
            "preview_rows": meta.preview_rows,
        }
    return {"success": False, "text": f"Execution error: {_compact_execution_error(meta.error)}"}


def _observe(sql: str, ctx: Any, schema_text: str) -> dict[str, Any]:
    """Validate then execute sql; return an observation dict."""
    return _observe_with_runner(sql, ctx.runner, schema_text)


# --- Initial message construction
# Verified exemplars keep the agent prompt anchored to examples that really execute.
def _build_initial_messages(
    nlq: str,
    schema_text: str,
    cfg: ReactAblationConfig,
) -> list[dict[str, str]]:
    """Build chat-format messages: system, schema, k verified exemplars, then the question."""
    ctx = get_agent_context()
    pool = list(ctx.exemplar_pool or [])
    messages: list[dict[str, str]] = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": "Schema Details:\n" + schema_text},
    ]

    # ai note copilot: "random.Random per-NLQ seeding pattern"
    if cfg.few_shot_k > 0 and pool:
        pool = [ex for ex in pool if ex.get("nlq") != nlq]
        if pool:
            rng = random.Random(f"{cfg.few_shot_seed}:{normalize_sql(nlq)}")
            temp_runner = QueryRunner(ctx.engine)
            accepted = 0
            for ex in rng.sample(pool, len(pool)):
                if accepted >= cfg.few_shot_k:
                    break
                sql = str(ex.get("sql") or "").strip()
                if not sql:
                    continue
                if not sql.endswith(";"):
                    sql += ";"
                obs = _observe_with_runner(sql, temp_runner, schema_text)
                if not obs.get("success"):
                    continue
                # Keep the exemplar reasoning short so the examples teach structure,
                # not long chains of hidden analysis.
                messages.append({"role": "user", "content": f"Example Question: {ex['nlq']}"})
                messages.append({
                    "role": "assistant",
                    "content": (
                        "Thought: I should run a SQL query that directly answers the example question.\n"
                        f"Action: query[{sql}]"
                    ),
                })
                messages.append({"role": "user", "content": f"Observation: {obs['text']}"})
                messages.append({
                    "role": "assistant",
                    "content": (
                        "Thought: The observed result looks consistent with the example question, so I can stop.\n"
                        f"Action: finish[{sql}]"
                    ),
                })
                accepted += 1

    messages.append({"role": "user", "content": f"Natural Language Question: {nlq}"})
    return messages


# --- Main loop

def run_react_pipeline(
    *,
    nlq: str,
    config: ReactAblationConfig | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Run the ReAct loop for one question; return (final_sql, trace).

    generate Thought+Action -> validate/execute -> append Observation -> repeat
    until finish action or max_steps reached.
    """
    cfg = config or core_react_config()
    ctx = get_agent_context()
    schema_text = ensure_schema_text(ctx)

    messages = _build_initial_messages(nlq, schema_text, cfg)
    trace: list[dict[str, Any]] = []
    current_sql = ""
    last_good_sql = ""
    success_refinements_used = 0

    # ai note copilot: "loop boilerplate for the ReAct generate→observe cycle; loop design and termination logic are mine"
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
            # finish[...] is a stop signal only; any SQL inside it is ignored.
            # The authoritative result is always the last successfully *executed*
            # query (last_good_sql), not whatever the model writes here.
            entry["observation"] = {"success": True, "text": "Agent finished."}
            trace.append(entry)
            break

        obs = _observe(sql, ctx, schema_text)
        entry["observation"] = obs
        trace.append(entry)
        messages.append({"role": "user", "content": f"Observation: {obs['text']}"})

        if obs["success"]:
            last_good_sql = sql
            if success_refinements_used < cfg.max_success_refinements:
                success_refinements_used += 1
                continue
            break

    return last_good_sql or current_sql, trace


__all__ = [
    "ReactAblationConfig",
    "core_react_config",
    "run_react_pipeline",
]
