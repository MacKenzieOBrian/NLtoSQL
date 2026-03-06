"""Simple ReAct-style loop for the agent notebook."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from ..core.llm import extract_first_select, generate_sql_from_messages
from ..core.postprocess import guarded_postprocess, normalize_sql
from ..core.prompting import make_few_shot_messages
from ..core.validation import validate_sql
from .agent_tools import ensure_schema_text, get_agent_context
from .prompts import SQL_GENERATOR_SYSTEM_PROMPT, SQL_REPAIR_SYSTEM_PROMPT


@dataclass(frozen=True)
class ReactAblationConfig:
    name: str = "react_core"
    use_repair_policy: bool = True
    max_repairs: int = 2
    max_steps: int = 8
    few_shot_k: int = 3
    few_shot_seed: int = 7
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.2
    top_p: float = 0.9


def core_react_config(name: str = "react_core") -> ReactAblationConfig:
    return ReactAblationConfig(name=name)


def _clean_sql_candidate(sql: str) -> str:
    sql = extract_first_select(str(sql)) or str(sql or "").strip()
    if sql and not sql.endswith(";"):
        sql += ";"
    return sql


def _schema_observation(schema_text: str) -> dict[str, Any]:
    """Small schema summary used in the trace."""
    return {
        "schema_lines": len((schema_text or "").splitlines()),
        "tables": [
            line.split("(")[0].strip()
            for line in (schema_text or "").splitlines()
            if "(" in line
        ][:10],
    }


def _build_prompt_messages(
    *,
    nlq: str,
    schema_text: str,
    system_prompt: str,
    config: ReactAblationConfig,
    final_user_content: str | None = None,
) -> list[dict[str, str]]:
    ctx = get_agent_context()
    exemplars: list[dict[str, Any]] = []
    pool = list(ctx.exemplar_pool or [])

    if config.few_shot_k > 0 and pool:
        pool = [ex for ex in pool if ex.get("nlq") != nlq]
        if pool:
            sample_n = min(config.few_shot_k, len(pool))
            # Keep exemplar selection stable for the same question and seed.
            rng = random.Random(f"{config.few_shot_seed}:{normalize_sql(nlq)}")
            exemplars = rng.sample(pool, sample_n)

    messages = make_few_shot_messages(
        schema=schema_text,
        exemplars=exemplars,
        nlq=nlq,
    )
    messages[0] = {"role": "system", "content": system_prompt}
    if final_user_content is not None:
        messages[-1] = {"role": "user", "content": final_user_content}
    return messages


def _repair_hint(error: str) -> str:
    if error.startswith("validate_sql:"):
        reason = error.split(":", 1)[1]
        if reason == "select_star_forbidden":
            return (
                "Repair hint: do not use SELECT *. Return only the smallest set of columns "
                "needed to answer the question."
            )
    return ""


def generate_sql(nlq: str, schema_text: str, config: ReactAblationConfig | None = None) -> str:
    """Generate one SQL candidate for the NLQ."""
    ctx = get_agent_context()
    cfg = config or core_react_config()
    if ctx.model is None or ctx.tok is None:
        raise RuntimeError("Agent context model/tokenizer not set for SQL generation.")

    messages = _build_prompt_messages(
        nlq=nlq,
        schema_text=schema_text,
        system_prompt=SQL_GENERATOR_SYSTEM_PROMPT,
        config=cfg,
    )
    out = generate_sql_from_messages(
        model=ctx.model,
        tokenizer=ctx.tok,
        messages=messages,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        extract_select=True,
        stop_on_semicolon=True,
    )
    return guarded_postprocess(_clean_sql_candidate(str(out)), nlq)


def repair_sql(nlq: str, bad_sql: str, error: str, schema_text: str, config: ReactAblationConfig | None = None) -> str:
    """Try to repair SQL using the question, bad SQL, and error message."""
    ctx = get_agent_context()
    cfg = config or core_react_config()
    if ctx.model is None or ctx.tok is None:
        raise RuntimeError("Agent context model/tokenizer not set for SQL repair.")

    hint = _repair_hint(error)
    repair_prompt = (
        f"Natural Language Question: {nlq}\n\n"
        f"Previous SQL:\n{bad_sql}\n\n"
        f"Observed Error:\n{error}\n"
    )
    if hint:
        repair_prompt += f"\n{hint}\n"
    repair_prompt += "\nReturn one corrected SQL SELECT."

    # Keep repair prompt simple: schema + broken SQL + error.
    messages = [
        {"role": "system", "content": SQL_REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": "Schema Details:\n" + schema_text},
        {"role": "user", "content": repair_prompt},
    ]
    out = generate_sql_from_messages(
        model=ctx.model,
        tokenizer=ctx.tok,
        messages=messages,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        extract_select=True,
        stop_on_semicolon=True,
    )
    return guarded_postprocess(_clean_sql_candidate(str(out)), nlq)


@dataclass
class _ReactState:
    step: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)
    current_sql: str | None = None
    repairs_used: int = 0


def _add_trace(
    state: _ReactState,
    action: str,
    *,
    observation: dict[str, Any],
    reason: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    state.trace.append(
        {
            "step": state.step,
            "action": action,
            "planned_action": action,
            "planner_text": f"Action: {action}[{json.dumps(payload or {}, ensure_ascii=True)}]",
            "payload": payload or {},
            "blocked": False,
            "reason": reason,
            "observation": observation,
        }
    )


def _stop_with(
    state: _ReactState,
    *,
    action: str,
    reason: str,
    observation: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    state.trace.append(
        {
            "step": state.step,
            "action": "stop",
            "planned_action": action,
            "planner_text": f"Action: {action}[{{}}]",
            "blocked": False,
            "reason": reason,
            "observation": observation,
        }
    )
    return state.current_sql or "", state.trace


def _validate_current_sql(state: _ReactState, schema_text: str) -> dict[str, Any]:
    """Validate the current SQL and record the trace entry."""
    sql_check = validate_sql(state.current_sql or "", schema_text)
    _add_trace(
        state,
        "validate_sql",
        observation=sql_check,
        reason=None if sql_check.get("valid") else sql_check.get("reason"),
    )
    return sql_check


def _run_current_sql(ctx: Any, state: _ReactState) -> Any:
    """Run the current SQL and record the trace entry."""
    meta = ctx.runner.run(state.current_sql or "")
    run_obs = {
        "success": bool(meta.success),
        "rowcount": int(meta.rowcount),
        "error": meta.error,
        "sql": state.current_sql,
    }
    _add_trace(
        state,
        "run_sql",
        observation=run_obs,
        reason=None if meta.success else (meta.error or "run_sql_failed"),
    )
    return meta


def _repair_or_stop(
    state: _ReactState,
    cfg: ReactAblationConfig,
    *,
    nlq: str,
    schema_text: str,
    error: str,
    stop_action: str,
    stop_reason: str,
) -> bool:
    if not cfg.use_repair_policy or state.repairs_used >= cfg.max_repairs or state.step >= cfg.max_steps:
        _stop_with(
            state,
            action=stop_action,
            reason=stop_reason,
            observation={"sql": state.current_sql, "error": error},
        )
        return False

    state.step += 1
    state.current_sql = repair_sql(nlq, state.current_sql or "", error, schema_text, cfg)
    state.repairs_used += 1
    _add_trace(
        state,
        "repair_sql",
        observation={"sql": state.current_sql, "repairs_used": state.repairs_used},
        payload={"error": error},
    )
    return True


def run_react_pipeline(
    *,
    nlq: str,
    config: ReactAblationConfig | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Run the agent loop for one question and return (sql, trace)."""
    # Inspired by a simple ReAct loop: generate -> validate -> run -> repair.
    cfg = config or core_react_config()
    ctx = get_agent_context()
    state = _ReactState()
    schema_text = ensure_schema_text(ctx)

    state.step += 1
    _add_trace(
        state,
        "get_schema",
        observation=_schema_observation(schema_text),
    )

    state.step += 1
    state.current_sql = generate_sql(nlq, schema_text, cfg)
    _add_trace(state, "generate_sql", observation={"sql": state.current_sql})

    while state.step < cfg.max_steps:
        state.step += 1
        sql_check = _validate_current_sql(state, schema_text)
        if not sql_check.get("valid"):
            if not _repair_or_stop(
                state, cfg,
                nlq=nlq,
                schema_text=schema_text,
                error=f"validate_sql:{sql_check.get('reason')}",
                stop_action="validate_sql",
                stop_reason="validation_failed",
            ):
                return state.current_sql or "", state.trace
            continue

        state.step += 1
        meta = _run_current_sql(ctx, state)
        if meta.success:
            return _stop_with(
                state,
                action="run_sql",
                reason="success",
                observation={"sql": state.current_sql},
            )

        if not _repair_or_stop(
            state, cfg,
            nlq=nlq,
            schema_text=schema_text,
            error=meta.error or "run_sql_failed",
            stop_action="run_sql",
            stop_reason="execution_failed",
        ):
            return state.current_sql or "", state.trace

    return _stop_with(
        state,
        action="run_sql",
        reason="max_steps_exhausted" if state.step >= cfg.max_steps else "repair_budget_exhausted",
        observation={"sql": state.current_sql},
    )
__all__ = [
    "ReactAblationConfig",
    "core_react_config",
    "generate_sql",
    "repair_sql",
    "run_react_pipeline",
]
