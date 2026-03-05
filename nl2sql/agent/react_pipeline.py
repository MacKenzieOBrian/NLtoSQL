"""
Bare-bones ReAct-style loop for notebook-driven NL->SQL experiments.

Design goals:
1) Keep the live loop short enough to explain clearly in a dissertation.
2) Favor execution-guided repair over hand-built semantic control logic.
3) Keep outputs compatible with existing notebook result handling.

Related methods and evaluation context: ReAct [19], DIN-SQL self-correction
[5], execution feedback optimization [6], Spider [22], and test-suite accuracy
[21].
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from sqlalchemy.engine import Engine

from ..core.llm import extract_first_select, generate_sql_from_messages
from ..core.postprocess import guarded_postprocess, normalize_sql
from ..core.prompting import make_few_shot_messages
from ..core.query_runner import now_utc_iso
from ..core.validation import validate_sql
from ..evaluation.eval import execution_accuracy, test_suite_accuracy_for_item
from .agent_tools import ensure_schema_text, get_agent_context
from .prompts import SQL_GENERATOR_SYSTEM_PROMPT, SQL_REPAIR_SYSTEM_PROMPT


@dataclass(frozen=True)
class ReactAblationConfig:
    name: str = "react_core"
    use_repair_policy: bool = True
    max_repairs: int = 2  # allows one validation repair + one execution-guided repair
    max_steps: int = 8
    few_shot_k: int = 3
    few_shot_seed: int = 7
    # Confound (Item 3): ReAct=256, baseline=128. Disclose in dissertation.
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
            # Per-NLQ deterministic RNG [Brown et al. 2020].
            # Confound (Item 5): baseline uses a global sequential RNG — different exemplar
            # sets are selected for the same NLQ/seed, limiting direct comparability.
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
    """Repair SQL from validator/runtime error feedback (DIN-SQL approach [5])."""
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

    # Zero-shot: generation exemplars (NLQ→SQL) are the wrong format for repair
    # (error+SQL→fix); absent a dedicated repair corpus, zero-shot is correct [5].
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


# ReAct loop helpers — explicit state threading avoids nonlocal/closure complexity.

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
    payload: Optional[dict[str, Any]] = None,
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


def _try_repair(
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
    """KEY FUNCTION — one Reason+Act+Observe loop for a single NLQ [19].

    Returns (final_sql, trace). Trace records every step for dissertation analysis.
    """
    cfg = config or core_react_config()
    ctx = get_agent_context()
    state = _ReactState()
    schema_text = ensure_schema_text(ctx)

    state.step += 1
    _add_trace(
        state,
        "get_schema",
        observation={
            "schema_lines": len((schema_text or "").splitlines()),
            "tables": [
                ln.split("(")[0].strip()
                for ln in (schema_text or "").splitlines()
                if "(" in ln
            ][:10],
        },
    )

    state.step += 1
    state.current_sql = generate_sql(nlq, schema_text, cfg)
    _add_trace(state, "generate_sql", observation={"sql": state.current_sql})

    while state.step < cfg.max_steps:
        state.step += 1
        sql_check = validate_sql(state.current_sql or "", schema_text, nlq=nlq)
        _add_trace(
            state,
            "validate_sql",
            observation=sql_check,
            reason=None if sql_check.get("valid") else sql_check.get("reason"),
        )
        if not sql_check.get("valid"):
            if not _try_repair(
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
        if meta.success:
            return _stop_with(
                state,
                action="run_sql",
                reason="success",
                observation={"sql": state.current_sql},
            )

        if not _try_repair(
            state, cfg,
            nlq=nlq,
            schema_text=schema_text,
            error=meta.error or "run_sql_failed",
            stop_action="run_sql",
            stop_reason="execution_failed",
        ):
            return state.current_sql or "", state.trace

    state.trace.append(
        {
            "step": state.step,
            "action": "stop",
            "planned_action": None,
            "planner_text": "",
            "blocked": False,
            "reason": "max_steps_exhausted" if state.step >= cfg.max_steps else "repair_budget_exhausted",
            "observation": {"sql": state.current_sql},
        }
    )
    return state.current_sql or "", state.trace


def evaluate_react_ablation(
    *,
    test_set: list[dict[str, Any]],
    engine: Engine,
    config: ReactAblationConfig,
    limit: int | None = None,
    ts_suite_db_names: Optional[list[str]] = None,
    ts_make_engine_fn: Optional[Callable[[str], Engine]] = None,
    ts_max_rows: int = 500,
    progress_every: int = 20,
    run_metadata: Optional[dict[str, Any]] = None,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run ReAct over a benchmark split and emit report dict compatible with notebook usage.
    """
    ctx = get_agent_context()
    items = test_set[:limit] if limit else list(test_set)
    out_items: list[dict[str, Any]] = []

    for i, item in enumerate(items):
        nlq = item.get("nlq", "")
        gold_sql = item.get("sql", "")
        pred_sql, trace = run_react_pipeline(nlq=nlq, config=config)

        if pred_sql:
            meta = ctx.runner.run(pred_sql)
            va = bool(meta.success)
            pred_err = meta.error
        else:
            va = False
            pred_err = "no_prediction"

        em = bool(normalize_sql(pred_sql) == normalize_sql(gold_sql)) if pred_sql else False
        ex, ex_pred_err, ex_gold_err = execution_accuracy(
            engine=engine,
            pred_sql=pred_sql if pred_sql else "SELECT 1;",
            gold_sql=gold_sql,
            max_compare_rows=10000,
        )
        if not pred_sql:
            ex = False

        ts: Optional[int] = None
        if (
            va
            and pred_sql
            and ts_suite_db_names
            and ts_make_engine_fn
        ):
            ts = test_suite_accuracy_for_item(
                make_engine_fn=ts_make_engine_fn,
                suite_db_names=ts_suite_db_names,
                gold_sql=gold_sql,
                pred_sql=pred_sql,
                max_rows=ts_max_rows,
            )

        out_items.append(
            {
                "i": i,
                "nlq": nlq,
                "gold_sql": gold_sql,
                "raw_sql": pred_sql,
                "pred_sql": pred_sql,
                "va": bool(va),
                "em": bool(em),
                "ex": bool(ex),
                "ts": ts,
                "error": pred_err or ex_pred_err,
                "gold_error": ex_gold_err,
                "trace": trace,
            }
        )

        if progress_every and ((i + 1) % progress_every == 0 or (i + 1) == len(items)):
            va_rate = sum(int(x["va"]) for x in out_items) / max(len(out_items), 1)
            em_rate = sum(int(x["em"]) for x in out_items) / max(len(out_items), 1)
            ex_rate = sum(int(x["ex"]) for x in out_items) / max(len(out_items), 1)
            print(
                f"ReAct progress {i + 1}/{len(items)} | "
                f"VA={va_rate:.3f} EM={em_rate:.3f} EX={ex_rate:.3f}"
            )

    n = len(out_items)
    va_rate = sum(int(x["va"]) for x in out_items) / max(n, 1)
    em_rate = sum(int(x["em"]) for x in out_items) / max(n, 1)
    ex_rate = sum(int(x["ex"]) for x in out_items) / max(n, 1)
    ts_values = [int(x["ts"]) for x in out_items if x.get("ts") is not None]
    ts_rate = (sum(ts_values) / len(ts_values)) if ts_values else None

    report: dict[str, Any] = {
        "timestamp": now_utc_iso(),
        "method": "react",
        "config": asdict(config),
        "n": n,
        "va_rate": va_rate,
        "em_rate": em_rate,
        "ex_rate": ex_rate,
        "ts_rate": ts_rate,
        "ts_n": len(ts_values),
        "items": out_items,
    }
    if run_metadata:
        report["run_metadata"] = run_metadata

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


__all__ = [
    "ReactAblationConfig",
    "core_react_config",
    "generate_sql",
    "repair_sql",
    "run_react_pipeline",
    "evaluate_react_ablation",
]
