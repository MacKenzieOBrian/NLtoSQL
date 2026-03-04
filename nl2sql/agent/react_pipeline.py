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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from sqlalchemy.engine import Engine

from ..core.llm import extract_first_select, generate_sql_from_messages
from ..core.postprocess import guarded_postprocess, normalize_sql
from ..core.prompting import make_few_shot_messages
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
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.2
    top_p: float = 0.9


def core_react_config(name: str = "react_core") -> ReactAblationConfig:
    return ReactAblationConfig(name=name)


_ACTIVE_CONFIG: ReactAblationConfig | None = None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _effective_config() -> ReactAblationConfig:
    return _ACTIVE_CONFIG or core_react_config()


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
    final_user_content: str | None = None,
) -> list[dict[str, str]]:
    ctx = get_agent_context()
    config = _effective_config()
    exemplars: list[dict[str, Any]] = []
    pool = list(ctx.exemplar_pool or [])

    if config.few_shot_k > 0 and pool:
        pool = [ex for ex in pool if ex.get("nlq") != nlq]
        if pool:
            sample_n = min(config.few_shot_k, len(pool))
            # Few-shot in-context learning: sample k exemplars from the pool.
            # Approach from Brown et al. (2020) "Language Models are Few-Shot Learners"
            # https://arxiv.org/abs/2005.14165
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


def generate_sql(nlq: str, schema_text: str) -> str:
    """
    Generate one SQL candidate for the NLQ.
    Signature is intentionally stable for notebook monkeypatch demos.
    """
    ctx = get_agent_context()
    config = _effective_config()
    if ctx.model is None or ctx.tok is None:
        raise RuntimeError("Agent context model/tokenizer not set for SQL generation.")

    messages = _build_prompt_messages(
        nlq=nlq,
        schema_text=schema_text,
        system_prompt=SQL_GENERATOR_SYSTEM_PROMPT,
    )
    out = generate_sql_from_messages(
        model=ctx.model,
        tokenizer=ctx.tok,
        messages=messages,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        temperature=config.temperature,
        top_p=config.top_p,
        extract_select=True,
        stop_on_semicolon=True,
    )
    return guarded_postprocess(_clean_sql_candidate(str(out)), nlq)


def repair_sql(nlq: str, bad_sql: str, error: str, schema_text: str) -> str:
    """
    Repair SQL from validator/runtime feedback.
    Signature is intentionally stable for notebook monkeypatch demos.

    Execution-guided repair follows the DIN-SQL approach:
    Pourreza & Rafiei (2023) "DIN-SQL: Decomposed In-Context Learning of Text-to-SQL"
    https://arxiv.org/abs/2304.11015
    """
    ctx = get_agent_context()
    config = _effective_config()
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

    # Zero-shot repair: NLQ→SQL generation exemplars are the wrong format for
    # the repair task (error+SQL→fix).  Repair exemplars would require a separate
    # corpus of (broken SQL, error, corrected SQL) triples; absent that, zero-shot
    # with schema context is the correct approach per DIN-SQL (Pourreza & Rafiei 2023).
    messages = [
        {"role": "system", "content": SQL_REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": "Schema Details:\n" + schema_text},
        {"role": "user", "content": repair_prompt},
    ]
    out = generate_sql_from_messages(
        model=ctx.model,
        tokenizer=ctx.tok,
        messages=messages,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        temperature=config.temperature,
        top_p=config.top_p,
        extract_select=True,
        stop_on_semicolon=True,
    )
    return guarded_postprocess(_clean_sql_candidate(str(out)), nlq)


def run_react_pipeline(
    *,
    nlq: str,
    config: ReactAblationConfig | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Execute one ReAct loop for a single NLQ and return (final_sql, trace).

    Implements the Reason+Act+Observe loop from:
    Yao et al. (2023) "ReAct: Synergizing Reasoning and Acting in Language Models", ICLR 2023
    https://arxiv.org/abs/2210.03629
    """
    global _ACTIVE_CONFIG
    cfg = config or core_react_config()
    _ACTIVE_CONFIG = cfg
    ctx = get_agent_context()

    trace: list[dict[str, Any]] = []
    schema_text = ensure_schema_text(ctx)
    current_sql: str | None = None
    last_error: str | None = None
    repairs_used = 0
    step = 0

    def next_step() -> int:
        nonlocal step
        step += 1
        return step

    def add_trace(
        action: str,
        *,
        observation: dict[str, Any],
        reason: str | None = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        trace.append(
            {
                "step": step,
                "action": action,
                "planned_action": action,
                "planner_text": f"Action: {action}[{json.dumps(payload or {}, ensure_ascii=True)}]",
                "payload": payload or {},
                "blocked": False,
                "reason": reason,
                "observation": observation,
            }
        )

    def stop_with(*, action: str, reason: str, observation: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        trace.append(
            {
                "step": step,
                "action": "stop",
                "planned_action": action,
                "planner_text": f"Action: {action}[{{}}]",
                "blocked": False,
                "reason": reason,
                "observation": observation,
            }
        )
        return current_sql or "", trace

    def try_repair(*, error: str, stop_action: str, stop_reason: str) -> bool:
        nonlocal current_sql, repairs_used
        if not cfg.use_repair_policy or repairs_used >= cfg.max_repairs or step >= cfg.max_steps:
            stop_with(
                action=stop_action,
                reason=stop_reason,
                observation={"sql": current_sql, "error": error},
            )
            return False

        next_step()
        current_sql = repair_sql(
            nlq,
            current_sql or "",
            error,
            schema_text,
        )
        repairs_used += 1
        add_trace(
            "repair_sql",
            observation={"sql": current_sql, "repairs_used": repairs_used},
            payload={"error": error},
        )
        return True

    try:
        next_step()
        add_trace(
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

        next_step()
        current_sql = generate_sql(nlq, schema_text)
        add_trace("generate_sql", observation={"sql": current_sql})

        while step < cfg.max_steps:
            next_step()
            sql_check = validate_sql(
                current_sql or "",
                schema_text,
                nlq=nlq,
            )
            add_trace("validate_sql", observation=sql_check, reason=None if sql_check.get("valid") else sql_check.get("reason"))
            if not sql_check.get("valid"):
                last_error = f"validate_sql:{sql_check.get('reason')}"
                if not try_repair(
                    error=last_error,
                    stop_action="validate_sql",
                    stop_reason="validation_failed",
                ):
                    return current_sql or "", trace
                continue

            last_error = None
            next_step()
            meta = ctx.runner.run(current_sql or "")
            run_obs = {
                "success": bool(meta.success),
                "rowcount": int(meta.rowcount),
                "error": meta.error,
                "sql": current_sql,
            }
            add_trace("run_sql", observation=run_obs, reason=None if meta.success else (meta.error or "run_sql_failed"))
            if meta.success:
                return stop_with(
                    action="run_sql",
                    reason="success",
                    observation={"sql": current_sql},
                )

            last_error = meta.error or "run_sql_failed"
            if not try_repair(
                error=last_error,
                stop_action="run_sql",
                stop_reason="execution_failed",
            ):
                return current_sql or "", trace

        trace.append(
            {
                "step": step,
                "action": "stop",
                "planned_action": None,
                "planner_text": "",
                "blocked": False,
                "reason": "max_steps_exhausted" if step >= cfg.max_steps else "repair_budget_exhausted",
                "observation": {"sql": current_sql},
            }
        )
        return current_sql or "", trace
    finally:
        _ACTIVE_CONFIG = None


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
        "timestamp": _now_utc_iso(),
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
