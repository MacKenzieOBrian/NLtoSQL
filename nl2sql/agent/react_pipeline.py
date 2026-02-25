"""
Lightweight ReAct loop for notebook-driven NL->SQL experiments.

Design goals:
1) Keep a clear, inspectable action loop for dissertation extension runs.
2) Reuse existing project validators/executors for consistency.
3) Keep outputs compatible with existing notebook result handling.

References (project anchors):
- `REFERENCES.md#ref-yao2023-react`
- `REFERENCES.md#ref-pourreza2023-dinsql`
- `REFERENCES.md#ref-zhai2025-excot`
- `REFERENCES.md#ref-yu2018-spider`
- `REFERENCES.md#ref-zhong2020-ts`
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from sqlalchemy.engine import Engine

from ..core.llm import extract_first_select, generate_sql_from_messages
from ..core.postprocess import normalize_sql
from ..core.validation import validate_constraints, validate_sql
from ..evaluation.eval import execution_accuracy, test_suite_accuracy_for_item
from .agent_schema_linking import build_schema_subset
from .agent_tools import ensure_schema_text, get_agent_context
from .constraint_policy import build_constraints
from .intent_rules import intent_constraints
from .prompts import (
    REACT_SYSTEM_PROMPT,
    SQL_GENERATOR_SYSTEM_PROMPT,
    SQL_REPAIR_SYSTEM_PROMPT,
)


@dataclass(frozen=True)
class ReactAblationConfig:
    name: str = "react_core"
    use_schema_link: bool = True
    use_constraint_policy: bool = True
    use_repair_policy: bool = True
    use_intent_gate: bool = False
    max_repairs: int = 1
    link_max_tables: int = 6
    max_steps: int = 8
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.2
    top_p: float = 0.9


def core_react_config(name: str = "react_core") -> ReactAblationConfig:
    return ReactAblationConfig(name=name)


_ACTION_RE = re.compile(r"(?is)action\s*:\s*([a-z_]+)\s*(?:\[(.*)\])?")
_ACTIVE_CONFIG: ReactAblationConfig | None = None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_action(text: str) -> tuple[str | None, dict[str, Any]]:
    s = (text or "").strip()
    m = _ACTION_RE.search(s)
    if not m:
        return None, {}

    action = (m.group(1) or "").strip().lower()
    payload_raw = (m.group(2) or "").strip()
    if not payload_raw:
        return action, {}
    try:
        payload = json.loads(payload_raw)
        if not isinstance(payload, dict):
            return action, {}
        return action, payload
    except Exception:
        return action, {}


def _history_text(nlq: str, trace: list[dict[str, Any]]) -> str:
    lines = [f"Question: {nlq}"]
    for t in trace[-8:]:
        action = t.get("action")
        obs = t.get("observation")
        lines.append(f"Step {t.get('step')}: Action={action}")
        if isinstance(obs, dict):
            if "error" in obs and obs["error"]:
                lines.append(f"Observation: error={obs['error']}")
            elif "sql" in obs and obs["sql"]:
                lines.append(f"Observation: sql={obs['sql']}")
            elif "success" in obs:
                lines.append(f"Observation: success={obs['success']}")
            elif "reason" in obs and obs["reason"]:
                lines.append(f"Observation: reason={obs['reason']}")
    return "\n".join(lines)


def _call_react_llm(history: str, *, config: ReactAblationConfig) -> str:
    """
    Planner call that emits one action.
    Kept small and deterministic to make traces auditable.
    """
    ctx = get_agent_context()
    if ctx.model is None or ctx.tok is None:
        raise RuntimeError("Agent context model/tokenizer not set for ReAct planner calls.")

    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                history
                + "\n\nReturn one action in this exact format:\n"
                + "Action: <name>[<json>]\n"
                + "Use {} when there are no arguments."
            ),
        },
    ]
    out = generate_sql_from_messages(
        model=ctx.model,
        tokenizer=ctx.tok,
        messages=messages,
        max_new_tokens=min(config.max_new_tokens, 128),
        do_sample=config.do_sample,
        temperature=config.temperature,
        top_p=config.top_p,
        constrained=False,
        extract_select=False,
        stop_on_semicolon=False,
    )
    return str(out or "").strip()


def _effective_config() -> ReactAblationConfig:
    return _ACTIVE_CONFIG or core_react_config()


def generate_sql(nlq: str, schema_text: str, constraints: dict[str, Any]) -> str:
    """
    Generate one SQL candidate for the NLQ.
    Signature is intentionally stable for notebook monkeypatch demos.
    """
    ctx = get_agent_context()
    config = _effective_config()
    if ctx.model is None or ctx.tok is None:
        raise RuntimeError("Agent context model/tokenizer not set for SQL generation.")

    constraint_text = json.dumps(constraints or {}, ensure_ascii=True)
    messages = [
        {"role": "system", "content": SQL_GENERATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Schema Details:\n{schema_text}\n\n"
                f"Constraint Hints (JSON):\n{constraint_text}\n\n"
                f"Natural Language Question: {nlq}"
            ),
        },
    ]
    out = generate_sql_from_messages(
        model=ctx.model,
        tokenizer=ctx.tok,
        messages=messages,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        temperature=config.temperature,
        top_p=config.top_p,
        constrained=False,
        extract_select=True,
        stop_on_semicolon=True,
    )
    sql = extract_first_select(str(out)) or str(out or "").strip()
    if sql and not sql.endswith(";"):
        sql += ";"
    return sql


def repair_sql(nlq: str, bad_sql: str, error: str, schema_text: str) -> str:
    """
    Repair SQL from validator/runtime feedback.
    Signature is intentionally stable for notebook monkeypatch demos.
    """
    ctx = get_agent_context()
    config = _effective_config()
    if ctx.model is None or ctx.tok is None:
        raise RuntimeError("Agent context model/tokenizer not set for SQL repair.")

    messages = [
        {"role": "system", "content": SQL_REPAIR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Schema Details:\n{schema_text}\n\n"
                f"Natural Language Question: {nlq}\n\n"
                f"Previous SQL:\n{bad_sql}\n\n"
                f"Observed Error:\n{error}\n\n"
                "Return one corrected SQL SELECT."
            ),
        },
    ]
    out = generate_sql_from_messages(
        model=ctx.model,
        tokenizer=ctx.tok,
        messages=messages,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        temperature=config.temperature,
        top_p=config.top_p,
        constrained=False,
        extract_select=True,
        stop_on_semicolon=True,
    )
    sql = extract_first_select(str(out)) or str(out or "").strip()
    if sql and not sql.endswith(";"):
        sql += ";"
    return sql


def run_react_pipeline(
    *,
    nlq: str,
    config: ReactAblationConfig | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Execute one ReAct loop for a single NLQ and return (final_sql, trace).
    """
    global _ACTIVE_CONFIG
    cfg = config or core_react_config()
    _ACTIVE_CONFIG = cfg
    ctx = get_agent_context()

    trace: list[dict[str, Any]] = []
    schema_full = ensure_schema_text(ctx)
    schema_working: str | None = None
    constraints: dict[str, Any] | None = None
    current_sql: str | None = None
    last_error: str | None = None
    repairs_used = 0
    sql_validated = False
    constraints_validated = False
    intent_validated = False

    try:
        for step in range(1, cfg.max_steps + 1):
            history = _history_text(nlq, trace)
            planner_text = ""
            planned_action = None
            payload: dict[str, Any] = {}
            try:
                planner_text = _call_react_llm(history, config=cfg)
                planned_action, payload = _parse_action(planner_text)
            except Exception as e:
                trace.append(
                    {
                        "step": step,
                        "action": "stop",
                        "planned_action": None,
                        "planner_text": planner_text,
                        "blocked": False,
                        "reason": "planner_error",
                        "observation": {"error": str(e)},
                    }
                )
                break

            # ReAct-faithful mode: planner action is mandatory.
            # If planner output is malformed, stop and record the failure.
            if not planned_action:
                trace.append(
                    {
                        "step": step,
                        "action": "stop",
                        "planned_action": None,
                        "planner_text": planner_text,
                        "blocked": False,
                        "reason": "invalid_action_format",
                        "observation": {"expected_format": "Action: <name>[<json>]"},
                    }
                )
                break

            action = planned_action

            blocked = False
            reason: str | None = None
            obs: dict[str, Any] = {}

            if action == "finish":
                trace.append(
                    {
                        "step": step,
                        "action": "stop",
                        "planned_action": planned_action,
                        "planner_text": planner_text,
                        "blocked": False,
                        "reason": "finish_action",
                        "observation": {"sql": current_sql},
                    }
                )
                break

            if action == "get_schema":
                if cfg.use_schema_link:
                    schema_working = build_schema_subset(
                        schema_summary=schema_full,
                        nlq=nlq,
                        max_tables=cfg.link_max_tables,
                        max_cols_per_table=8,
                    )
                else:
                    schema_working = schema_full
                obs = {
                    "schema_lines": len((schema_working or "").splitlines()),
                    "tables": [ln.split("(")[0].strip() for ln in (schema_working or "").splitlines() if "(" in ln][:10],
                }

            elif action == "extract_constraints":
                if cfg.use_constraint_policy:
                    constraints = build_constraints(nlq, schema_working or schema_full)
                else:
                    constraints = {}
                obs = {
                    "rule_tags": (constraints or {}).get("rule_tags", []),
                    "explicit_fields": (constraints or {}).get("explicit_fields", []),
                }

            elif action == "generate_sql":
                constraints = constraints if constraints is not None else {}
                current_sql = generate_sql(nlq, schema_working or schema_full, constraints)
                last_error = None
                sql_validated = False
                constraints_validated = False
                intent_validated = False
                obs = {"sql": current_sql}

            elif action == "repair_sql":
                if not cfg.use_repair_policy:
                    blocked = True
                    reason = "repair_policy_disabled"
                elif not current_sql:
                    blocked = True
                    reason = "no_sql_to_repair"
                elif repairs_used >= cfg.max_repairs:
                    blocked = True
                    reason = "repair_budget_exhausted"
                else:
                    current_sql = repair_sql(
                        nlq,
                        current_sql,
                        payload.get("error") or last_error or "unknown_error",
                        schema_working or schema_full,
                    )
                    repairs_used += 1
                    last_error = None
                    sql_validated = False
                    constraints_validated = False
                    intent_validated = False
                    obs = {"sql": current_sql, "repairs_used": repairs_used}

            elif action == "validate_sql":
                if not current_sql:
                    blocked = True
                    reason = "no_sql_to_validate"
                else:
                    v = validate_sql(current_sql, schema_working or schema_full)
                    obs = v
                    sql_validated = bool(v.get("valid"))
                    if not sql_validated:
                        last_error = f"validate_sql:{v.get('reason')}"

            elif action == "validate_constraints":
                if not current_sql:
                    blocked = True
                    reason = "no_sql_for_constraints"
                else:
                    v = validate_constraints(
                        current_sql,
                        constraints or {},
                        schema_text=schema_working or schema_full,
                    )
                    obs = v
                    constraints_validated = bool(v.get("valid"))
                    if not constraints_validated:
                        last_error = f"validate_constraints:{v.get('reason')}"

            elif action == "intent_check":
                if not current_sql:
                    blocked = True
                    reason = "no_sql_for_intent_check"
                elif not cfg.use_intent_gate:
                    intent_validated = True
                    obs = {"valid": True, "reason": "intent_gate_disabled"}
                else:
                    ok, why = intent_constraints(nlq, current_sql)
                    obs = {"valid": bool(ok), "reason": why}
                    intent_validated = bool(ok)
                    if not ok:
                        last_error = f"intent_mismatch:{why}"

            elif action == "run_sql":
                if not current_sql:
                    blocked = True
                    reason = "no_sql_to_run"
                else:
                    meta = ctx.runner.run(current_sql, capture_df=False)
                    obs = {
                        "success": bool(meta.success),
                        "rowcount": int(meta.rowcount),
                        "error": meta.error,
                        "sql": current_sql,
                    }
                    if meta.success:
                        trace.append(
                            {
                                "step": step,
                                "action": action,
                                "planned_action": planned_action,
                                "planner_text": planner_text,
                                "payload": payload,
                                "blocked": False,
                                "reason": None,
                                "observation": obs,
                            }
                        )
                        trace.append(
                            {
                                "step": step,
                                "action": "stop",
                                "planned_action": planned_action,
                                "planner_text": planner_text,
                                "blocked": False,
                                "reason": "success",
                                "observation": {"sql": current_sql},
                            }
                        )
                        return current_sql, trace
                    last_error = meta.error or "run_sql_failed"

            else:
                trace.append(
                    {
                        "step": step,
                        "action": "stop",
                        "planned_action": planned_action,
                        "planner_text": planner_text,
                        "payload": payload,
                        "blocked": False,
                        "reason": f"unknown_action:{action}",
                        "observation": {},
                    }
                )
                break

            if blocked and not reason:
                reason = "blocked"
            trace.append(
                {
                    "step": step,
                    "action": action,
                    "planned_action": planned_action,
                    "planner_text": planner_text,
                    "payload": payload,
                    "blocked": bool(blocked),
                    "reason": reason,
                    "observation": obs,
                }
            )

        if trace and trace[-1].get("action") == "stop":
            return current_sql or "", trace

        trace.append(
            {
                "step": cfg.max_steps,
                "action": "stop",
                "planned_action": None,
                "planner_text": "",
                "blocked": False,
                "reason": "max_steps_exhausted",
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
            meta = ctx.runner.run(pred_sql, capture_df=False)
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
            allow_extra_columns=False,
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
            ts, _ = test_suite_accuracy_for_item(
                make_engine_fn=ts_make_engine_fn,
                suite_db_names=ts_suite_db_names,
                gold_sql=gold_sql,
                pred_sql=pred_sql,
                max_rows=ts_max_rows,
                strict_gold=True,
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
    "_call_react_llm",
]
