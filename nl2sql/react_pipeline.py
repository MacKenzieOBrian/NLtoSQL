"""
Canonical, module-level ReAct-style NL->SQL pipeline.

This module freezes the tool order in code so notebooks only call module APIs.
It also provides cumulative ablations:
1) baseline
2) +schema link
3) +constraint policy
4) +repair policy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .agent_tools import (
    extract_constraints,
    generate_sql,
    get_schema,
    link_schema,
    repair_sql,
    run_sql,
    schema_to_text,
    validate_constraints,
    validate_sql,
)
from .eval import execution_accuracy, test_suite_accuracy_for_item
from .intent_rules import intent_constraints
from .postprocess import guarded_postprocess, normalize_sql
from .query_runner import QueryRunner
from .sql_guardrails import clean_candidate_with_reason


@dataclass(frozen=True)
class ReactAblationConfig:
    name: str
    use_schema_link: bool
    use_constraint_policy: bool
    use_repair_policy: bool
    max_repairs: int = 2
    link_max_tables: int = 6


def default_ablation_plan() -> list[ReactAblationConfig]:
    """Cumulative ablation path used in dissertation experiments."""
    return [
        ReactAblationConfig(
            name="baseline",
            use_schema_link=False,
            use_constraint_policy=False,
            use_repair_policy=False,
        ),
        ReactAblationConfig(
            name="plus_schema_link",
            use_schema_link=True,
            use_constraint_policy=False,
            use_repair_policy=False,
        ),
        ReactAblationConfig(
            name="plus_constraint_policy",
            use_schema_link=True,
            use_constraint_policy=True,
            use_repair_policy=False,
        ),
        ReactAblationConfig(
            name="plus_repair_policy",
            use_schema_link=True,
            use_constraint_policy=True,
            use_repair_policy=True,
        ),
    ]


def _apply_guardrails(raw_sql: str, nlq: str, constraints: Optional[dict]) -> tuple[str, str]:
    """
    Apply deterministic SQL cleanup:
    1) extract single SELECT candidate
    2) apply postprocess projection/order cleanup
    """
    cleaned, reason = clean_candidate_with_reason(raw_sql)
    if not cleaned:
        return "", reason
    explicit_fields = (constraints or {}).get("explicit_fields")
    explicit_projection = (constraints or {}).get("explicit_projection")
    required_fields = (constraints or {}).get("required_output_fields")
    sql = guarded_postprocess(
        cleaned,
        nlq,
        explicit_fields=explicit_fields if explicit_projection else None,
        required_fields=required_fields,
    )
    return sql, "ok"


def _coerce_candidate(raw: str | list[str]) -> str:
    if isinstance(raw, list):
        for c in raw:
            if isinstance(c, str) and c.strip():
                return c.strip()
        return ""
    return (raw or "").strip()


def run_react_pipeline(
    *,
    nlq: str,
    config: ReactAblationConfig,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Run one NLQ through the canonical, fixed-order tool pipeline.
    Returns: (pred_sql, trace)
    """
    trace: list[dict[str, Any]] = []

    schema = get_schema()
    schema_text_full = schema_to_text(schema)
    schema_text_focus = schema_text_full

    trace.append({"stage": "get_schema", "tables": [t["name"] for t in schema.get("tables", [])]})

    if config.use_schema_link:
        linked = link_schema(nlq, schema_text_full, max_tables=config.link_max_tables)
        schema_text_focus = linked.get("schema_text") or schema_text_full
        trace.append({"stage": "link_schema", "changed": linked.get("changed"), "link_debug": linked.get("link_debug")})
    else:
        trace.append({"stage": "link_schema", "skipped": True})

    constraints: Optional[dict] = None
    if config.use_constraint_policy:
        constraints = extract_constraints(nlq)
        trace.append({"stage": "extract_constraints", "constraints": constraints})
    else:
        trace.append({"stage": "extract_constraints", "skipped": True})

    gen_constraints = constraints or {}
    raw = generate_sql(nlq, schema_text_focus, gen_constraints)
    candidate = _coerce_candidate(raw)
    sql, clean_reason = _apply_guardrails(candidate, nlq, constraints)
    trace.append(
        {
            "stage": "generate_sql",
            "raw_sql": candidate,
            "sql_after_guardrails": sql,
            "guardrail_reason": clean_reason,
        }
    )
    if not sql:
        return "", trace

    repair_count = 0
    while True:
        v_sql = validate_sql(sql, schema_text_full)
        trace.append({"stage": "validate_sql", "sql": sql, "result": v_sql})
        if not v_sql.get("valid"):
            if not config.use_repair_policy or repair_count >= config.max_repairs:
                return sql, trace
            repaired = repair_sql(nlq, sql, v_sql.get("reason", "validate_sql_failed"), schema_text_full)
            repair_raw = _coerce_candidate(repaired)
            sql, clean_reason = _apply_guardrails(repair_raw, nlq, constraints)
            repair_count += 1
            trace.append(
                {
                    "stage": "repair_sql",
                    "reason": v_sql.get("reason", "validate_sql_failed"),
                    "raw_sql": repair_raw,
                    "sql_after_guardrails": sql,
                    "guardrail_reason": clean_reason,
                    "repair_count": repair_count,
                }
            )
            if not sql:
                return "", trace
            continue

        if constraints is not None:
            v_c = validate_constraints(sql, constraints)
            trace.append({"stage": "validate_constraints", "sql": sql, "result": v_c})
            if not v_c.get("valid"):
                if not config.use_repair_policy or repair_count >= config.max_repairs:
                    return sql, trace
                repaired = repair_sql(nlq, sql, v_c.get("reason", "validate_constraints_failed"), schema_text_full)
                repair_raw = _coerce_candidate(repaired)
                sql, clean_reason = _apply_guardrails(repair_raw, nlq, constraints)
                repair_count += 1
                trace.append(
                    {
                        "stage": "repair_sql",
                        "reason": v_c.get("reason", "validate_constraints_failed"),
                        "raw_sql": repair_raw,
                        "sql_after_guardrails": sql,
                        "guardrail_reason": clean_reason,
                        "repair_count": repair_count,
                    }
                )
                if not sql:
                    return "", trace
                continue
        else:
            trace.append({"stage": "validate_constraints", "skipped": True})

        run = run_sql(sql)
        trace.append({"stage": "run_sql", "sql": sql, "result": run})
        if run.get("success"):
            ok, why = intent_constraints(nlq, sql)
            trace.append({"stage": "intent_check", "ok": ok, "reason": why})
            if ok:
                return sql, trace
            if not config.use_repair_policy or repair_count >= config.max_repairs:
                return sql, trace
            repaired = repair_sql(nlq, sql, f"intent_mismatch:{why}", schema_text_full)
            repair_raw = _coerce_candidate(repaired)
            sql, clean_reason = _apply_guardrails(repair_raw, nlq, constraints)
            repair_count += 1
            trace.append(
                {
                    "stage": "repair_sql",
                    "reason": f"intent_mismatch:{why}",
                    "raw_sql": repair_raw,
                    "sql_after_guardrails": sql,
                    "guardrail_reason": clean_reason,
                    "repair_count": repair_count,
                }
            )
            if not sql:
                return "", trace
            continue

        if not config.use_repair_policy or repair_count >= config.max_repairs:
            return sql, trace
        repaired = repair_sql(nlq, sql, run.get("error", "execution_failed"), schema_text_full)
        repair_raw = _coerce_candidate(repaired)
        sql, clean_reason = _apply_guardrails(repair_raw, nlq, constraints)
        repair_count += 1
        trace.append(
            {
                "stage": "repair_sql",
                "reason": run.get("error", "execution_failed"),
                "raw_sql": repair_raw,
                "sql_after_guardrails": sql,
                "guardrail_reason": clean_reason,
                "repair_count": repair_count,
            }
        )
        if not sql:
            return "", trace


def evaluate_react_ablation(
    *,
    test_set: list[dict[str, Any]],
    engine: Any,
    config: ReactAblationConfig,
    limit: Optional[int] = None,
    allow_extra_columns_ex: bool = False,
    ts_suite_db_names: Optional[list[str]] = None,
    ts_make_engine_fn: Optional[Callable[[str], Any]] = None,
    ts_max_rows: int = 500,
    progress_every: int = 20,
) -> dict[str, Any]:
    """
    Evaluate one ablation config over a test set and return metrics + item logs.
    """
    rows = test_set[:limit] if limit else test_set
    qr = QueryRunner(engine, max_rows=50)
    items: list[dict[str, Any]] = []

    va_hits = 0
    em_hits = 0
    ex_hits = 0

    for i, sample in enumerate(rows):
        nlq = sample["nlq"]
        gold_sql = sample["sql"]
        pred_sql, trace = run_react_pipeline(nlq=nlq, config=config)

        if pred_sql:
            meta = qr.run(pred_sql, capture_df=False)
            va = bool(meta.success)
            error = meta.error
        else:
            va = False
            error = "no_prediction"

        em = normalize_sql(pred_sql or "") == normalize_sql(gold_sql or "")
        if pred_sql:
            ex, pred_err, gold_err = execution_accuracy(
                engine=engine,
                pred_sql=pred_sql,
                gold_sql=gold_sql,
                allow_extra_columns=allow_extra_columns_ex,
            )
        else:
            ex, pred_err, gold_err = False, "no_prediction", None
        ts = None
        ts_debug = None
        if ts_suite_db_names and ts_make_engine_fn and pred_sql:
            ts, ts_debug = test_suite_accuracy_for_item(
                make_engine_fn=ts_make_engine_fn,
                suite_db_names=ts_suite_db_names,
                gold_sql=gold_sql,
                pred_sql=pred_sql,
                max_rows=ts_max_rows,
                strict_gold=True,
            )

        va_hits += int(va)
        em_hits += int(em)
        ex_hits += int(ex)
        items.append(
            {
                "i": i,
                "nlq": nlq,
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
                "va": int(va),
                "em": int(em),
                "ex": int(ex),
                "ts": ts,
                "error": error or pred_err,
                "gold_error": gold_err,
                "trace": trace,
                "ts_debug": ts_debug,
            }
        )

        if progress_every and (i + 1) % progress_every == 0:
            print(f"[{config.name}] Processed {i + 1}/{len(rows)}")

    n = max(len(rows), 1)
    ts_values = [r["ts"] for r in items if r["ts"] is not None]

    return {
        "config": config.name,
        "n": len(rows),
        "va_rate": va_hits / n,
        "em_rate": em_hits / n,
        "ex_rate": ex_hits / n,
        "ts_rate": (sum(int(v) for v in ts_values) / max(len(ts_values), 1)) if ts_values else None,
        "items": items,
    }


def run_ablation_suite(
    *,
    test_set: list[dict[str, Any]],
    engine: Any,
    limit: Optional[int] = None,
    allow_extra_columns_ex: bool = False,
    ts_suite_db_names: Optional[list[str]] = None,
    ts_make_engine_fn: Optional[Callable[[str], Any]] = None,
    ts_max_rows: int = 500,
    progress_every: int = 20,
) -> list[dict[str, Any]]:
    """
    Run the default cumulative ablation plan and return one report per stage.
    """
    reports: list[dict[str, Any]] = []
    for cfg in default_ablation_plan():
        report = evaluate_react_ablation(
            test_set=test_set,
            engine=engine,
            config=cfg,
            limit=limit,
            allow_extra_columns_ex=allow_extra_columns_ex,
            ts_suite_db_names=ts_suite_db_names,
            ts_make_engine_fn=ts_make_engine_fn,
            ts_max_rows=ts_max_rows,
            progress_every=progress_every,
        )
        reports.append(report)
        ts_s = "NA" if report["ts_rate"] is None else f"{report['ts_rate']:.3f}"
        print(
            f"{cfg.name}: VA={report['va_rate']:.3f} EM={report['em_rate']:.3f} "
            f"EX={report['ex_rate']:.3f} TS={ts_s}"
        )
    return reports
