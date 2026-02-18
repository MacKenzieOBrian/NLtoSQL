"""
Module-level ReAct NL->SQL pipeline.

How to read this file:
1) `ReactAblationConfig` defines loop behavior switches.
2) `run_react_pipeline()` runs one NLQ through Thought->Action->Observation.
3) `evaluate_react_ablation()` runs the loop over a dataset and computes metrics.

References:
- ReAct paper: https://arxiv.org/abs/2210.03629
- JSON format docs (trace serialization): https://docs.python.org/3/library/json.html
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .agent_tools import (
    extract_constraints,
    generate_sql,
    get_agent_context,
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
from .prompts import REACT_SYSTEM_PROMPT
from .query_runner import QueryRunner
from .sql_guardrails import clean_candidate_with_reason


@dataclass(frozen=True)
class ReactAblationConfig:
    """Config for a model-driven ReAct tool loop."""

    name: str
    use_schema_link: bool
    use_constraint_policy: bool
    use_repair_policy: bool
    use_intent_gate: bool = False
    max_repairs: int = 2
    link_max_tables: int = 6
    max_steps: int = 8
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.2
    top_p: float = 0.9


def core_react_config(name: str = "react_core") -> ReactAblationConfig:
    """Paper-aligned default: model chooses actions, bounded by step budget."""
    return ReactAblationConfig(
        name=name,
        use_schema_link=True,
        use_constraint_policy=True,
        use_repair_policy=True,
        use_intent_gate=False,
        max_repairs=1,
        link_max_tables=6,
        max_steps=8,
    )


def default_ablation_plan() -> list[ReactAblationConfig]:
    """
    Recommended ablation path:
    1) core minimal loop (primary reporting default)
    2) no-repair variant (shows effect of execution-guided repair)
    3) extra-repair variant (tests complexity vs gain)
    """
    return [
        core_react_config(),
        ReactAblationConfig(
            name="react_no_repair",
            use_schema_link=True,
            use_constraint_policy=True,
            use_repair_policy=False,
            use_intent_gate=False,
            max_repairs=0,
            link_max_tables=6,
            max_steps=8,
        ),
        ReactAblationConfig(
            name="react_extra_repair",
            use_schema_link=True,
            use_constraint_policy=True,
            use_repair_policy=True,
            use_intent_gate=False,
            max_repairs=2,
            link_max_tables=6,
            max_steps=10,
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


def _call_react_llm(history: str, *, config: ReactAblationConfig) -> str:
    """
    Generate one ReAct step from transcript history.

    Uses the chat template directly so we preserve raw Thought/Action output
    instead of SQL extraction helpers.
    """
    import torch

    ctx = get_agent_context()
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": history},
    ]
    input_ids = ctx.tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(ctx.model.device)
    attention_mask = torch.ones_like(input_ids)

    pad_token_id = getattr(ctx.tok, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(ctx.tok, "eos_token_id", None)
    eos_token_id = getattr(ctx.tok, "eos_token_id", None)

    kwargs: dict[str, Any] = {
        "max_new_tokens": int(config.max_new_tokens),
        "do_sample": bool(config.do_sample),
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
    }
    if config.do_sample:
        kwargs["temperature"] = float(config.temperature)
        kwargs["top_p"] = float(config.top_p)

    with torch.no_grad():
        out = ctx.model.generate(
            input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    gen_ids = out[0][input_ids.shape[-1] :]
    return ctx.tok.decode(gen_ids, skip_special_tokens=True).strip()


_ACTION_RE = re.compile(
    r"^\s*Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\[(.*?)\]\s*$",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


def _normalize_llm_text(text: str) -> str:
    t = (text or "").replace("```json", "```").replace("```sql", "```")
    t = re.sub(r"```(.*?)```", r"\1", t, flags=re.DOTALL)
    return t.strip()


def _parse_action(text: str) -> tuple[str | None, dict[str, Any]]:
    """
    Parse the last Action block:
    Action: tool_name[json_args]
    """
    clean = _normalize_llm_text(text)
    matches = list(_ACTION_RE.finditer(clean))
    if not matches:
        return None, {}
    m = matches[-1]
    name = m.group(1).strip()
    raw_args = (m.group(2) or "").strip()
    if not raw_args:
        return name, {}
    try:
        parsed = json.loads(raw_args)
    except Exception:
        return name, {}
    return name, parsed if isinstance(parsed, dict) else {}


def _obs_to_history_text(obs: Any) -> str:
    if isinstance(obs, str):
        return obs
    return json.dumps(obs, ensure_ascii=False, default=str)


def _preview_rows(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    return rows[: max(limit, 0)]


def _coerce_candidate(raw: str | list[str]) -> str:
    if isinstance(raw, list):
        for c in raw:
            if isinstance(c, str) and c.strip():
                return c.strip()
        return ""
    return (raw or "").strip()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _dataset_signature(rows: list[dict[str, Any]]) -> str:
    """
    Stable dataset hash for reproducibility/audit.
    """
    canonical = [{"nlq": r.get("nlq"), "sql": r.get("sql")} for r in rows]
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def run_react_pipeline(
    *,
    nlq: str,
    config: ReactAblationConfig,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Run one NLQ through a model-driven ReAct Thought/Action/Observation loop.
    Returns: (pred_sql, trace)
    """
    trace: list[dict[str, Any]] = []
    history: list[str] = [f"User question: {nlq}"]

    def _stop_no_prediction(reason: str, **extra: Any) -> tuple[str, list[dict[str, Any]]]:
        payload: dict[str, Any] = {"step": len(trace), "action": "stop", "reason": reason}
        payload.update(extra)
        trace.append(payload)
        return "", trace

    schema_text_full = ""
    schema_text_focus = ""
    schema_tables: list[str] = []
    constraints: Optional[dict] = None
    last_sql: str | None = None
    last_error: str | None = None
    last_run: dict[str, Any] | None = None
    final_sql: str | None = None
    repair_count = 0
    for step in range(max(int(config.max_steps), 1)):
        llm_text = _call_react_llm("\n".join(history), config=config)
        action, args = _parse_action(llm_text)
        args = args if isinstance(args, dict) else {}
        history.append(llm_text.strip())

        if action is None:
            obs = {"error": "No Action found. Respond with Action: tool_name[json_args]."}
            history.append(f"Observation: {_obs_to_history_text(obs)}")
            trace.append({"step": step, "llm": llm_text, "action": None, "args": {}, "observation": obs, "blocked": True})
            continue

        obs: Any = None
        blocked = False

        if action == "finish":
            if last_run and last_run.get("success") and final_sql:
                obs = {"answer": str(last_run.get("rows", [])), "sql": final_sql}
                history.append(f"Observation: {_obs_to_history_text(obs)}")
                trace.append({"step": step, "llm": llm_text, "action": action, "args": args, "observation": obs, "blocked": False})
                return final_sql, trace
            obs = {"error": "finish called before successful run_sql"}
            blocked = True

        elif action == "get_schema":
            schema = get_schema()
            schema_text_full = schema_to_text(schema)
            if not schema_text_focus:
                schema_text_focus = schema_text_full
            schema_tables = [t["name"] for t in schema.get("tables", [])]
            obs = {
                "tables": schema_tables,
                "schema_text": schema_text_full,
            }

        elif action == "link_schema":
            if not config.use_schema_link:
                obs = {"error": "link_schema disabled by config"}
                blocked = True
            elif not schema_text_full:
                obs = {"error": "call get_schema before link_schema"}
                blocked = True
            else:
                try:
                    max_tables = int(args.get("max_tables", config.link_max_tables))
                except Exception:
                    max_tables = int(config.link_max_tables)
                linked = link_schema(nlq, schema_text_full, max_tables=max_tables)
                schema_text_focus = linked.get("schema_text") or schema_text_full
                obs = linked

        elif action == "extract_constraints":
            if not config.use_constraint_policy:
                obs = {"error": "extract_constraints disabled by config"}
                blocked = True
            else:
                constraints = extract_constraints(nlq)
                obs = constraints

        elif action == "generate_sql":
            if not schema_text_focus:
                obs = {"error": "call get_schema (and optionally link_schema) before generate_sql"}
                blocked = True
            else:
                raw = generate_sql(nlq, schema_text_focus, constraints or {})
                candidate = _coerce_candidate(raw)
                sql, clean_reason = _apply_guardrails(candidate, nlq, constraints)
                if not sql:
                    last_error = f"guardrail_reject:{clean_reason}"
                    obs = {
                        "error": last_error,
                        "raw_sql": candidate,
                    }
                else:
                    last_sql = sql
                    last_error = None
                    obs = {"sql": sql}

        elif action == "validate_sql":
            if not last_sql:
                obs = {"error": "No SQL to validate. Call generate_sql first."}
                blocked = True
            else:
                if not schema_text_full:
                    obs = {"error": "call get_schema before validate_sql"}
                    blocked = True
                else:
                    res = validate_sql(last_sql, schema_text_full)
                    obs = res
                    if not res.get("valid"):
                        last_error = res.get("reason", "validate_sql_failed")

        elif action == "validate_constraints":
            if not config.use_constraint_policy:
                obs = {"error": "validate_constraints disabled by config"}
                blocked = True
            elif not last_sql:
                obs = {"error": "No SQL to validate. Call generate_sql first."}
                blocked = True
            elif constraints is None:
                obs = {"error": "No constraints. Call extract_constraints first."}
                blocked = True
            else:
                res = validate_constraints(last_sql, constraints)
                obs = res
                if not res.get("valid"):
                    last_error = res.get("reason", "validate_constraints_failed")

        elif action == "run_sql":
            if not last_sql:
                obs = {"error": "No SQL to run. Call generate_sql first."}
                blocked = True
            else:
                run = run_sql(last_sql)
                if run.get("success") and config.use_intent_gate:
                    ok, why = intent_constraints(nlq, last_sql)
                    run["intent_ok"] = bool(ok)
                    if not ok:
                        run = {"success": False, "error": f"intent_mismatch:{why}"}
                if run.get("success"):
                    final_sql = last_sql
                    last_error = None
                    last_run = run
                    # Keep observations concise to avoid transcript bloat.
                    rows = run.get("rows") or []
                    obs = dict(run)
                    obs["rows"] = _preview_rows(rows, limit=5)
                else:
                    last_run = run
                    last_error = run.get("error", "execution_failed")
                    obs = run

        elif action == "repair_sql":
            if not config.use_repair_policy:
                obs = {"error": "repair_sql disabled by config"}
                blocked = True
            elif not last_sql:
                obs = {"error": "No SQL to repair. Call generate_sql first."}
                blocked = True
            elif repair_count >= int(config.max_repairs):
                return _stop_no_prediction(
                    "repair_budget_exhausted",
                    final_sql=last_sql,
                    last_error=last_error,
                )
            else:
                err = str(args.get("error") or last_error or "repair_requested")
                schema_ref = schema_text_full or schema_text_focus
                repaired = repair_sql(
                    nlq,
                    last_sql,
                    err,
                    schema_ref,
                )
                candidate = _coerce_candidate(repaired)
                sql, clean_reason = _apply_guardrails(candidate, nlq, constraints)
                repair_count += 1
                if not sql:
                    last_error = f"guardrail_reject:{clean_reason}"
                    obs = {
                        "error": last_error,
                        "raw_sql": candidate,
                        "repair_count": repair_count,
                    }
                else:
                    last_sql = sql
                    last_error = None
                    obs = {"sql": sql, "repair_count": repair_count}

        else:
            obs = {"error": f"Unknown action: {action}"}
            blocked = True

        history.append(f"Observation: {_obs_to_history_text(obs)}")
        trace.append(
            {
                "step": step,
                "llm": llm_text,
                "action": action,
                "args": args,
                "observation": obs,
                "blocked": blocked,
            }
        )

    if final_sql:
        trace.append({"step": int(config.max_steps), "action": "stop", "reason": "max_steps_after_success", "sql": final_sql})
        return final_sql, trace
    return _stop_no_prediction(
        "max_steps_no_success",
        tables=schema_tables,
        final_sql=last_sql,
        last_error=last_error,
    )


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
    seed: Optional[int] = None,
    run_metadata: Optional[dict[str, Any]] = None,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Evaluate one ablation config over a test set and return metrics + item logs.
    """
    rows = test_set[:limit] if limit else test_set
    if seed is not None:
        random.seed(seed)
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

    report: dict[str, Any] = {
        "timestamp": _now_utc_iso(),
        "seed": seed,
        "limit": limit,
        "dataset_signature": _dataset_signature(rows),
        "config": config.name,
        "config_snapshot": asdict(config),
        "n": len(rows),
        "va_rate": va_hits / n,
        "em_rate": em_hits / n,
        "ex_rate": ex_hits / n,
        "ts_rate": (sum(int(v) for v in ts_values) / max(len(ts_values), 1)) if ts_values else None,
        "items": items,
    }
    if run_metadata:
        report["run_metadata"] = run_metadata

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"[{config.name}] Saved report: {out}")

    return report


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
    seed: Optional[int] = None,
    run_metadata: Optional[dict[str, Any]] = None,
    save_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Run the default cumulative ablation plan and return one report per stage.
    """
    reports: list[dict[str, Any]] = []
    for cfg in default_ablation_plan():
        cfg_save_path: Path | None = None
        if save_dir is not None:
            cfg_save_path = Path(save_dir) / f"{cfg.name}.json"
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
            seed=seed,
            run_metadata=run_metadata,
            save_path=cfg_save_path,
        )
        reports.append(report)
        ts_s = "NA" if report["ts_rate"] is None else f"{report['ts_rate']:.3f}"
        print(
            f"{cfg.name}: VA={report['va_rate']:.3f} EM={report['em_rate']:.3f} "
            f"EX={report['ex_rate']:.3f} TS={ts_s}"
        )
    return reports
