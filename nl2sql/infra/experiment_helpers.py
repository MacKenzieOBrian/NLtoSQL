"""Notebook orchestration helpers shared across the experiment notebooks."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from sqlalchemy.engine import Engine

from ..evaluation.eval import EVAL_PROFILE_MODEL_ONLY_RAW


def model_alias_from_id(model_id: str) -> str:
    """Convert a model id into a filesystem-safe alias."""
    tail = (model_id or "model").split("/")[-1]
    alias = re.sub(r"[^a-z0-9]+", "_", tail.lower()).strip("_")
    return alias or "model"


def git_short_commit(default: str = "unknown") -> str:
    """Return the current short git hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return default


def build_run_metadata(
    *,
    model_id: str,
    model_alias: str,
    method: str,
    notebook: str,
    adapter_dir: str | None = None,
    commit: str | None = None,
) -> dict[str, Any]:
    """Build consistent run metadata payloads for JSON result files."""
    payload: dict[str, Any] = {
        "commit": commit or git_short_commit(),
        "model_id": model_id,
        "model_alias": model_alias,
        "notebook": notebook,
        "method": method,
    }
    if adapter_dir:
        payload["adapter_dir"] = adapter_dir
    return payload


QLORA_EXPERIMENT_PRESETS: dict[str, dict[str, Any]] = {
    "llama3_8b": {
        "label": "Llama-3-8B QLoRA",
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "adapter_output_dir": "results/adapters/qlora_llama3_8b_classicmodels",
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "train_batch_size": 1,
        "grad_accum_steps": 8,
        "learning_rate": 1e-4,
        "num_train_epochs": 3,
        "warmup_ratio": 0.05,
        "max_seq_length": 1024,
        "save_steps": 200,
        "save_total_limit": 2,
    },
    "qwen2_5_7b": {
        "label": "Qwen2.5-7B QLoRA",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "adapter_output_dir": "results/adapters/qlora_qwen2_5_7b_classicmodels",
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "train_batch_size": 1,
        "grad_accum_steps": 8,
        "learning_rate": 1e-4,
        "num_train_epochs": 3,
        "warmup_ratio": 0.05,
        "max_seq_length": 1024,
        "save_steps": 200,
        "save_total_limit": 2,
    },
}


@dataclass(frozen=True)
class GridPlan:
    """Small notebook-level plan for one baseline/QLoRA rerun."""

    eval_profile: str = EVAL_PROFILE_MODEL_ONLY_RAW
    k_values: list[int] = field(default_factory=lambda: [0, 3])
    seeds: list[int] = field(default_factory=lambda: [7, 17, 27])
    enable_ts: bool = True
    ts_for_k_values: list[int] = field(default_factory=lambda: [3])
    ts_n: int = 10
    ts_prefix: str = "classicmodels_ts"
    ts_max_rows: int = 500
    max_new_tokens: int | None = None


def primary_grid_plan(*, max_new_tokens: int | None = None) -> GridPlan:
    """Return the default dissertation rerun settings."""
    return GridPlan(max_new_tokens=max_new_tokens)


def print_grid_plan(run_tag: str, plan: GridPlan, *, extra: str | None = None) -> None:
    """Print one short line that explains the grid run settings."""
    suffix = f" | {extra}" if extra else ""
    print(
        f"Run: {run_tag} | profile={plan.eval_profile} | "
        f"k={plan.k_values} | seeds={plan.seeds} | "
        f"TS_k={plan.ts_for_k_values if plan.enable_ts else 'off'}{suffix}"
    )


def print_grid_run_report(report: dict[str, Any]) -> None:
    """Print the saved paths and simple per-run rows for a grid report."""
    print("Saved run dir:", report["run_dir"])
    print("Saved run report:", report["report_path"])
    print("Per-run metric rows:")
    for row in report["runs"]:
        print(row)


@dataclass(frozen=True)
class ReactEvalPlan:
    """Small notebook-level plan for one ReAct rerun."""

    quick_limit: int | None = None
    ts_n: int = 10
    ts_prefix: str = "classicmodels_ts"
    ts_max_rows: int = 500


def primary_react_eval_plan(
    *,
    quick_limit: int | None = None,
    ts_n: int = 10,
    ts_prefix: str = "classicmodels_ts",
    ts_max_rows: int = 500,
) -> ReactEvalPlan:
    """Return the default ReAct notebook run settings."""
    return ReactEvalPlan(
        quick_limit=quick_limit,
        ts_n=ts_n,
        ts_prefix=ts_prefix,
        ts_max_rows=ts_max_rows,
    )


def print_react_plan(config_name: str, plan: ReactEvalPlan) -> None:
    """Print one short line that explains the ReAct notebook settings."""
    print({
        "config": config_name,
        "quick_limit": plan.quick_limit,
        "ts_n": plan.ts_n,
        "ts_prefix": plan.ts_prefix,
        "ts_max_rows": plan.ts_max_rows,
    })


def print_react_run_report(label: str, report: dict[str, Any], out_path: Path) -> None:
    """Print the compact metric summary and saved path for a ReAct run."""
    print_eval_rates(label, report)
    print("Saved report:", out_path)


def configure_react_notebook(
    *,
    engine: Any,
    db_name: str,
    model: Any,
    tokenizer: Any,
    exemplar_pool: list[dict[str, Any]],
    name: str = "react_barebones_notebook",
    use_repair_policy: bool = True,
    max_repairs: int = 2,
    max_steps: int = 8,
    few_shot_k: int = 3,
    few_shot_seed: int = 7,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> Any:
    """Set the shared agent context and return the matching ReAct config."""
    from nl2sql.agent.agent_tools import AgentContext, set_agent_context
    from nl2sql.agent.react_pipeline import ReactAblationConfig
    from nl2sql.core.query_runner import QueryRunner

    set_agent_context(
        AgentContext(
            engine=engine,
            db_name=db_name,
            model=model,
            tok=tokenizer,
            runner=QueryRunner(engine),
            exemplar_pool=exemplar_pool,
        )
    )
    return ReactAblationConfig(
        name=name,
        use_repair_policy=use_repair_policy,
        max_repairs=max_repairs,
        max_steps=max_steps,
        few_shot_k=few_shot_k,
        few_shot_seed=few_shot_seed,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )


def train_qlora_adapter(
    *,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    experiment_config: dict[str, Any],
    output_dir: str | Path,
    use_bf16: bool,
) -> dict[str, Any]:
    """Train one QLoRA adapter and save a simple run card."""
    from trl import SFTTrainer
    from transformers import TrainingArguments

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = experiment_config
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg["train_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=10,
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        bf16=use_bf16,
        fp16=(not use_bf16),
        optim="paged_adamw_8bit",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        args=training_args,
        max_seq_length=cfg["max_seq_length"],
    )
    trainer.train()
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    run_card = {
        **cfg,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "precision": "bf16" if use_bf16 else "fp16",
    }
    (output_dir / "run_card.json").write_text(json.dumps(run_card, indent=2), encoding="utf-8")
    return run_card


def run_model_grid_notebook_eval(
    *,
    test_set: list[dict[str, Any]],
    schema_summary: str,
    model: Any,
    tokenizer: Any,
    engine: Any,
    db_config: dict[str, Any] | None = None,
    instance_connection_name: str | None = None,
    db_user: str | None = None,
    db_pass: str | None = None,
    model_id: str,
    method: str,
    notebook: str,
    run_tag: str,
    runs_dir: str,
    grid_plan: GridPlan,
    model_alias: str | None = None,
    adapter_dir: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the shared eval grid and return a plain run report."""
    from nl2sql.evaluation.grid_runner import run_eval_grid

    resolved_db = db_config or {}
    instance_connection_name = instance_connection_name or resolved_db.get("instance_connection_name")
    db_user = db_user or resolved_db.get("db_user")
    db_pass = db_pass or resolved_db.get("db_pass")
    if not instance_connection_name or not db_user or not db_pass:
        raise ValueError("Provide db_config or explicit instance_connection_name/db_user/db_pass values.")

    resolved_alias = model_alias or model_alias_from_id(model_id)
    run_metadata = build_run_metadata(
        model_id=model_id,
        model_alias=resolved_alias,
        method=method,
        notebook=notebook,
        adapter_dir=adapter_dir,
    )
    resolved_max_new_tokens = 128 if grid_plan.max_new_tokens is None else int(grid_plan.max_new_tokens)
    return run_eval_grid(
        test_set=test_set,
        schema_summary=schema_summary,
        model=model,
        tokenizer=tokenizer,
        engine=engine,
        instance_connection_name=instance_connection_name,
        db_user=db_user,
        db_pass=db_pass,
        k_values=grid_plan.k_values,
        seeds=grid_plan.seeds,
        run_tag=run_tag,
        runs_dir=runs_dir,
        run_metadata=run_metadata,
        limit=limit,
        enable_ts_for_k=set(grid_plan.ts_for_k_values) if grid_plan.enable_ts else None,
        ts_n=grid_plan.ts_n,
        ts_prefix=grid_plan.ts_prefix,
        ts_max_rows=grid_plan.ts_max_rows,
        max_new_tokens=resolved_max_new_tokens,
        eval_profile=grid_plan.eval_profile,
    )


def _react_progress_rates(items: list[dict[str, Any]]) -> tuple[float, float, float]:
    n_items = max(len(items), 1)
    return (
        sum(int(x["va"]) for x in items) / n_items,
        sum(int(x["em"]) for x in items) / n_items,
        sum(int(x["ex"]) for x in items) / n_items,
    )


def _react_report(
    *,
    config: Any,
    out_items: list[dict[str, Any]],
    run_metadata: dict[str, Any] | None,
    save_path: str | Path | None,
) -> dict[str, Any]:
    from nl2sql.core.query_runner import now_utc_iso

    n = len(out_items)
    va_rate, em_rate, ex_rate = _react_progress_rates(out_items)
    ts_values = [int(x["ts"]) for x in out_items if x.get("ts") is not None]
    report: dict[str, Any] = {
        "timestamp": now_utc_iso(),
        "method": "react",
        "config": asdict(config),
        "n": n,
        "va_rate": va_rate,
        "em_rate": em_rate,
        "ex_rate": ex_rate,
        "ts_rate": (sum(ts_values) / len(ts_values)) if ts_values else None,
        "ts_n": len(ts_values),
        "items": out_items,
    }
    if run_metadata:
        report["run_metadata"] = run_metadata
    if save_path:
        save_target = Path(save_path)
        save_target.parent.mkdir(parents=True, exist_ok=True)
        save_target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate_react_ablation(
    *,
    test_set: list[dict[str, Any]],
    engine: Engine,
    config: Any,
    limit: int | None = None,
    ts_suite_db_names: Optional[list[str]] = None,
    ts_make_engine_fn: Optional[Callable[[str], Engine]] = None,
    ts_max_rows: int = 500,
    progress_every: int = 20,
    run_metadata: Optional[dict[str, Any]] = None,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run ReAct over a split and return a report dict."""
    from nl2sql.agent.react_pipeline import run_react_pipeline
    from nl2sql.agent.agent_tools import get_agent_context
    from nl2sql.core.postprocess import normalize_sql
    from nl2sql.evaluation.eval import execution_accuracy, test_suite_accuracy_for_item

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
        if va and pred_sql and ts_suite_db_names and ts_make_engine_fn:
            ts = test_suite_accuracy_for_item(
                make_engine_fn=ts_make_engine_fn,
                suite_db_names=ts_suite_db_names,
                gold_sql=gold_sql,
                pred_sql=pred_sql,
                max_rows=ts_max_rows,
            )

        out_items.append({
            "i": i, "nlq": nlq, "gold_sql": gold_sql,
            "raw_sql": pred_sql, "pred_sql": pred_sql,
            "va": bool(va), "em": bool(em), "ex": bool(ex), "ts": ts,
            "error": pred_err or ex_pred_err, "gold_error": ex_gold_err,
            "trace": trace,
        })

        if progress_every and ((i + 1) % progress_every == 0 or (i + 1) == len(items)):
            va_rate, em_rate, ex_rate = _react_progress_rates(out_items)
            print(
                f"ReAct progress {i + 1}/{len(items)} | "
                f"VA={va_rate:.3f} EM={em_rate:.3f} EX={ex_rate:.3f}"
            )
    return _react_report(
        config=config,
        out_items=out_items,
        run_metadata=run_metadata,
        save_path=save_path,
    )


def run_react_notebook_eval(
    *,
    test_set: list[dict[str, Any]],
    engine: Any,
    config: Any,
    model_id: str,
    adapter_path: str,
    react_plan: ReactEvalPlan,
    ts_make_engine_fn: Any,
    notebook: str,
    out_root: str | Path = "results/agent/runs",
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    """Run ReAct eval and save one report file in a timestamped run folder."""
    run_tag = f"react_{config.name}"
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_dir = Path(out_root) / f"{run_tag}_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_filename = "results_react_200.json" if react_plan.quick_limit is None else "results_react_eval.json"
    out_path = run_dir / run_filename

    suite_dbs = (
        [f"{react_plan.ts_prefix}_{i:02d}" for i in range(1, react_plan.ts_n + 1)]
        if react_plan.ts_n and react_plan.ts_n > 0
        else []
    )
    run_metadata = {
        "commit": git_short_commit(),
        "notebook": notebook,
        "model_id": model_id,
        "adapter_path": adapter_path,
        "config_name": config.name,
        "quick_limit": react_plan.quick_limit,
        "ts_n": react_plan.ts_n,
    }

    report = evaluate_react_ablation(
        test_set=test_set,
        engine=engine,
        config=config,
        limit=react_plan.quick_limit,
        ts_suite_db_names=suite_dbs if suite_dbs else None,
        ts_make_engine_fn=ts_make_engine_fn if suite_dbs else None,
        ts_max_rows=react_plan.ts_max_rows,
        progress_every=20,
        run_metadata=run_metadata,
        save_path=out_path,
    )
    return report, report.get("items", []), out_path


def print_eval_rates(label: str, report: dict[str, Any]) -> None:
    """Compact metric summary for notebook output."""
    ts_rate = report.get("ts_rate")
    print(
        label,
        "VA=", round(report.get("va_rate", 0.0), 3),
        "EM=", round(report.get("em_rate", 0.0), 3),
        "EX=", round(report.get("ex_rate", 0.0), 3),
        "TS=", "NA" if ts_rate is None else round(ts_rate, 3),
    )


__all__ = [
    "build_run_metadata",
    "configure_react_notebook",
    "evaluate_react_ablation",
    "git_short_commit",
    "GridPlan",
    "primary_react_eval_plan",
    "model_alias_from_id",
    "primary_grid_plan",
    "print_grid_plan",
    "print_grid_run_report",
    "print_react_plan",
    "print_react_run_report",
    "QLORA_EXPERIMENT_PRESETS",
    "ReactEvalPlan",
    "print_eval_rates",
    "run_model_grid_notebook_eval",
    "run_react_notebook_eval",
    "train_qlora_adapter",
]
