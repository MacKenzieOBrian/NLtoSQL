"""ReAct setup and QLoRA training helpers for notebooks and fixed scripts.

Baseline and QLoRA grid execution now call ``run_eval_grid()`` directly. This
module keeps only the parts that still have distinct orchestration logic.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from sqlalchemy.engine import Engine

def _git_short_commit(default: str = "unknown") -> str:
    """Return the current short git hash."""
    # used from python docs
    # https://docs.python.org/3/library/subprocess.html#subprocess.check_output
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return default


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

_PRIMARY_REACT_TS_N = 10
_PRIMARY_REACT_TS_PREFIX = "classicmodels_ts"
_PRIMARY_REACT_TS_MAX_ROWS = 500


def configure_react_notebook(
    *,
    engine: Any,
    db_name: str,
    model: Any,
    tokenizer: Any,
    exemplar_pool: list[dict[str, Any]],
) -> Any:
    """Set the shared agent context and return the fixed dissertation ReAct config."""
    from nl2sql.agent.agent_tools import AgentContext, set_agent_context
    from nl2sql.agent.react_pipeline import ReactAblationConfig
    from nl2sql.core.query_runner import QueryRunner

    # ai note copilot: scaffold block only, i edited final logic
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
    # ai note copilot: scaffold block only, i edited final logic
    return ReactAblationConfig(
        name="react_barebones_notebook",
        max_steps=7,
        few_shot_k=3,
        few_shot_seed=7,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.2,
        top_p=0.9,
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
    # trainer setup from docs
    # https://huggingface.co/docs/trl/main/en/sft_trainer
    # https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
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


def _react_progress_rates(items: list[dict[str, Any]]) -> tuple[float, float, float]:
    # simple rate calc with sum
    # https://docs.python.org/3/library/functions.html#sum
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


def _evaluate_react_ablation(
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

    # ai note copilot: scaffold block only, i edited final logic
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
    ts_make_engine_fn: Any,
    notebook: str,
    out_root: str | Path = "results/agent/runs",
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    """Run ReAct eval and save one report file in a timestamped run folder."""
    # ai note copilot: scaffold block only, i edited final logic
    run_tag = f"react_{config.name}"
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_dir = Path(out_root) / f"{run_tag}_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "results_react_200.json"

    suite_dbs = (
        [f"{_PRIMARY_REACT_TS_PREFIX}_{i:02d}" for i in range(1, _PRIMARY_REACT_TS_N + 1)]
        if _PRIMARY_REACT_TS_N > 0
        else []
    )
    run_metadata = {
        "commit": _git_short_commit(),
        "notebook": notebook,
        "model_id": model_id,
        "adapter_path": adapter_path,
        "config_name": config.name,
        "ts_n": _PRIMARY_REACT_TS_N,
    }

    # ai note copilot: scaffold block only, i edited final logic
    report = _evaluate_react_ablation(
        test_set=test_set,
        engine=engine,
        config=config,
        limit=None,
        ts_suite_db_names=suite_dbs if suite_dbs else None,
        ts_make_engine_fn=ts_make_engine_fn if suite_dbs else None,
        ts_max_rows=_PRIMARY_REACT_TS_MAX_ROWS,
        progress_every=20,
        run_metadata=run_metadata,
        save_path=out_path,
    )
    return report, report.get("items", []), out_path


__all__ = [
    "configure_react_notebook",
    "QLORA_EXPERIMENT_PRESETS",
    "run_react_notebook_eval",
    "train_qlora_adapter",
]
