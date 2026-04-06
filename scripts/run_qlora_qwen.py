#!/usr/bin/env python3
"""Run the fixed Qwen QLoRA campaign for the dissertation rerun."""

from __future__ import annotations

import gc
import subprocess
import sys
from pathlib import Path

import torch
from datasets import Dataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from nl2sql.core.prompting import make_training_example
from nl2sql.core.schema import build_schema_summary
from nl2sql.evaluation.grid_runner import run_eval_grid
from nl2sql.infra.db import connect_notebook_db
from nl2sql.infra.experiment_helpers import (
    QLORA_EXPERIMENT_PRESETS,
    train_qlora_adapter,
)
from nl2sql.infra.model_loading import build_trainable_qlora_model, load_eval_adapter_model
from nl2sql.infra.notebook_utils import ensure_hf_token, load_test_and_train_sets


PRESET_NAME = "qwen2_5_7b"
RUN_TAG = "qwen_qlora"


def _git_short_commit() -> str:
    # used from python docs
    # https://docs.python.org/3/library/subprocess.html#subprocess.check_output
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    """Train the fixed Qwen adapter, evaluate it, and print copy targets."""
    # ai note copilot: "script entrypoint: db connect, model load, eval grid call, print copy targets"
    project_root = Path(__file__).resolve().parents[1]
    experiment = QLORA_EXPERIMENT_PRESETS[PRESET_NAME]
    model_id = experiment["model_id"]

    test_set, train_records = load_test_and_train_sets(
        test_path=project_root / "data/classicmodels_test_200.json",
        train_path=project_root / "data/train/classicmodels_train_200.jsonl",
    )
    engine, connector, db_config = connect_notebook_db(default_db_name="classicmodels")
    try:
        schema_summary = build_schema_summary(engine, db_name=db_config["db_name"], max_cols_per_table=50)
        token = ensure_hf_token(prompt_if_missing=True)
        model, tok, bnb_config, compute_dtype, use_bf16 = build_trainable_qlora_model(
            experiment_config=experiment,
            token=token,
        )
        train_texts = [make_training_example(r["nlq"], r["sql"], schema_summary, tok) for r in train_records]
        train_ds = Dataset.from_dict({"text": train_texts})
        print("GPU:", torch.cuda.get_device_name(0), "| use_bf16:", use_bf16)
        train_qlora_adapter(
            model=model,
            tokenizer=tok,
            train_dataset=train_ds,
            experiment_config=experiment,
            output_dir=experiment["adapter_output_dir"],
            use_bf16=use_bf16,
        )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        eval_model, adapter_path = load_eval_adapter_model(
            model_id=model_id,
            adapter_path=experiment["adapter_output_dir"],
            bnb_config=bnb_config,
            compute_dtype=compute_dtype,
            token=token,
        )

        print({
            "run": "qlora",
            "model_id": model_id,
            "adapter": str(adapter_path),
            "k_values": [0, 3],
            "seed_policy": {"k0": [7], "k3": [7, 17, 27, 37, 47, 57, 67, 77, 87, 97]},
            "ts_k": [3],
        })
        report = run_eval_grid(
            test_set=test_set,
            schema_summary=schema_summary,
            model=eval_model,
            tokenizer=tok,
            engine=engine,
            instance_connection_name=db_config["instance_connection_name"],
            db_user=db_config["db_user"],
            db_pass=db_config["db_pass"],
            run_tag=RUN_TAG,
            runs_dir="results/qlora/runs",
            run_metadata={
                "commit": _git_short_commit(),
                "model_id": model_id,
                "method": "qlora",
                "notebook": "scripts/run_qlora_qwen.py",
                "adapter_dir": str(adapter_path),
            },
        )
        print("Saved run dir:", report["run_dir"])
        print("Per-run metric rows:")
        for row in report["runs"]:
            print(row)

        print("Copy these files into results/final_pack/:")
        for row in report["runs"]:
            print(f"  {row['json_path']} -> qwen_qlora_k{row['k']}_seed{row['seed']}.json")
    finally:
        connector.close()


if __name__ == "__main__":
    main()
