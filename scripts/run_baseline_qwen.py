#!/usr/bin/env python3
"""Run the fixed Qwen baseline campaign for the dissertation rerun."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from nl2sql.core.schema import build_schema_summary
from nl2sql.evaluation.grid_runner import run_eval_grid
from nl2sql.infra.db import connect_notebook_db
from nl2sql.infra.model_loading import load_quantized_model
from nl2sql.infra.notebook_utils import ensure_hf_token, load_test_set


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
RUN_TAG = "qwen_baseline"


def _git_short_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    """Run the fixed Qwen baseline grid and print the final-pack copy targets."""
    project_root = Path(__file__).resolve().parents[1]
    engine, connector, db_config = connect_notebook_db(default_db_name="classicmodels")
    try:
        test_set = load_test_set(project_root / "data/classicmodels_test_200.json")
        schema_summary = build_schema_summary(engine, db_name=db_config["db_name"])
        token = ensure_hf_token(prompt_if_missing=True)
        model, tok = load_quantized_model(MODEL_ID, token=token)

        print({
            "run": "baseline",
            "model_id": MODEL_ID,
            "k_values": [0, 3],
            "seed_policy": {"k0": [7], "k3": [7, 17, 27, 37, 47]},
            "ts_k": [3],
        })
        report = run_eval_grid(
            test_set=test_set,
            schema_summary=schema_summary,
            model=model,
            tokenizer=tok,
            engine=engine,
            instance_connection_name=db_config["instance_connection_name"],
            db_user=db_config["db_user"],
            db_pass=db_config["db_pass"],
            run_tag=RUN_TAG,
            runs_dir="results/baseline/runs",
            run_metadata={
                "commit": _git_short_commit(),
                "model_id": MODEL_ID,
                "method": "baseline",
                "notebook": "scripts/run_baseline_qwen.py",
            },
        )
        print("Saved run dir:", report["run_dir"])
        print("Per-run metric rows:")
        for row in report["runs"]:
            print(row)

        print("Copy these files into results/final_pack/:")
        for row in report["runs"]:
            print(f"  {row['json_path']} -> qwen_base_k{row['k']}_seed{row['seed']}.json")
    finally:
        connector.close()


if __name__ == "__main__":
    main()
