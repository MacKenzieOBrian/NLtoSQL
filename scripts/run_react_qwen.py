#!/usr/bin/env python3
"""Run the fixed Qwen base-model ReAct campaign for the dissertation rerun."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from nl2sql.infra.db import connect_notebook_db, make_cached_engine_factory
from nl2sql.infra.experiment_helpers import (
    configure_react_notebook,
    run_react_notebook_eval,
)
from nl2sql.infra.model_loading import load_quantized_model
from nl2sql.infra.notebook_utils import ensure_hf_token, load_test_set


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = None
SMOKE_VA_MIN = 0.70
SMOKE_EX_MIN = 0.30


def main() -> None:
    """Run the fixed Qwen ReAct rerun and print the final-pack copy target."""
    project_root = Path(__file__).resolve().parents[1]
    engine, connector, db_config = connect_notebook_db(default_db_name="classicmodels", verify=True)
    try:
        test_set = load_test_set(project_root / "data/classicmodels_test_200.json")
        token = ensure_hf_token(prompt_if_missing=True)
        model, tok = load_quantized_model(MODEL_ID, token=token, adapter_path=ADAPTER_PATH)
        react_config = configure_react_notebook(
            engine=engine,
            db_name=db_config["db_name"],
            model=model,
            tokenizer=tok,
            exemplar_pool=test_set,
        )
        ts_make_engine = make_cached_engine_factory(
            connector=connector,
            instance_connection_name=db_config["instance_connection_name"],
            user=db_config["db_user"],
            password=db_config["db_pass"],
        )
        smoke_report, _, smoke_out_path = run_react_notebook_eval(
            test_set=test_set[:20],
            engine=engine,
            config=react_config,
            model_id=MODEL_ID,
            adapter_path=ADAPTER_PATH,
            ts_make_engine_fn=ts_make_engine,
            notebook="scripts/run_react_qwen.py:smoke",
        )
        smoke_ok = (
            smoke_report.get("va_rate", 0.0) >= SMOKE_VA_MIN
            and smoke_report.get("ex_rate", 0.0) >= SMOKE_EX_MIN
        )
        print({
            "run": "react_smoke",
            "model_id": MODEL_ID,
            "adapter_path": ADAPTER_PATH,
            "limit": 20,
            "va_min": SMOKE_VA_MIN,
            "ex_min": SMOKE_EX_MIN,
            "pass": smoke_ok,
        })
        print(
            "ReAct smoke",
            "VA=", round(smoke_report.get("va_rate", 0.0), 3),
            "EM=", round(smoke_report.get("em_rate", 0.0), 3),
            "EX=", round(smoke_report.get("ex_rate", 0.0), 3),
            "TS=", "NA" if smoke_report.get("ts_rate") is None else round(smoke_report["ts_rate"], 3),
        )
        print("Saved smoke report:", smoke_out_path)
        if not smoke_ok:
            raise RuntimeError(
                f"Smoke test failed: require VA >= {SMOKE_VA_MIN:.2f} and EX >= {SMOKE_EX_MIN:.2f}"
            )
        report, _, out_path = run_react_notebook_eval(
            test_set=test_set,
            engine=engine,
            config=react_config,
            model_id=MODEL_ID,
            adapter_path=ADAPTER_PATH,
            ts_make_engine_fn=ts_make_engine,
            notebook="scripts/run_react_qwen.py",
        )
        print({
            "run": "react",
            "config": react_config.name,
            "model_id": MODEL_ID,
            "adapter_path": ADAPTER_PATH,
            "run_size": 200,
            "ts_n": 10,
            "ts_prefix": "classicmodels_ts",
            "ts_max_rows": 500,
        })
        print(
            "ReAct",
            "VA=", round(report.get("va_rate", 0.0), 3),
            "EM=", round(report.get("em_rate", 0.0), 3),
            "EX=", round(report.get("ex_rate", 0.0), 3),
            "TS=", "NA" if report.get("ts_rate") is None else round(report["ts_rate"], 3),
        )
        print("Saved report:", out_path)
        print(f"Copy this file into results/final_pack/: {out_path} -> qwen_react_k3_seed7.json")
    finally:
        connector.close()


if __name__ == "__main__":
    main()
