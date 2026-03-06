"""
Evaluation grid runner — shared by baseline and QLoRA notebooks.

Iterates (k, seed) combinations, calls eval_run(), and writes one JSON file
per run. Keeps output plain: raw result files first, later analysis second.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine

from ..infra.db import create_engine_with_connector
from ..evaluation.eval import (
    EVAL_PROFILE_MODEL_ONLY_RAW,
    build_eval_run_config,
    eval_run,
)


def _build_ts_suite_names(ts_enabled_k: set[int], ts_prefix: str, ts_n: int) -> list[str] | None:
    if not ts_enabled_k or ts_n <= 0:
        return None
    return [f"{ts_prefix}_{i:02d}" for i in range(1, ts_n + 1)]


def _summarize_items(items: list[Any]) -> dict[str, Any]:
    n = len(items)
    ts_values = [int(x.ts) for x in items if getattr(x, "ts", None) is not None]
    return {
        "n": n,
        "va_rate": sum(int(x.va) for x in items) / max(n, 1),
        "em_rate": sum(int(x.em) for x in items) / max(n, 1),
        "ex_rate": sum(int(x.ex) for x in items) / max(n, 1),
        "ts_rate": (sum(ts_values) / len(ts_values)) if ts_values else None,
        "ts_n": len(ts_values),
    }


def _write_run_report(
    *,
    rows: list[dict[str, Any]],
    run_dir: Path,
    run_tag: str,
    eval_profile: str,
    k_values: list[int],
    seeds: list[int],
) -> Path:
    report_path = run_dir / "run_report.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_tag": run_tag,
        "eval_profile": eval_profile,
        "k_values": list(k_values),
        "seeds": list(seeds),
        "runs": rows,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def run_eval_grid(
    *,
    test_set: list[dict[str, Any]],
    schema_summary: str,
    model: Any,
    tokenizer: Any,
    engine: Engine,
    instance_connection_name: str,
    db_user: str,
    db_pass: str,
    k_values: list[int],
    seeds: list[int],
    run_tag: str,
    runs_dir: str | Path,
    run_metadata: dict[str, Any],
    limit: int | None = None,
    enable_ts_for_k: set[int] | None = None,
    ts_n: int = 10,
    ts_prefix: str = "classicmodels_ts",
    ts_max_rows: int = 500,
    max_new_tokens: int = 128,
    eval_profile: str = EVAL_PROFILE_MODEL_ONLY_RAW,
) -> dict[str, Any]:
    """Run a (k × seed) grid and save one JSON file per condition.

    Returns a plain run report with saved file paths and simple per-run metrics.
    """
    if not seeds:
        raise ValueError("Provide at least one seed.")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_dir = Path(runs_dir) / f"{run_tag}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    ts_enabled_k: set[int] = set(enable_ts_for_k or set())
    ts_suite_db_names = _build_ts_suite_names(ts_enabled_k, ts_prefix, ts_n)

    ts_connectors: dict[str, Any] = {}

    @lru_cache(maxsize=32)
    def _make_engine_cached(db_name: str) -> Engine:
        eng, conn = create_engine_with_connector(
            instance_connection_name=instance_connection_name,
            user=db_user,
            password=db_pass,
            db_name=db_name,
        )
        ts_connectors[db_name] = conn
        return eng

    rows: list[dict[str, Any]] = []

    try:
        for k in k_values:
            for seed in seeds:
                save_path = run_dir / f"results_k{k}_seed{seed}.json"

                run_meta = dict(run_metadata)
                run_meta.update({
                    "run_tag": run_tag,
                    "k": k,
                    "seed": seed,
                    "eval_profile": eval_profile,
                    "exemplar_pool_size": len(test_set),
                    "ts_enabled": k in ts_enabled_k,
                    "ts_n": ts_n if ts_suite_db_names else 0,
                })

                items = eval_run(
                    test_set=test_set,
                    exemplar_pool=test_set,
                    k=k,
                    limit=limit,
                    seed=seed,
                    engine=engine,
                    model=model,
                    tokenizer=tokenizer,
                    schema_summary=schema_summary,
                    save_path=str(save_path),
                    run_metadata=run_meta,
                    config=build_eval_run_config(
                        eval_profile=eval_profile,
                        max_new_tokens=max_new_tokens,
                        avoid_exemplar_leakage=True,
                        ts_suite_db_names=ts_suite_db_names if k in ts_enabled_k else None,
                        ts_make_engine_fn=_make_engine_cached if k in ts_enabled_k else None,
                        ts_max_rows=ts_max_rows,
                    ),
                )

                rows.append({
                    "run_tag": run_tag,
                    "k": k,
                    "seed": seed,
                    **_summarize_items(items),
                    "json_path": str(save_path),
                })

    finally:
        for conn in ts_connectors.values():
            try:
                conn.close()
            except Exception:
                pass

    rows = sorted(rows, key=lambda row: (row["k"], row["seed"]))
    report_path = _write_run_report(
        rows=rows,
        run_dir=run_dir,
        run_tag=run_tag,
        eval_profile=eval_profile,
        k_values=k_values,
        seeds=seeds,
    )

    print("Saved run dir:", run_dir, "| eval_profile:", eval_profile)
    return {
        "run_dir": run_dir,
        "report_path": report_path,
        "eval_profile": eval_profile,
        "runs": rows,
    }


__all__ = ["run_eval_grid"]
