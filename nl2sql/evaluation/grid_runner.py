"""
Evaluation grid runner — shared by baseline and QLoRA notebooks.

Iterates the fixed dissertation grid, calls ``eval_run()``, and writes one
JSON file per run. The official outputs are the raw per-run JSON files only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine

from ..infra.db import create_engine_with_connector
from ..evaluation.eval import (
    build_eval_run_config,
    eval_run,
)

PRIMARY_GRID_K_VALUES = [0, 3]
PRIMARY_GRID_SEEDS_BY_K = {
    0: [7],
    3: [7, 17, 27, 37, 47, 57, 67, 77, 87, 97],
}
PRIMARY_GRID_TS_K_VALUES = {3}
PRIMARY_GRID_TS_N = 10
PRIMARY_GRID_TS_PREFIX = "classicmodels_ts"
PRIMARY_GRID_TS_MAX_ROWS = 500
PRIMARY_GRID_MAX_NEW_TOKENS = 128


def _build_ts_suite_names(ts_enabled_k: set[int], ts_prefix: str, ts_n: int) -> list[str] | None:
    """Build the perturbed database names used for TS scoring."""
    if not ts_enabled_k or ts_n <= 0:
        return None
    return [f"{ts_prefix}_{i:02d}" for i in range(1, ts_n + 1)]


def _summarize_items(items: list[Any]) -> dict[str, Any]:
    """Collapse one run's item list into the small report shown in notebooks."""
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
    run_tag: str,
    runs_dir: str | Path,
    run_metadata: dict[str, Any],
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the fixed dissertation (k × seed) grid and save one JSON file per condition.

    Returns the saved run directory plus a compact per-run summary for printing.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_dir = Path(runs_dir) / f"{run_tag}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # The dissertation recipe only enables TS for k=3, where few-shot runs are the main comparison.
    ts_enabled_k: set[int] = set(PRIMARY_GRID_TS_K_VALUES)
    ts_suite_db_names = _build_ts_suite_names(ts_enabled_k, PRIMARY_GRID_TS_PREFIX, PRIMARY_GRID_TS_N)

    ts_connectors: dict[str, Any] = {}

    @lru_cache(maxsize=32)
    def _make_engine_cached(db_name: str) -> Engine:
        """Reuse perturbed-DB engines so TS does not recreate connectors per item."""
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
        for k in PRIMARY_GRID_K_VALUES:
            seeds = PRIMARY_GRID_SEEDS_BY_K.get(k, [])
            for seed in seeds:
                save_path = run_dir / f"results_k{k}_seed{seed}.json"

                run_meta = dict(run_metadata)
                run_meta.update({
                    "run_tag": run_tag,
                    "k": k,
                    "seed": seed,
                    "exemplar_pool_size": len(test_set),
                    # Persist the TS policy so the saved JSON still explains how it was run.
                    "ts_enabled": k in ts_enabled_k,
                    "ts_n": PRIMARY_GRID_TS_N if ts_suite_db_names else 0,
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
                        max_new_tokens=PRIMARY_GRID_MAX_NEW_TOKENS,
                        avoid_exemplar_leakage=True,
                        ts_suite_db_names=ts_suite_db_names if k in ts_enabled_k else None,
                        ts_make_engine_fn=_make_engine_cached if k in ts_enabled_k else None,
                        ts_max_rows=PRIMARY_GRID_TS_MAX_ROWS,
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

    print("Saved run dir:", run_dir)
    return {
        "run_dir": run_dir,
        "runs": rows,
    }


__all__ = ["run_eval_grid"]
