"""
Evaluation grid runner — shared by baseline and QLoRA notebooks.

Iterates (k, seed) combinations, calls eval_run(), and writes per-run JSONs
plus aggregated grid_summary CSVs to a timestamped run directory.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
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


def _copy_canonical_result(*, save_path: Path, canonical_dir: str | Path, k: int) -> None:
    name = "results_zero_shot_200.json" if k == 0 else "results_few_shot_k3_200.json"
    target = Path(canonical_dir) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(save_path, target)
    print(f"Updated canonical: {target}")


def _write_grid_summaries(rows: list[dict[str, Any]], run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(rows).sort_values(["k", "seed"]).reset_index(drop=True)
    df.to_csv(run_dir / "grid_summary.csv", index=False)

    agg = df.groupby(["k"], as_index=False).agg(
        runs=("seed", "count"),
        va_mean=("va_rate", "mean"), va_std=("va_rate", "std"),
        em_mean=("em_rate", "mean"), em_std=("em_rate", "std"),
        ex_mean=("ex_rate", "mean"), ex_std=("ex_rate", "std"),
        ts_mean=("ts_rate", "mean"), ts_std=("ts_rate", "std"),
    )
    agg.to_csv(run_dir / "grid_summary_by_k.csv", index=False)
    return df, agg


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
    copy_canonical: bool = False,
    canonical_dir: str | Path | None = None,
    enable_ts_for_k: set[int] | None = None,
    ts_n: int = 10,
    ts_prefix: str = "classicmodels_ts",
    ts_max_rows: int = 500,
    max_new_tokens: int = 128,
    eval_profile: str = EVAL_PROFILE_MODEL_ONLY_RAW,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Run a (k × seed) evaluation grid and write results to runs_dir.

    Returns (per-run DataFrame, per-k aggregated DataFrame, run directory Path).
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
    primary_seed = seeds[0]

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

                if copy_canonical and seed == primary_seed and k in {0, 3} and canonical_dir:
                    _copy_canonical_result(save_path=save_path, canonical_dir=canonical_dir, k=k)

    finally:
        for conn in ts_connectors.values():
            try:
                conn.close()
            except Exception:
                pass

    df, agg = _write_grid_summaries(rows, run_dir)

    print("Saved grid run to:", run_dir, "| eval_profile:", eval_profile)
    return df, agg, run_dir


__all__ = ["run_eval_grid"]
