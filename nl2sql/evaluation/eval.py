"""Evaluation helpers for VA, EM, EX, and TS."""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import sqlalchemy
from sqlalchemy.engine import Engine

from ..infra.db import safe_connection
from ..core.llm import generate_sql_from_messages
from ..core.postprocess import normalize_sql as _normalize_sql
from ..core.prompting import make_few_shot_messages
from ..core.query_runner import check_sql_safety, now_utc_iso, QueryRunner

# Frozen dataclasses keep scored records immutable after construction:
# https://docs.python.org/3/library/dataclasses.html
@dataclass(frozen=True)
class EvalItem:
    """Scored output for one benchmark item within one evaluation run."""
    i: int
    nlq: str
    gold_sql: str
    raw_sql: str      # exact model output before any post-processing
    pred_sql: str     # SQL actually scored (kept separate so later analysis can stay explicit)
    va: bool
    em: bool
    ex: bool
    ts: Optional[int]
    error: Optional[str]
    gold_error: Optional[str]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "i": self.i,
            "nlq": self.nlq,
            "gold_sql": self.gold_sql,
            "raw_sql": self.raw_sql,
            "pred_sql": self.pred_sql,
            "va": self.va,
            "em": self.em,
            "ex": self.ex,
            "ts": self.ts,
            "error": self.error,
            "gold_error": self.gold_error,
        }


def execute_fetch(
    *,
    engine: Engine,
    sql: str,
    max_rows: int = 10000,
) -> tuple[bool, list[str] | None, list[tuple] | None, str | None]:
    """Execute one query for EX comparison and return rows or an error string."""
    try:
        check_sql_safety(sql)
        with safe_connection(engine) as conn:
            result = conn.execute(sqlalchemy.text(sql))
            cols = list(result.keys())
            rows = result.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            return False, cols, None, f"Result set too large (> {max_rows} rows) for comparison"
        return True, cols, [tuple(r) for r in rows], None
    except Exception as e:
        return False, None, None, str(e)


# ai note copilot: "Counter bag equality for result-set comparison"
def execution_accuracy(
    *,
    engine: Engine,
    pred_sql: str,
    gold_sql: str,
    max_compare_rows: int = 10000,
) -> tuple[bool, str | None, str | None]:
    """Return whether predicted and gold SQL produce the same result rows."""
    pred_ok, _, pred_rows, pred_err = execute_fetch(engine=engine, sql=pred_sql, max_rows=max_compare_rows)
    gold_ok, _, gold_rows, gold_err = execute_fetch(engine=engine, sql=gold_sql, max_rows=max_compare_rows)

    if not gold_ok:
        return False, pred_err, gold_err
    if not pred_ok:
        return False, pred_err, gold_err

    return Counter(pred_rows) == Counter(gold_rows), None, None


def _coerce_cell(x: Any) -> Any:
    # Normalize floats so row comparisons stay stable.
    if x is None:
        return None
    if isinstance(x, float):
        if math.isnan(x):
            return "NaN"
        return round(x, 10)
    return x


def _run_query_ts(engine: Engine, sql: str, max_rows: int = 2000):
    """Execute a SELECT on a perturbed database; return normalised rows or None on failure."""
    try:
        check_sql_safety(sql)
        with safe_connection(engine) as conn:
            res = conn.execute(sqlalchemy.text(sql))
            rows = res.fetchmany(max_rows)
        return [tuple(_coerce_cell(v) for v in r) for r in rows]
    except Exception:
        return None


# ai note copilot: "multi-replica loop with any-failure-returns-0 rule"
def test_suite_accuracy_for_item(
    *,
    make_engine_fn: Callable[[str], Engine],
    suite_db_names: list[str],
    gold_sql: str,
    pred_sql: str,
    max_rows: int = 2000,
) -> int:
    """Return 1 only if pred_sql matches gold_sql on every checked perturbed database.

    If the gold query itself fails on any replica, this helper returns 0 rather than
    skipping that replica. That keeps the scoring rule strict and easy to explain.
    """
    # Inspired by distilled test-suite evaluation [19]: check the same SQL pair
    # on several perturbed database variants. This project uses a simplified
    # local helper rather than the exact benchmark toolkit.
    # ORDER BY queries need order-sensitive comparison; others can use bag equality.
    ordered = bool(re.search(r"(?i)\border\s+by\b", gold_sql or ""))
    usable = 0
    for db in suite_db_names:
        eng = make_engine_fn(db)
        gold_rows = _run_query_ts(eng, gold_sql, max_rows)
        if gold_rows is None:
            return 0  # any replica where gold fails → reject
        pred_rows = _run_query_ts(eng, pred_sql, max_rows)
        usable += 1
        if pred_rows is None:
            return 0
        match = gold_rows == pred_rows if ordered else Counter(gold_rows) == Counter(pred_rows)
        if not match:
            return 0
    return 1 if usable > 0 else 0

# Remove the current test item from the exemplar pool so few-shot examples cannot leak the answer.
# ai note copilot: "list comprehension to filter test item from exemplar pool"
def _build_item_pool(
    *,
    item: dict[str, Any],
    test_set: list[dict[str, Any]],
    exemplar_pool: list[dict[str, Any]] | None,
    avoid_exemplar_leakage: bool,
) -> list[dict[str, Any]]:
    """Exclude the current test item from the exemplar pool to prevent leakage.

    Match on both nlq and sql so paraphrases are not removed by mistake.
    """
    pool = exemplar_pool if exemplar_pool is not None else test_set
    if not avoid_exemplar_leakage:
        return pool
    nlq = item["nlq"]
    gold_sql = item["sql"]
    return [
        ex
        for ex in pool
        if not (ex.get("nlq") == nlq and ex.get("sql") == gold_sql)
    ]


def _sample_exemplars(*, rng: random.Random, pool: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    # Separate `Random` instances keep run sampling isolated from global RNG state:
    # https://docs.python.org/3/library/random.html#random.Random
    if k <= 0:
        return []
    if len(pool) < k:
        raise ValueError(f"Exemplar pool too small: k={k} but pool has {len(pool)} items")
    return rng.sample(pool, k)

# Frozen dataclasses keep scored configs immutable after construction:
# https://docs.python.org/3/library/dataclasses.html
@dataclass(frozen=True)
class EvalRunConfig:
    """Settings for one raw-model evaluation run."""
    max_new_tokens: int = 128
    max_rows: int = 50
    max_compare_rows: int = 10000
    avoid_exemplar_leakage: bool = True
    ts_suite_db_names: Optional[list[str]] = None
    ts_make_engine_fn: Optional[Callable[[str], Engine]] = None
    ts_max_rows: int = 500


def build_eval_run_config(
    *,
    max_new_tokens: int = 128,
    max_rows: int = 50,
    max_compare_rows: int = 10000,
    avoid_exemplar_leakage: bool = True,
    ts_suite_db_names: Optional[list[str]] = None,
    ts_make_engine_fn: Optional[Callable[[str], Engine]] = None,
    ts_max_rows: int = 500,
) -> EvalRunConfig:
    """Build the single raw-output evaluation config used by this project."""
    return EvalRunConfig(
        max_new_tokens=max_new_tokens,
        max_rows=max_rows,
        max_compare_rows=max_compare_rows,
        avoid_exemplar_leakage=avoid_exemplar_leakage,
        ts_suite_db_names=ts_suite_db_names,
        ts_make_engine_fn=ts_make_engine_fn,
        ts_max_rows=ts_max_rows,
    )


def _generate_candidate_sql(
    *,
    nlq: str,
    exemplars: list[dict[str, Any]],
    schema_summary: str,
    model: Any,
    tokenizer: Any,
    config: EvalRunConfig,
) -> tuple[str, str]:
    messages = make_few_shot_messages(
        schema=schema_summary,
        exemplars=exemplars,
        nlq=nlq,
    )
    raw_sql = generate_sql_from_messages(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=config.max_new_tokens,
        extract_select=False,
        stop_on_semicolon=False,
    )
    pred_sql = (raw_sql or "").strip()
    return raw_sql, pred_sql


def _maybe_test_suite_score(
    *,
    meta: Any,
    pred_sql: str,
    gold_sql: str,
    config: EvalRunConfig,
) -> Optional[int]:
    if not (bool(meta.success) and config.ts_suite_db_names and config.ts_make_engine_fn and pred_sql):
        return None
    return test_suite_accuracy_for_item(
        make_engine_fn=config.ts_make_engine_fn,
        suite_db_names=config.ts_suite_db_names,
        gold_sql=gold_sql,
        pred_sql=pred_sql,
        max_rows=config.ts_max_rows,
    )


def _score_prediction(
    *,
    pred_sql: str,
    gold_sql: str,
    qr: QueryRunner,
    engine: Engine,
    config: EvalRunConfig,
) -> dict[str, Any]:
    meta = qr.run(pred_sql)
    em = _normalize_sql(pred_sql) == _normalize_sql(gold_sql)
    ex, ex_pred_err, ex_gold_err = execution_accuracy(
        engine=engine,
        pred_sql=pred_sql,
        gold_sql=gold_sql,
        max_compare_rows=config.max_compare_rows,
    )
    return {
        "meta": meta,
        "em": bool(em),
        "ex": bool(ex),
        "ts": _maybe_test_suite_score(meta=meta, pred_sql=pred_sql, gold_sql=gold_sql, config=config),
        "error": meta.error or ex_pred_err,
        "gold_error": ex_gold_err,
    }


def _evaluate_item(
    *,
    i: int,
    item: dict[str, Any],
    exemplars: list[dict[str, Any]],
    schema_summary: str,
    model: Any,
    tokenizer: Any,
    qr: QueryRunner,
    engine: Engine,
    config: EvalRunConfig,
) -> EvalItem:
    nlq = item["nlq"]
    gold_sql = item["sql"]

    raw_sql, pred_sql = _generate_candidate_sql(
        nlq=nlq,
        exemplars=exemplars,
        schema_summary=schema_summary,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )
    scores = _score_prediction(
        pred_sql=pred_sql,
        gold_sql=gold_sql,
        qr=qr,
        engine=engine,
        config=config,
    )

    return EvalItem(
        i=i,
        nlq=nlq,
        gold_sql=gold_sql,
        raw_sql=raw_sql,
        pred_sql=pred_sql,
        va=bool(scores["meta"].success),
        em=bool(scores["em"]),
        ex=bool(scores["ex"]),
        ts=scores["ts"],
        error=scores["error"],
        gold_error=scores["gold_error"],
    )


def _summarize_eval(out: list[EvalItem]) -> tuple[float, float, float, float | None, list[int]]:
    va_rate = sum(r.va for r in out) / max(len(out), 1)
    em_rate = sum(r.em for r in out) / max(len(out), 1)
    ex_rate = sum(r.ex for r in out) / max(len(out), 1)
    ts_values = [int(r.ts) for r in out if r.ts is not None]
    ts_rate = (sum(ts_values) / len(ts_values)) if ts_values else None
    return va_rate, em_rate, ex_rate, ts_rate, ts_values


def _build_eval_payload(
    *,
    out: list[EvalItem],
    k: int,
    seed: int,
    limit: int | None,
    exemplar_pool: list[dict[str, Any]] | None,
    config: EvalRunConfig,
    run_metadata: Optional[dict[str, Any]],
) -> dict[str, Any]:
    va_rate, em_rate, ex_rate, ts_rate, ts_values = _summarize_eval(out)
    payload: dict[str, Any] = {
        "timestamp": now_utc_iso(),
        "k": k,
        "seed": seed,
        "limit": limit,
        "n": len(out),
        "va_rate": va_rate,
        "em_rate": em_rate,
        "ex_rate": ex_rate,
        "ts_rate": ts_rate,
        "ts_n": len(ts_values),
        "exemplar_policy": "custom_pool" if exemplar_pool is not None else "benchmark_pool",
        "avoid_exemplar_leakage": bool(config.avoid_exemplar_leakage),
        "results": [r.to_jsonable() for r in out],
    }
    if run_metadata:
        payload["run_metadata"] = run_metadata
    return payload


def eval_run(
    *,
    test_set: list[dict[str, Any]],
    exemplar_pool: list[dict[str, Any]] | None = None,
    k: int,
    engine: Engine,
    model: Any,
    tokenizer: Any,
    schema_summary: str,
    limit: int | None = 50,
    seed: int = 7,
    save_path: str | Path | None = None,
    run_metadata: Optional[dict[str, Any]] = None,
    config: EvalRunConfig | None = None,
) -> list[EvalItem]:
    """Run one evaluation setting and return per-item results."""
    cfg = config or EvalRunConfig()
    rng = random.Random(seed)
    items = test_set[:limit] if limit else test_set

    qr = QueryRunner(engine, max_rows=cfg.max_rows)
    out: list[EvalItem] = []

    for i, item in enumerate(items):
        pool = _build_item_pool(
            item=item,
            test_set=test_set,
            exemplar_pool=exemplar_pool,
            avoid_exemplar_leakage=cfg.avoid_exemplar_leakage,
        )
        exemplars = _sample_exemplars(rng=rng, pool=pool, k=k)
        out.append(
            _evaluate_item(
                i=i,
                item=item,
                exemplars=exemplars,
                schema_summary=schema_summary,
                model=model,
                tokenizer=tokenizer,
                qr=qr,
                engine=engine,
                config=cfg,
            )
        )

    va_rate, em_rate, ex_rate, ts_rate, ts_values = _summarize_eval(out)
    ts_s = "NA" if ts_rate is None else f"{ts_rate:.3f}"
    print(f"k={k} | n={len(out)} | VA={va_rate:.3f} | EM={em_rate:.3f} | EX={ex_rate:.3f} | TS={ts_s}")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _build_eval_payload(
            out=out,
            k=k,
            seed=seed,
            limit=limit,
            exemplar_pool=exemplar_pool,
            config=cfg,
            run_metadata=run_metadata,
        )
        save_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("Saved:", str(save_path))

    return out


# ai note copilot: "if/elif chain for EX failure category mapping"
def categorize_failure(item: dict) -> str:
    """Classify one result item into an EX failure category."""
    pred = item.get("pred_sql")
    va = int(item.get("va", 0))
    ex = int(item.get("ex", 0))
    err = str(item.get("error") or "").lower()

    if not pred:
        return "repair_budget_exhausted" if "repair_budget_exhausted" in err else "no_prediction"
    if va == 0:
        if "guardrail_reject" in err:
            return "guardrail_reject"
        if "validate_sql" in err:
            return "validate_sql_failed"
        return "invalid_sql"
    if ex == 1:
        return "correct"
    if "intent_mismatch" in err:
        return "intent_mismatch"
    return "semantic_mismatch"


def save_failure_profile(
    *,
    items: list[dict[str, Any]],
    config_name: str,
    out_path: str | Path,
) -> tuple[dict[str, int], Path]:
    """Persist EX failure category counts for notebook diagnostics."""
    counts = Counter(categorize_failure(r) for r in items)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "counts": dict(counts),
        "n_items": len(items),
        "config_name": config_name,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dict(counts), path
