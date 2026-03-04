"""
Evaluation helpers for VA/EM/EX/TS.

Default mode:
1) Build few-shot prompts from schema + exemplars.
2) Generate raw model text with no decoding constraints.
3) Score VA / EM / EX (and optional TS).

Optional mode:
- Enable constrained decoding and reliability cleanup layers
  (guardrails/postprocess) for extension runs.

Related benchmarks and metrics: Spider [22], distilled test suites [21], and
LLM text-to-SQL benchmark studies [3, 23].
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import sqlalchemy
from sqlalchemy.engine import Engine

from ..core.db import safe_connection
from ..core.llm import generate_sql_from_messages
from ..core.postprocess import guarded_postprocess, normalize_sql as _normalize_sql
from ..core.prompting import make_few_shot_messages
from ..core.query_runner import DEFAULT_FORBIDDEN_TOKENS, QueryRunner
from ..core.sql_guardrails import clean_candidate_with_reason


def _clean_sql(
    *,
    sql_text: str,
    nlq: str,
    apply_sql_guardrails: bool,
    apply_postprocess: bool,
) -> str:
    # Optional cleanup layer — disabled for primary runs, enabled only in extension runs.
    out = (sql_text or "").strip()

    if apply_sql_guardrails:
        cleaned, reason = clean_candidate_with_reason(out)
        if cleaned:
            out = cleaned
        elif reason == "empty":
            out = ""

    if apply_postprocess and out:
        out = guarded_postprocess(out, nlq)

    return out.strip()


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# frozen=True makes EvalItem immutable after creation. Each item is a permanent
# record of one evaluation: once scored it must not change. Immutability also
# makes EvalItem hashable so it can be stored in sets or used as a dict key.
@dataclass(frozen=True)
class EvalItem:
    i: int
    nlq: str
    gold_sql: str
    raw_sql: str      # exact model output before any post-processing
    pred_sql: str     # SQL actually scored (may differ from raw_sql if reliability layer is on)
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


def _safety_check(sql: str) -> None:
    lowered = (sql or "").strip().lower()
    if not lowered:
        raise ValueError("Empty SQL string")
    for token in DEFAULT_FORBIDDEN_TOKENS:
        if token in lowered:
            raise ValueError(f"Destructive SQL token detected: {token.strip()}")


def execute_fetch(
    *,
    engine: Engine,
    sql: str,
    max_rows: int = 10000,
) -> tuple[bool, list[str] | None, list[tuple] | None, str | None]:
    try:
        _safety_check(sql)
        with safe_connection(engine) as conn:
            result = conn.execute(sqlalchemy.text(sql))
            cols = list(result.keys())
            rows = result.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            return False, cols, None, f"Result set too large (> {max_rows} rows) for comparison"
        return True, cols, [tuple(r) for r in rows], None
    except Exception as e:
        return False, None, None, str(e)


def execution_accuracy(
    *,
    engine: Engine,
    pred_sql: str,
    gold_sql: str,
    max_compare_rows: int = 10000,
) -> tuple[bool, str | None, str | None]:
    pred_ok, _, pred_rows, pred_err = execute_fetch(engine=engine, sql=pred_sql, max_rows=max_compare_rows)
    gold_ok, _, gold_rows, gold_err = execute_fetch(engine=engine, sql=gold_sql, max_rows=max_compare_rows)

    if not gold_ok:
        return False, pred_err, gold_err
    if not pred_ok:
        return False, pred_err, gold_err

    # Counter gives bag/multiset equality: order-insensitive result comparison.
    # This matches the EX metric definition in the Spider benchmark (Yu et al. 2018).
    return Counter(pred_rows) == Counter(gold_rows), None, None


def _coerce_cell(x: Any) -> Any:
    # Normalise individual result cells before Counter comparison.
    # NaN must be checked before round() because math.isnan(None) raises TypeError,
    # and more importantly float('nan') != float('nan') in Python — two identical
    # NULL-like results would appear unequal without this guard.
    # round(x, 10) prevents floating-point precision drift (e.g. 3.9999999999 vs 4.0)
    # from causing a false mismatch between pred and gold result sets.
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
        _safety_check(sql)
        with safe_connection(engine) as conn:
            res = conn.execute(sqlalchemy.text(sql))
            rows = res.fetchmany(max_rows)
        return [tuple(_coerce_cell(v) for v in r) for r in rows]
    except Exception:
        return None


def test_suite_accuracy_for_item(
    *,
    make_engine_fn: Callable[[str], Engine],
    suite_db_names: list[str],
    gold_sql: str,
    pred_sql: str,
    max_rows: int = 2000,
) -> int:
    """Return 1 if pred_sql matches gold_sql on every usable perturbed database, else 0."""
    # ORDER BY queries need order-sensitive comparison; others use Counter (bag equality).
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


def _build_item_pool(
    *,
    item: dict[str, Any],
    test_set: list[dict[str, Any]],
    exemplar_pool: list[dict[str, Any]] | None,
    avoid_exemplar_leakage: bool,
) -> list[dict[str, Any]]:
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
    if k <= 0:
        return []
    if len(pool) < k:
        raise ValueError(f"Exemplar pool too small: k={k} but pool has {len(pool)} items")
    return rng.sample(pool, k)


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
    max_new_tokens: int,
    max_compare_rows: int,
    generation_extract_select: bool,
    generation_stop_on_semicolon: bool,
    apply_sql_guardrails: bool,
    apply_postprocess: bool,
    ts_suite_db_names: Optional[list[str]],
    ts_make_engine_fn: Optional[Callable[[str], Engine]],
    ts_max_rows: int,
) -> EvalItem:
    nlq = item["nlq"]
    gold_sql = item["sql"]

    messages = make_few_shot_messages(
        schema=schema_summary,
        exemplars=exemplars,
        nlq=nlq,
    )
    raw_sql = generate_sql_from_messages(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=max_new_tokens,
        extract_select=generation_extract_select,
        stop_on_semicolon=generation_stop_on_semicolon,
    )
    pred_sql = _clean_sql(
        sql_text=raw_sql,
        nlq=nlq,
        apply_sql_guardrails=apply_sql_guardrails,
        apply_postprocess=apply_postprocess,
    )

    meta = qr.run(pred_sql)
    em = _normalize_sql(pred_sql) == _normalize_sql(gold_sql)
    ex, ex_pred_err, ex_gold_err = execution_accuracy(
        engine=engine,
        pred_sql=pred_sql,
        gold_sql=gold_sql,
        max_compare_rows=max_compare_rows,
    )

    ts: Optional[int] = None
    if bool(meta.success) and ts_suite_db_names and ts_make_engine_fn and pred_sql:
        ts = test_suite_accuracy_for_item(
            make_engine_fn=ts_make_engine_fn,
            suite_db_names=ts_suite_db_names,
            gold_sql=gold_sql,
            pred_sql=pred_sql,
            max_rows=ts_max_rows,
        )

    return EvalItem(
        i=i,
        nlq=nlq,
        gold_sql=gold_sql,
        raw_sql=raw_sql,
        pred_sql=pred_sql,
        va=bool(meta.success),
        em=bool(em),
        ex=bool(ex),
        ts=ts,
        error=meta.error or ex_pred_err,
        gold_error=ex_gold_err,
    )


def _summarize_eval(out: list[EvalItem]) -> tuple[float, float, float, float | None, list[int]]:
    va_rate = sum(r.va for r in out) / max(len(out), 1)
    em_rate = sum(r.em for r in out) / max(len(out), 1)
    ex_rate = sum(r.ex for r in out) / max(len(out), 1)
    ts_values = [int(r.ts) for r in out if r.ts is not None]
    ts_rate = (sum(ts_values) / len(ts_values)) if ts_values else None
    return va_rate, em_rate, ex_rate, ts_rate, ts_values


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
    max_rows: int = 50,
    max_new_tokens: int = 128,
    run_metadata: Optional[dict[str, Any]] = None,
    avoid_exemplar_leakage: bool = True,
    max_compare_rows: int = 10000,
    ts_suite_db_names: Optional[list[str]] = None,
    ts_make_engine_fn: Optional[Callable[[str], Engine]] = None,
    ts_max_rows: int = 500,
    generation_extract_select: bool = False,
    generation_stop_on_semicolon: bool = False,
    apply_sql_guardrails: bool = False,
    apply_postprocess: bool = False,
) -> list[EvalItem]:
    # Isolated RNG keeps exemplar sampling reproducible for a fixed seed.
    rng = random.Random(seed)
    items = test_set[:limit] if limit else test_set

    qr = QueryRunner(engine, max_rows=max_rows)
    out: list[EvalItem] = []

    for i, item in enumerate(items):
        pool = _build_item_pool(
            item=item,
            test_set=test_set,
            exemplar_pool=exemplar_pool,
            avoid_exemplar_leakage=avoid_exemplar_leakage,
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
                max_new_tokens=max_new_tokens,
                max_compare_rows=max_compare_rows,
                generation_extract_select=generation_extract_select,
                generation_stop_on_semicolon=generation_stop_on_semicolon,
                apply_sql_guardrails=apply_sql_guardrails,
                apply_postprocess=apply_postprocess,
                ts_suite_db_names=ts_suite_db_names,
                ts_make_engine_fn=ts_make_engine_fn,
                ts_max_rows=ts_max_rows,
            )
        )

    va_rate, em_rate, ex_rate, ts_rate, ts_values = _summarize_eval(out)
    ts_s = "NA" if ts_rate is None else f"{ts_rate:.3f}"
    print(f"k={k} | n={len(out)} | VA={va_rate:.3f} | EM={em_rate:.3f} | EX={ex_rate:.3f} | TS={ts_s}")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        optional_reliability_enabled = any((
            generation_extract_select,
            generation_stop_on_semicolon,
            apply_sql_guardrails,
            apply_postprocess,
        ))
        eval_profile = "optional_reliability_layer" if optional_reliability_enabled else "model_only_raw"

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
            "eval_profile": eval_profile,
            "generation_extract_select": bool(generation_extract_select),
            "generation_stop_on_semicolon": bool(generation_stop_on_semicolon),
            "sql_guardrails": "enabled" if apply_sql_guardrails else "disabled",
            "postprocess": "guarded_postprocess" if apply_postprocess else "none",
            "results": [r.to_jsonable() for r in out],
        }
        if run_metadata:
            payload["run_metadata"] = run_metadata
        payload["exemplar_policy"] = "custom_pool" if exemplar_pool is not None else "benchmark_pool"
        payload["avoid_exemplar_leakage"] = bool(avoid_exemplar_leakage)

        save_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("Saved:", str(save_path))

    return out
