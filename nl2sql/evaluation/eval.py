"""
Evaluation helpers for VA/EM/EX/TS.

Default mode:
1) Build few-shot prompts from schema + exemplars.
2) Generate raw model text with no decoding constraints.
3) Score VA / EM / EX (and optional TS).

Optional mode:
- Enable constrained decoding and reliability cleanup layers
  (guardrails/postprocess) for extension runs.
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
from typing import Any, Callable, Iterable, Optional

import sqlalchemy
from sqlalchemy.engine import Engine

from ..core.db import safe_connection
from ..core.llm import generate_sql_from_messages
from ..core.postprocess import guarded_postprocess
from ..core.prompting import make_few_shot_messages
from ..core.query_runner import DEFAULT_FORBIDDEN_TOKENS, QueryRunner
from ..core.sql_guardrails import clean_candidate_with_reason


def _normalize_sql(s: str) -> str:
    """Light SQL normalization used only for EM comparison."""
    return " ".join((s or "").strip().rstrip(";").lower().split())


def _passthrough_sql(s: str) -> str:
    """Keep model output unchanged except surrounding whitespace trim."""
    return (s or "").strip()


def _apply_optional_reliability_layer(
    *,
    sql_text: str,
    nlq: str,
    apply_sql_guardrails: bool,
    apply_postprocess: bool,
    explicit_fields: Iterable[str] | None = None,
    required_fields: Iterable[str] | None = None,
) -> str:
    out = _passthrough_sql(sql_text)

    if apply_sql_guardrails:
        cleaned, reason = clean_candidate_with_reason(out)
        if cleaned:
            out = cleaned
        elif reason == "empty":
            out = ""

    if apply_postprocess and out:
        out = guarded_postprocess(
            out,
            nlq,
            explicit_fields=explicit_fields,
            required_fields=required_fields,
        )

    return _passthrough_sql(out)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class EvalItem:
    i: int
    nlq: str
    gold_sql: str
    raw_sql: str
    pred_sql: str
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
    allow_extra_columns: bool = False,
) -> tuple[bool, str | None, str | None]:
    pred_ok, pred_cols, pred_rows, pred_err = execute_fetch(
        engine=engine,
        sql=pred_sql,
        max_rows=max_compare_rows,
    )
    gold_ok, gold_cols, gold_rows, gold_err = execute_fetch(
        engine=engine,
        sql=gold_sql,
        max_rows=max_compare_rows,
    )

    if not gold_ok:
        return False, pred_err, gold_err
    if not pred_ok:
        return False, pred_err, gold_err

    if allow_extra_columns and pred_cols and gold_cols:
        def _norm_col(c: str) -> str:
            c = (c or "").strip()
            c = c.split(".")[-1]
            c = c.strip("`\"[]")
            return c.lower()

        pred_map = {}
        for i, c in enumerate(pred_cols):
            key = _norm_col(c)
            if key not in pred_map:
                pred_map[key] = i

        gold_keys = [_norm_col(c) for c in gold_cols]
        if all(k in pred_map for k in gold_keys):
            idxs = [pred_map[k] for k in gold_keys]
            pred_rows = [tuple(r[i] for i in idxs) for r in pred_rows]

    return Counter(pred_rows) == Counter(gold_rows), None, None


TS_ORDER_BY_RE = re.compile(r"(?is)\border\s+by\b")


def _has_order_by(sql: str) -> bool:
    return bool(TS_ORDER_BY_RE.search(sql or ""))


def _coerce_cell(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, float):
        if math.isnan(x):
            return "NaN"
        return round(x, 10)
    return x


def _normalize_rows(rows: Iterable[Iterable[Any]]) -> list[tuple[Any, ...]]:
    out: list[tuple[Any, ...]] = []
    for r in rows:
        out.append(tuple(_coerce_cell(v) for v in r))
    return out


def _sorted_rows(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return sorted(rows, key=lambda t: tuple("" if v is None else v for v in t))


@dataclass
class TSQueryRun:
    ok: bool
    rows: Optional[list[tuple[Any, ...]]] = None
    cols: Optional[tuple[str, ...]] = None
    error: Optional[str] = None


def _run_select_ts(engine: Engine, sql: str, max_rows: int = 2000) -> TSQueryRun:
    try:
        _safety_check(sql)
        with safe_connection(engine) as conn:
            res = conn.execute(sqlalchemy.text(sql))
            cols = tuple(res.keys())
            fetched = res.fetchmany(max_rows)
            rows = _normalize_rows(fetched)
        return TSQueryRun(ok=True, rows=rows, cols=cols)
    except Exception as e:
        return TSQueryRun(ok=False, rows=None, cols=None, error=str(e))


def _results_match_ts(
    gold_rows: list[tuple[Any, ...]],
    pred_rows: list[tuple[Any, ...]],
    ordered: bool,
) -> bool:
    if ordered:
        return gold_rows == pred_rows
    return _sorted_rows(gold_rows) == _sorted_rows(pred_rows)


def test_suite_accuracy_for_item(
    *,
    make_engine_fn: Callable[[str], Engine],
    suite_db_names: list[str],
    gold_sql: str,
    pred_sql: str,
    max_rows: int = 2000,
    strict_gold: bool = True,
) -> tuple[int, dict]:
    ordered = _has_order_by(gold_sql)

    per_db: list[dict[str, Any]] = []
    usable = 0
    all_ok = True

    for db in suite_db_names:
        eng = make_engine_fn(db)

        g = _run_select_ts(eng, gold_sql, max_rows=max_rows)
        p = _run_select_ts(eng, pred_sql, max_rows=max_rows)

        if not g.ok:
            per_db.append(
                {
                    "db": db,
                    "gold_ok": False,
                    "pred_ok": p.ok,
                    "gold_error": g.error,
                    "pred_error": p.error if not p.ok else None,
                    "match": False,
                }
            )
            if strict_gold:
                all_ok = False
            continue

        usable += 1

        if not p.ok:
            per_db.append(
                {
                    "db": db,
                    "gold_ok": True,
                    "pred_ok": False,
                    "gold_error": None,
                    "pred_error": p.error,
                    "match": False,
                }
            )
            all_ok = False
            continue

        if g.cols is not None and p.cols is not None and len(g.cols) != len(p.cols):
            per_db.append(
                {
                    "db": db,
                    "gold_ok": True,
                    "pred_ok": True,
                    "match": False,
                    "ordered_compare": ordered,
                    "gold_cols": g.cols,
                    "pred_cols": p.cols,
                }
            )
            all_ok = False
            continue

        match = _results_match_ts(g.rows or [], p.rows or [], ordered=ordered)
        per_db.append(
            {
                "db": db,
                "gold_ok": True,
                "pred_ok": True,
                "match": match,
                "ordered_compare": ordered,
            }
        )
        if not match:
            all_ok = False

    if not strict_gold and usable == 0:
        all_ok = False

    ts = 1 if all_ok else 0
    return ts, {"ordered_compare": ordered, "usable_dbs": usable, "per_db": per_db}


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
    allow_extra_columns_ex: bool = False,
    ts_suite_db_names: Optional[list[str]] = None,
    ts_make_engine_fn: Optional[Callable[[str], Engine]] = None,
    ts_max_rows: int = 500,
    ts_strict_gold: bool = True,
    table_descriptions: str | None = None,
    generation_constrained: bool = False,
    generation_extract_select: bool = False,
    generation_stop_on_semicolon: bool = False,
    apply_sql_guardrails: bool = False,
    apply_postprocess: bool = False,
) -> list[EvalItem]:
    rng = random.Random(seed)
    items = test_set[:limit] if limit else test_set

    qr = QueryRunner(engine, max_rows=max_rows)
    out: list[EvalItem] = []

    for i, item in enumerate(items):
        nlq = item["nlq"]
        gold_sql = item["sql"]

        pool = exemplar_pool if exemplar_pool is not None else test_set
        if avoid_exemplar_leakage:
            pool = [
                ex
                for ex in pool
                if not (ex.get("nlq") == nlq and ex.get("sql") == gold_sql)
            ]

        if k > 0:
            if len(pool) < k:
                raise ValueError(f"Exemplar pool too small: k={k} but pool has {len(pool)} items")
            exemplars = rng.sample(pool, k)
        else:
            exemplars = []

        messages = make_few_shot_messages(
            schema=schema_summary,
            exemplars=exemplars,
            nlq=nlq,
            table_descriptions=table_descriptions,
        )

        # Generation options are configurable so reliability layers can be used as an optional extension.
        raw_sql = generate_sql_from_messages(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=max_new_tokens,
            constrained=generation_constrained,
            extract_select=generation_extract_select,
            stop_on_semicolon=generation_stop_on_semicolon,
        )
        pred_sql = _apply_optional_reliability_layer(
            sql_text=raw_sql,
            nlq=nlq,
            apply_sql_guardrails=apply_sql_guardrails,
            apply_postprocess=apply_postprocess,
            explicit_fields=item.get("explicit_fields"),
            required_fields=item.get("required_output_fields") or item.get("required_fields"),
        )

        meta = qr.run(pred_sql, capture_df=False)
        em = _normalize_sql(pred_sql) == _normalize_sql(gold_sql)
        ex, ex_pred_err, ex_gold_err = execution_accuracy(
            engine=engine,
            pred_sql=pred_sql,
            gold_sql=gold_sql,
            max_compare_rows=max_compare_rows,
            allow_extra_columns=allow_extra_columns_ex,
        )

        ts: Optional[int] = None
        if bool(meta.success) and ts_suite_db_names and ts_make_engine_fn and pred_sql:
            ts, _ = test_suite_accuracy_for_item(
                make_engine_fn=ts_make_engine_fn,
                suite_db_names=ts_suite_db_names,
                gold_sql=gold_sql,
                pred_sql=pred_sql,
                max_rows=ts_max_rows,
                strict_gold=ts_strict_gold,
            )

        out.append(
            EvalItem(
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
        )

    va_rate = sum(r.va for r in out) / max(len(out), 1)
    em_rate = sum(r.em for r in out) / max(len(out), 1)
    ex_rate = sum(r.ex for r in out) / max(len(out), 1)
    ts_values = [int(r.ts) for r in out if r.ts is not None]
    ts_rate = (sum(ts_values) / len(ts_values)) if ts_values else None
    ts_s = "NA" if ts_rate is None else f"{ts_rate:.3f}"
    print(f"k={k} | n={len(out)} | VA={va_rate:.3f} | EM={em_rate:.3f} | EX={ex_rate:.3f} | TS={ts_s}")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        optional_reliability_enabled = any(
            [
                generation_constrained,
                generation_extract_select,
                generation_stop_on_semicolon,
                apply_sql_guardrails,
                apply_postprocess,
            ]
        )
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
            "generation_constrained": bool(generation_constrained),
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
