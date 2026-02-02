"""
TS (Test Suite Accuracy) utilities.
Extracted from the notebook so the notebook stays readable.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Optional
import re
import math
from sqlalchemy import text
from sqlalchemy.engine import Engine

TS_ORDER_BY_RE = re.compile(r"(?is)order\s+by")

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
    return [tuple(_coerce_cell(v) for v in r) for r in rows]

def _sorted_rows(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return sorted(rows, key=lambda t: tuple("" if v is None else v for v in t))

@dataclass
class QueryRun:
    ok: bool
    rows: Optional[list[tuple[Any, ...]]] = None
    error: Optional[str] = None

def run_select(engine: Engine, sql: str, max_rows: int = 2000) -> QueryRun:
    try:
        with engine.connect() as conn:
            res = conn.execute(text(sql))
            fetched = res.fetchmany(max_rows)
            rows = _normalize_rows(fetched)
        return QueryRun(ok=True, rows=rows)
    except Exception as e:
        return QueryRun(ok=False, rows=None, error=str(e))

def results_match(gold_rows: list[tuple[Any, ...]], pred_rows: list[tuple[Any, ...]], ordered: bool) -> bool:
    if ordered:
        return gold_rows == pred_rows
    return _sorted_rows(gold_rows) == _sorted_rows(pred_rows)

def test_suite_accuracy_for_item(
    make_engine_fn,
    suite_db_names: list[str],
    gold_sql: str,
    pred_sql: str,
    *,
    max_rows: int = 2000,
    strict_gold: bool = True,
) -> tuple[int, dict]:
    ordered = _has_order_by(gold_sql) or _has_order_by(pred_sql)

    per_db = []
    usable = 0
    all_ok = True

    for db in suite_db_names:
        eng = make_engine_fn(db)
        g = run_select(eng, gold_sql, max_rows=max_rows)
        p = run_select(eng, pred_sql, max_rows=max_rows)

        if not g.ok:
            per_db.append({
                "db": db,
                "gold_ok": False,
                "pred_ok": p.ok,
                "gold_error": g.error,
                "pred_error": p.error if not p.ok else None,
                "match": False,
            })
            if strict_gold:
                all_ok = False
            continue

        usable += 1

        if not p.ok:
            per_db.append({
                "db": db,
                "gold_ok": True,
                "pred_ok": False,
                "gold_error": None,
                "pred_error": p.error,
                "match": False,
            })
            all_ok = False
            continue

        match = results_match(g.rows or [], p.rows or [], ordered=ordered)
        per_db.append({
            "db": db,
            "gold_ok": True,
            "pred_ok": True,
            "match": match,
            "ordered_compare": ordered,
            "gold_sample": (g.rows or [])[:10],
            "pred_sample": (p.rows or [])[:10],
        })
        if not match:
            all_ok = False

    if not strict_gold and usable == 0:
        all_ok = False

    ts = 1 if all_ok else 0
    debug = {"ordered_compare": ordered, "usable_dbs": usable, "per_db": per_db}
    return ts, debug
