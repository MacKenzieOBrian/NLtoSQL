"""
Evaluation helpers (VA/EM/EX).
Refs: execution-based metrics common in NL->SQL work (e.g., Spider/EMNLP'20 TS:
https://aclanthology.org/2020.emnlp-main.29/) and Ojuri et al. style VA/EX.
SQLAlchemy execution docs: https://docs.sqlalchemy.org/en/20/core/connections.html#sqlalchemy.engine.Connection.execute


# Used here: execute predicted SQL, compare to gold results, and compute VA/EX/EM.
# What these are: VA = does it run, EX = does it return the right rows, EM = strict string match. 
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import sqlalchemy
from sqlalchemy.engine import Engine

from .db import safe_connection
from .llm import generate_sql_from_messages
from .postprocess import guarded_postprocess, normalize_sql
from .agent_utils import _extract_required_columns
from .prompting import make_few_shot_messages
from .query_runner import DEFAULT_FORBIDDEN_TOKENS, QueryRunner


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
            "error": self.error,
            "gold_error": self.gold_error,
        }


def _safety_check(sql: str) -> None:
    # This evaluation harness executes model-generated SQL.
    # Even in a controlled ClassicModels setting, we hard-block destructive tokens
    # to prevent accidental DB mutation during experiments.
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
            # Use SQLAlchemy TextClause so execution is consistent across DB-API drivers.
            result = conn.execute(sqlalchemy.text(sql))
            cols = list(result.keys())
            # Bound result size: large result sets are slow to compare and can blow up memory.
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
    # EX is strict and may penalise semantically equivalent SQL (projection/alias drift).
    # EX is computed by executing BOTH predicted and gold SQL and comparing results.
    # We intentionally compare row tuples as a multiset (Counter) to:
    # - ignore row order when no ORDER BY is specified
    # - preserve duplicate rows (set() would incorrectly drop duplicates)
    pred_ok, pred_cols, pred_rows, pred_err = execute_fetch(
        engine=engine, sql=pred_sql, max_rows=max_compare_rows
    )
    gold_ok, gold_cols, gold_rows, gold_err = execute_fetch(
        engine=engine, sql=gold_sql, max_rows=max_compare_rows
    )

    if not gold_ok:
        return False, pred_err, gold_err
    if not pred_ok:
        return False, pred_err, gold_err

    # Optionally align/reorder projected columns by gold column names, allowing extra columns
    # in the predicted query as long as all gold columns are present.
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

    from collections import Counter

    # NOTE: we compare rows only (not column names) because earlier experiments showed
    # EX was dominated by projection/alias drift even when the underlying row sets matched.
    # Rationale: keeps EX focused on semantic equivalence rather than presentation.
    return Counter(pred_rows) == Counter(gold_rows), None, None


# ----------------------------
# Test-Suite Accuracy (TS)
# ----------------------------
# Suite-based semantic check: compare gold vs predicted across multiple perturbed DB replicas.
# This mirrors the notebook TS harness but lives here for reuse in scripts.

TS_ORDER_BY_RE = re.compile(r"(?is)\border\s+by\b")
# Regex reference: https://docs.python.org/3/library/re.html
# Rationale: ordering only matters for TS when the gold query explicitly orders results.


def _has_order_by(sql: str) -> bool:
    return bool(TS_ORDER_BY_RE.search(sql or ""))


def _coerce_cell(x: Any) -> Any:
    """
    Normalize SQL result cells for robust equality.
    - keep None
    - convert NaN -> string (so NaN equals NaN)
    - round floats to reduce minor numeric drift
    """
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
    # Stable ordering for unordered comparisons; None sorts before strings/numbers.
    return sorted(rows, key=lambda t: tuple("" if v is None else v for v in t))


@dataclass
class TSQueryRun:
    ok: bool
    rows: Optional[list[tuple[Any, ...]]] = None
    cols: Optional[tuple[str, ...]] = None
    error: Optional[str] = None


def _run_select_ts(engine: Engine, sql: str, max_rows: int = 2000) -> TSQueryRun:
    """Execute SELECT and fetch up to max_rows. Returns rows + column names."""
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
    """Compare result sets; ordered if ORDER BY exists in gold."""
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
    """
    Returns (ts_pass, debug_info).

    strict_gold=True:
      if gold fails on any suite db, treat as TS=0 (suite generation bug / invalid gold on that db)
    strict_gold=False:
      ignore suite DBs where gold fails (uses remaining DBs)
    """
    # Gold defines the expected semantics; we treat results as ordered only when
    # the gold query contains ORDER BY. (Pred may include spurious ORDER BY.)
    ordered = _has_order_by(gold_sql)

    per_db: list[dict[str, Any]] = []
    usable = 0
    all_ok = True

    for db in suite_db_names:
        eng = make_engine_fn(db)

        g = _run_select_ts(eng, gold_sql, max_rows=max_rows)
        p = _run_select_ts(eng, pred_sql, max_rows=max_rows)

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
                # If gold SQL breaks on a TS replica, that replica is not a valid semantic test.
                # With strict_gold=True we treat this as TS=0 to avoid inflating TS.
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

        if g.cols is not None and p.cols is not None and len(g.cols) != len(p.cols):
            per_db.append({
                "db": db,
                "gold_ok": True,
                "pred_ok": True,
                "match": False,
                "ordered_compare": ordered,
                "gold_cols": g.cols,
                "pred_cols": p.cols,
            })
            all_ok = False
            continue

        match = _results_match_ts(g.rows or [], p.rows or [], ordered=ordered)
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
    postprocess: Callable[..., str] = guarded_postprocess,
    run_metadata: Optional[dict[str, Any]] = None,
    avoid_exemplar_leakage: bool = True,
    max_compare_rows: int = 10000,
    allow_extra_columns_ex: bool = False,
) -> list[EvalItem]:
    rng = random.Random(seed)
    items = test_set[:limit] if limit else test_set

    # QueryRunner provides the SELECT-only "execution gate" and VA signal.
    # We keep it inside eval_run so baseline + QLoRA use the same executor behavior.
    qr = QueryRunner(engine, max_rows=max_rows)
    out: list[EvalItem] = []

    for i, item in enumerate(items):
        nlq = item["nlq"]
        gold_sql = item["sql"]

        pool = exemplar_pool if exemplar_pool is not None else test_set
        if avoid_exemplar_leakage:
            # Avoid giving the model the exact test item as a few-shot exemplar.
            # This prevents inflated results due to copy/paste leakage.
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
        messages = make_few_shot_messages(schema=schema_summary, exemplars=exemplars, nlq=nlq)

        # Generation is deterministic by default (see nl2sql.llm.generate_sql_from_messages).
        # This keeps evaluation stable across reruns.
        raw_sql = generate_sql_from_messages(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=max_new_tokens,
        )
        # Postprocess is intentionally deterministic: it cleans common formatting/projection issues
        # without changing model weights. See nl2sql.postprocess.guarded_postprocess.
        explicit_fields = _extract_required_columns(nlq)
        pred_sql = postprocess(raw_sql, nlq, explicit_fields=explicit_fields)

        # VA (executability) from the QueryRunner.
        meta = qr.run(pred_sql, capture_df=False)
        # EM is diagnostic: strict surface-form match after normalization.
        em = normalize_sql(pred_sql) == normalize_sql(gold_sql)
        # EX is semantic: compare executed results vs gold results.
        ex, ex_pred_err, ex_gold_err = execution_accuracy(
            engine=engine,
            pred_sql=pred_sql,
            gold_sql=gold_sql,
            max_compare_rows=max_compare_rows,
            allow_extra_columns=allow_extra_columns_ex,
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
                error=meta.error or ex_pred_err,
                gold_error=ex_gold_err,
            )
        )

    va_rate = sum(r.va for r in out) / max(len(out), 1)
    em_rate = sum(r.em for r in out) / max(len(out), 1)
    ex_rate = sum(r.ex for r in out) / max(len(out), 1)
    print(f"k={k} | n={len(out)} | VA={va_rate:.3f} | EM={em_rate:.3f} | EX={ex_rate:.3f}")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "timestamp": now_utc_iso(),
            "k": k,
            "seed": seed,
            "limit": limit,
            "n": len(out),
            "va_rate": va_rate,
            "em_rate": em_rate,
            "ex_rate": ex_rate,
            "results": [r.to_jsonable() for r in out],
        }
        if run_metadata:
            payload["run_metadata"] = run_metadata
        payload["exemplar_policy"] = (
            "custom_pool" if exemplar_pool is not None else "benchmark_pool"
        )
        payload["avoid_exemplar_leakage"] = bool(avoid_exemplar_leakage)

        save_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("Saved:", str(save_path))

    return out
