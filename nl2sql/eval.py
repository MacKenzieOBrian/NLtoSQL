"""
Evaluation helpers (VA/EM/EX).
Refs: execution-based metrics common in NL→SQL work (e.g., Spider/EMNLP'20 TS:
https://aclanthology.org/2020.emnlp-main.29/) and Ojuri et al. style VA/EX.
SQLAlchemy execution docs: https://docs.sqlalchemy.org/en/20/core/connections.html#sqlalchemy.engine.Connection.execute


# Used here: execute predicted SQL, compare to gold results, and compute VA/EX/EM.
# This mirrors execution-based evaluation in NL→SQL literature, implemented by us.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import sqlalchemy
from sqlalchemy.engine import Engine

from .db import safe_connection
from .llm import generate_sql_from_messages
from .postprocess import enforce_minimal_projection, normalize_sql
from .prompting import make_few_shot_messages
from .query_runner import QueryRunner


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


FORBIDDEN_TOKENS = [
    "drop ",
    "delete ",
    "truncate ",
    "alter ",
    "create ",
    "update ",
    "insert ",
]


def _safety_check(sql: str) -> None:
    lowered = (sql or "").strip().lower()
    if not lowered:
        raise ValueError("Empty SQL string")
    for token in FORBIDDEN_TOKENS:
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

    if pred_cols != gold_cols:
        return False, "Column mismatch", None

    from collections import Counter

    return Counter(pred_rows) == Counter(gold_rows), None, None


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
    postprocess: Callable[[str, str], str] = enforce_minimal_projection,
    run_metadata: Optional[dict[str, Any]] = None,
    avoid_exemplar_leakage: bool = True,
    max_compare_rows: int = 10000,
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
        messages = make_few_shot_messages(schema=schema_summary, exemplars=exemplars, nlq=nlq)

        raw_sql = generate_sql_from_messages(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=max_new_tokens,
        )
        pred_sql = postprocess(raw_sql, nlq)

        meta = qr.run(pred_sql, capture_df=False)
        em = normalize_sql(pred_sql) == normalize_sql(gold_sql)
        ex, ex_pred_err, ex_gold_err = execution_accuracy(
            engine=engine,
            pred_sql=pred_sql,
            gold_sql=gold_sql,
            max_compare_rows=max_compare_rows,
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
