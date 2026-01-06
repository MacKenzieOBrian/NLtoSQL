from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from sqlalchemy.engine import Engine

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
    ex: bool
    error: Optional[str]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "i": self.i,
            "nlq": self.nlq,
            "gold_sql": self.gold_sql,
            "raw_sql": self.raw_sql,
            "pred_sql": self.pred_sql,
            "va": self.va,
            "ex": self.ex,
            "error": self.error,
        }


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
        ex = normalize_sql(pred_sql) == normalize_sql(gold_sql)

        out.append(
            EvalItem(
                i=i,
                nlq=nlq,
                gold_sql=gold_sql,
                raw_sql=raw_sql,
                pred_sql=pred_sql,
                va=bool(meta.success),
                ex=bool(ex),
                error=meta.error,
            )
        )

    va_rate = sum(r.va for r in out) / max(len(out), 1)
    ex_rate = sum(r.ex for r in out) / max(len(out), 1)
    print(f"k={k} | n={len(out)} | VA={va_rate:.3f} | EX={ex_rate:.3f}")

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
