"""
Safe query executor.

How to read this file:
1) `QueryRunner` executes model SQL in read-only mode.
2) It blocks destructive keywords and caps returned rows.
3) Results are stored as `QueryResult` records for traceability.

References (project anchors):
- `REFERENCES.md#ref-zhong2020-ts`
- `REFERENCES.md#ref-gao2025-llm-sql`

Implementation docs:
- SQLAlchemy execute docs: https://docs.sqlalchemy.org/en/20/core/connections.html
- Python dataclasses docs: https://docs.python.org/3/library/dataclasses.html
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import sqlalchemy
from sqlalchemy.engine import Engine

from .db import safe_connection


class QueryExecutionError(Exception):
    pass


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


DEFAULT_FORBIDDEN_TOKENS = [
    "drop ",
    "delete ",
    "truncate ",
    "alter ",
    "create ",
    "update ",
    "insert ",
]
# rationale: simple select-only guard to keep evaluation safe and reproducible.


@dataclass(frozen=True)
class QueryResult:
    sql: str
    params: Optional[dict[str, Any]]
    timestamp: str
    success: bool
    rowcount: int
    truncated: bool
    exec_time_s: Optional[float]
    error: Optional[str]
    columns: Optional[list[str]]
    result_preview: Optional[pd.DataFrame]

    def to_jsonable(self) -> dict[str, Any]:
        d = {
            "sql": self.sql,
            "params": self.params,
            "timestamp": self.timestamp,
            "success": self.success,
            "rowcount": self.rowcount,
            "truncated": self.truncated,
            "exec_time_s": self.exec_time_s,
            "error": self.error,
            "columns": self.columns,
        }
        return d


class QueryRunner:
    def __init__(self, engine: Engine, *, max_rows: int = 1000, forbidden_tokens: Optional[list[str]] = None):
        self.engine = engine
        self.max_rows = max_rows
        self.history: list[QueryResult] = []
        self.forbidden_tokens = forbidden_tokens or list(DEFAULT_FORBIDDEN_TOKENS)

    def _safety_check(self, sql: str) -> None:
        lowered = (sql or "").strip().lower()
        if not lowered:
            raise QueryExecutionError("Empty SQL string")
        # motivation: this project executes model-generated sql against a real db.
        # A simple token blocklist is a pragmatic safety layer for evaluation runs.
        for token in self.forbidden_tokens:
            if token in lowered:
                raise QueryExecutionError(f"Destructive SQL token detected: {token.strip()}")

    def run(self, sql: str, *, params: Optional[dict[str, Any]] = None, capture_df: bool = True) -> QueryResult:
        timestamp = now_utc_iso()
        try:
            self._safety_check(sql)
            start = datetime.now(timezone.utc)

            with safe_connection(self.engine) as conn:
                result = conn.execute(sqlalchemy.text(sql), params or {})
                cols = list(result.keys())
                # bound how much we fetch: queryrunner is for gating + debugging previews,
                # not for full result materialization (ex/ts do their own bounded fetch).
                rows = result.fetchmany(self.max_rows + 1)
                truncated = len(rows) > self.max_rows
                if truncated:
                    rows = rows[: self.max_rows]

            end = datetime.now(timezone.utc)
            exec_time_s = (end - start).total_seconds()

            df = None
            if capture_df:
                # the preview dataframe is a debugging aid for notebooks; it is not used for scoring.
                df = pd.DataFrame(rows, columns=cols)

            out = QueryResult(
                sql=sql,
                params=params,
                timestamp=timestamp,
                success=True,
                rowcount=min(len(rows), self.max_rows),
                truncated=bool(truncated),
                exec_time_s=exec_time_s,
                error=None,
                columns=cols,
                result_preview=df,
            )
        except Exception as e:
            out = QueryResult(
                sql=sql,
                params=params,
                timestamp=timestamp,
                success=False,
                rowcount=0,
                truncated=False,
                exec_time_s=None,
                error=str(e),
                columns=None,
                result_preview=None,
            )

        self.history.append(out)
        return out

    def last(self) -> Optional[QueryResult]:
        return self.history[-1] if self.history else None

    def save_history(self, path: str) -> None:
        serializable = [h.to_jsonable() for h in self.history]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)
