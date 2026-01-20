"""
Safe query executor.
Refs:
- SQLAlchemy connection/execute docs: https://docs.sqlalchemy.org/en/20/core/connections.html
- GCP connector examples (custom creator) and safe SELECT-only guards used in NL→SQL eval practice.
Purpose: give the ReAct loop a controlled Act step and enforce read-only VA/EX runs.
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


@dataclass(frozen=True)
class QueryResult:
    sql: str
    params: Optional[dict[str, Any]]
    timestamp: str
    success: bool
    rowcount: int
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
        self.forbidden_tokens = forbidden_tokens or [
            "drop ",
            "delete ",
            "truncate ",
            "alter ",
            "create ",
            "update ",
            "insert ",
        ]

    def _safety_check(self, sql: str) -> None:
        lowered = (sql or "").strip().lower()
        if not lowered:
            raise QueryExecutionError("Empty SQL string")
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
                rows = result.fetchall()
                cols = list(result.keys())

            end = datetime.now(timezone.utc)
            exec_time_s = (end - start).total_seconds()

            df = None
            if capture_df:
                df = pd.DataFrame(rows, columns=cols)
                if len(df) > self.max_rows:
                    df = df.iloc[: self.max_rows]

            out = QueryResult(
                sql=sql,
                params=params,
                timestamp=timestamp,
                success=True,
                rowcount=min(len(rows), self.max_rows),
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
