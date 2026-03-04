"""
Safe query executor.

Executes model-generated SQL in read-only mode. Blocks destructive keywords,
caps returned rows, and stores results as QueryResult records for traceability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

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
# simple blocklist to keep evaluation runs safe — blocks destructive statements.


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
        # model-generated sql runs against a real db, so check for destructive tokens first.
        for token in self.forbidden_tokens:
            if token in lowered:
                raise QueryExecutionError(f"Destructive SQL token detected: {token.strip()}")

    def run(self, sql: str, *, params: Optional[dict[str, Any]] = None) -> QueryResult:
        timestamp = now_utc_iso()
        try:
            self._safety_check(sql)
            start = datetime.now(timezone.utc)

            with safe_connection(self.engine) as conn:
                result = conn.execute(sqlalchemy.text(sql), params or {})
                cols = list(result.keys())
                # QueryRunner is for bounded preview execution, not full result materialization.
                rows = result.fetchmany(self.max_rows + 1)
                truncated = len(rows) > self.max_rows
                if truncated:
                    rows = rows[: self.max_rows]

            end = datetime.now(timezone.utc)
            exec_time_s = (end - start).total_seconds()

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
            )

        self.history.append(out)
        return out

    def last(self) -> Optional[QueryResult]:
        return self.history[-1] if self.history else None
