"""
Safe query executor.

Wraps SQLAlchemy to run model-generated SQL in a read-only, row-limited,
safety-checked manner. Every execution is logged as an immutable QueryResult.
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
    "drop ", "delete ", "truncate ", "alter ", "create ",
    "update ", "insert ", "grant ", "revoke ",
]


def check_sql_safety(sql: str, forbidden_tokens: Optional[list[str]] = None) -> None:
    """Raise ValueError if sql is empty or contains a destructive DML token."""
    tokens = forbidden_tokens if forbidden_tokens is not None else DEFAULT_FORBIDDEN_TOKENS
    lowered = (sql or "").strip().lower()
    if not lowered:
        raise ValueError("Empty SQL string")
    for token in tokens:
        if token in lowered:
            raise ValueError(f"Destructive SQL token detected: {token.strip()}")


@dataclass(frozen=True)
class QueryResult:
    """Immutable record of one execution — frozen so it cannot be altered after scoring."""
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
        return {
            "sql": self.sql, "params": self.params, "timestamp": self.timestamp,
            "success": self.success, "rowcount": self.rowcount, "truncated": self.truncated,
            "exec_time_s": self.exec_time_s, "error": self.error, "columns": self.columns,
        }


class QueryRunner:
    """Safe, logged SQL executor: check_sql_safety → safe_connection → execute → QueryResult.

    Single execution path — safety and logging cannot be bypassed by any caller.
    https://docs.sqlalchemy.org/en/20/core/connections.html
    """

    def __init__(self, engine: Engine, *, max_rows: int = 1000, forbidden_tokens: Optional[list[str]] = None):
        self.engine = engine
        self.max_rows = max_rows
        self.history: list[QueryResult] = []
        self.forbidden_tokens = forbidden_tokens or list(DEFAULT_FORBIDDEN_TOKENS)

    def _safety_check(self, sql: str) -> None:
        # Re-raises as QueryExecutionError so run() callers get one exception type.
        try:
            check_sql_safety(sql, self.forbidden_tokens)
        except ValueError as exc:
            raise QueryExecutionError(str(exc)) from exc

    def run(self, sql: str, *, params: Optional[dict[str, Any]] = None) -> QueryResult:
        """Execute sql and return a QueryResult. KEY METHOD — used by eval and ReAct loop."""
        timestamp = now_utc_iso()
        try:
            self._safety_check(sql)
            start = datetime.now(timezone.utc)

            with safe_connection(self.engine) as conn:
                result = conn.execute(sqlalchemy.text(sql), params or {})
                cols = list(result.keys())
                # +1 to detect truncation without materialising full result.
                rows = result.fetchmany(self.max_rows + 1)
                truncated = len(rows) > self.max_rows
                if truncated:
                    rows = rows[: self.max_rows]

            exec_time_s = (datetime.now(timezone.utc) - start).total_seconds()
            out = QueryResult(
                sql=sql, params=params, timestamp=timestamp,
                success=True, rowcount=min(len(rows), self.max_rows),
                truncated=bool(truncated), exec_time_s=exec_time_s,
                error=None, columns=cols,
            )
        except Exception as e:
            out = QueryResult(
                sql=sql, params=params, timestamp=timestamp,
                success=False, rowcount=0, truncated=False,
                exec_time_s=None, error=str(e), columns=None,
            )

        self.history.append(out)
        return out

    def last(self) -> Optional[QueryResult]:
        return self.history[-1] if self.history else None
