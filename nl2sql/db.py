"""
Database connection helpers.

How to read this file:
1) `create_engine_with_connector()` builds a SQLAlchemy engine via Cloud SQL Connector.
2) `safe_connection()` gives a short-lived connection context.

References:
- Cloud SQL connector docs: https://cloud.google.com/sql/docs/mysql/connect-run
- SQLAlchemy engine creator docs: https://docs.sqlalchemy.org/en/20/core/engines.html#custom-dbapi-connect
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import sqlalchemy
from google.cloud.sql.connector import Connector
from sqlalchemy.engine import Engine


def create_engine_with_connector(
    *,
    instance_connection_name: str,
    user: str,
    password: str,
    db_name: str,
) -> tuple[Engine, Connector]:
    connector = Connector()

    def getconn():
        return connector.connect(
            instance_connection_name,
            "pymysql",
            user=user,
            password=password,
            db=db_name,
        )

    engine: Engine = sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=getconn,
        future=True,
    )
    return engine, connector


@contextmanager
def safe_connection(engine: Engine) -> Iterator[sqlalchemy.engine.Connection]:
    conn = engine.connect()
    try:
        yield conn
    finally:
        conn.close()
