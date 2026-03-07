"""Database helpers used by the notebooks."""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from getpass import getpass
from typing import Callable, Iterator

import sqlalchemy
from google.cloud.sql.connector import Connector
from sqlalchemy import text
from sqlalchemy.engine import Engine


def create_engine_with_connector(
    *,
    instance_connection_name: str,
    user: str,
    password: str,
    db_name: str,
) -> tuple[Engine, Connector]:
    """Build a SQLAlchemy engine backed by the Cloud SQL Python connector."""
    # SQLAlchemy's `creator=` hook is the clean way to hand off connection creation:
    # https://docs.sqlalchemy.org/en/20/core/engines.html#sqlalchemy.create_engine
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
    )
    return engine, connector


@contextmanager
def safe_connection(engine: Engine) -> Iterator[sqlalchemy.engine.Connection]:
    """Yield one DB connection and guarantee it closes afterwards."""
    # Lightweight context-manager pattern from Python stdlib:
    # https://docs.python.org/3/library/contextlib.html#contextlib.contextmanager
    conn = engine.connect()
    try:
        yield conn
    finally:
        conn.close()


def load_db_config_from_env_or_prompt(default_db_name: str = "classicmodels") -> dict[str, str]:
    """Read DB credentials from env vars first, then prompt in the notebook."""
    instance_connection_name = os.getenv("INSTANCE_CONNECTION_NAME")
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASS")
    db_name = os.getenv("DB_NAME") or default_db_name

    if not instance_connection_name:
        instance_connection_name = input("Enter INSTANCE_CONNECTION_NAME: ").strip()
    if not db_user:
        db_user = input("Enter DB_USER: ").strip()
    if not db_pass:
        db_pass = getpass("Enter DB_PASS: ")

    return {
        "instance_connection_name": instance_connection_name,
        "db_user": db_user,
        "db_pass": db_pass,
        "db_name": db_name,
    }


def connect_notebook_db(
    *,
    default_db_name: str = "classicmodels",
    verify: bool = False,
) -> tuple[Engine, Connector, dict[str, str]]:
    """Prompt for DB config when needed and return (engine, connector, config)."""
    config = load_db_config_from_env_or_prompt(default_db_name=default_db_name)
    engine, connector = create_engine_with_connector(
        instance_connection_name=config["instance_connection_name"],
        user=config["db_user"],
        password=config["db_pass"],
        db_name=config["db_name"],
    )
    if verify:
        with safe_connection(engine) as conn:
            conn.execute(text("SELECT 1"))
    return engine, connector, config


def make_cached_engine_factory(
    *,
    connector: Connector,
    instance_connection_name: str,
    user: str,
    password: str,
    maxsize: int = 32,
) -> Callable[[str], Engine]:
    """Return a cached helper that builds one Engine per database name.

    Useful for test-suite scoring where the notebook needs the same connection
    pattern across many replica databases.
    """

    @lru_cache(maxsize=maxsize)
    def _make_engine(db_name: str) -> Engine:
        def getconn_for_db():
            return connector.connect(
                instance_connection_name,
                "pymysql",
                user=user,
                password=password,
                db=db_name,
            )

        return sqlalchemy.create_engine("mysql+pymysql://", creator=getconn_for_db, future=True)

    return _make_engine


__all__ = [
    "connect_notebook_db",
    "create_engine_with_connector",
    "load_db_config_from_env_or_prompt",
    "make_cached_engine_factory",
    "safe_connection",
]
