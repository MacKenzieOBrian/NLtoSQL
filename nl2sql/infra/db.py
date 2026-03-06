"""Database helpers used by the notebooks."""

from __future__ import annotations

import os
from contextlib import contextmanager
from getpass import getpass
from typing import Iterator

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


__all__ = [
    "connect_notebook_db",
    "create_engine_with_connector",
    "load_db_config_from_env_or_prompt",
    "safe_connection",
]
