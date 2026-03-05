"""
Schema inspection helpers.

Queries INFORMATION_SCHEMA and builds the compact table(col, ...) text used in all prompts.
"""

from __future__ import annotations

import re

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .db import safe_connection


NAME_LIKE_RE = re.compile(r"name|id|line|code|number", re.IGNORECASE)


def list_tables(engine: Engine) -> list[str]:
    with safe_connection(engine) as conn:
        rows = conn.execute(text("SHOW TABLES;")).fetchall()
    return [r[0] for r in rows]


def get_table_columns(engine: Engine, *, db_name: str, table_name: str) -> pd.DataFrame:
    query = text(
        """
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_KEY
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :db AND TABLE_NAME = :table
        ORDER BY ORDINAL_POSITION
        """
    )
    with safe_connection(engine) as conn:
        # pandas.read_sql has adapter issues in some Colab environments; use raw execution.
        result = conn.execute(query, {"db": db_name, "table": table_name})
        rows = result.fetchall()

    if not rows:
        return pd.DataFrame(columns=["COLUMN_NAME", "DATA_TYPE", "IS_NULLABLE", "COLUMN_KEY"])

    return pd.DataFrame(rows, columns=["COLUMN_NAME", "DATA_TYPE", "IS_NULLABLE", "COLUMN_KEY"])


def build_schema_summary(engine: Engine, *, db_name: str, max_cols_per_table: int = 50) -> str:
    """Build compact schema text for model prompts: table(col1, col2, ...).

    Primary-key and name-like columns are listed first to surface join-relevant
    identifiers early in the token stream. Schema-linking motivation: Lin et al. [2].
    """
    chunks: list[str] = []
    for table in list_tables(engine):
        cols_df = get_table_columns(engine, db_name=db_name, table_name=table)

        priority_mask = cols_df["COLUMN_KEY"].fillna("").isin(["PRI"]) | cols_df["COLUMN_NAME"].astype(
            str
        ).str.contains(NAME_LIKE_RE, regex=True)
        priority = cols_df.loc[priority_mask, "COLUMN_NAME"].tolist()
        rest = [c for c in cols_df["COLUMN_NAME"].tolist() if c not in priority]
        cols = (priority + rest)[:max_cols_per_table]

        chunks.append(f"{table}({', '.join(cols)})")

    return "\n".join(chunks)
