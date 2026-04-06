"""Read the live DB schema and compress it into prompt-ready text.

This core-layer module keeps schema introspection separate from prompt
construction so the same summary can be reused across notebooks and runs.
"""

from __future__ import annotations

import re

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..infra.db import safe_connection


# Put obvious identifier columns first to make the prompt easier for the model.
# ai note copilot: "regex for name/id/code column detection"
NAME_LIKE_RE = re.compile(r"name|id|line|code|number", re.IGNORECASE)


def list_tables(engine: Engine) -> list[str]:
    """Return the tables visible in the connected database."""
    with safe_connection(engine) as conn:
        rows = conn.execute(text("SHOW TABLES;")).fetchall()
    return [r[0] for r in rows]


def get_table_columns(engine: Engine, *, db_name: str, table_name: str) -> pd.DataFrame:
    """Return basic column metadata for one table from INFORMATION_SCHEMA."""
    query = text(
        """
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_KEY
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :db AND TABLE_NAME = :table
        ORDER BY ORDINAL_POSITION
        """
    )
    with safe_connection(engine) as conn:
        # Raw execution is more reliable here than pandas.read_sql in Colab.
        result = conn.execute(query, {"db": db_name, "table": table_name})
        rows = result.fetchall()

    if not rows:
        return pd.DataFrame(columns=["COLUMN_NAME", "DATA_TYPE", "IS_NULLABLE", "COLUMN_KEY"])

    return pd.DataFrame(rows, columns=["COLUMN_NAME", "DATA_TYPE", "IS_NULLABLE", "COLUMN_KEY"])


def build_schema_summary(engine: Engine, *, db_name: str, max_cols_per_table: int = 50) -> str:
    """Build compact prompt text in the form ``table(col1, col2, ...)`` with PK and name-like columns first."""
    chunks: list[str] = []
    for table in list_tables(engine):
        cols_df = get_table_columns(engine, db_name=db_name, table_name=table)

        # ai note copilot: "pandas mask to sort PK and name-like columns first"
        # Prioritize keys and name-like columns because they are most useful in NL questions.
        priority_mask = cols_df["COLUMN_KEY"].fillna("").isin(["PRI"]) | cols_df["COLUMN_NAME"].astype(
            str
        ).str.contains(NAME_LIKE_RE, regex=True)
        priority = cols_df.loc[priority_mask, "COLUMN_NAME"].tolist()
        rest = [c for c in cols_df["COLUMN_NAME"].tolist() if c not in priority]
        # Truncate very wide tables so the schema summary stays usable in prompts.
        cols = (priority + rest)[:max_cols_per_table]

        chunks.append(f"{table}({', '.join(cols)})")

    return "\n".join(chunks)
