"""Sanitize raw model text into one executable SQL candidate.

This core-layer helper strips obvious chat-output noise and rejects dangerous
tokens. It is not a full SQL validator and does not attempt semantic repair.
"""

from __future__ import annotations

import re
from typing import Optional

from .llm import extract_first_select as _extract_first_select
from .query_runner import DEFAULT_FORBIDDEN_TOKENS


def clean_candidate_with_reason(raw: str) -> tuple[Optional[str], str]:
    """
    Extract one safe-looking SELECT statement from raw model output.

    Returns:
      (sql, "ok") on success
      (None, reason) on rejection
    """
    if not raw or not raw.strip():
        return None, "empty"

    text = raw

    # Chat models often wrap SQL in fenced blocks; remove that presentation noise first.
    text = re.sub(r"```(.*?)```", r"\1", text, flags=re.S).strip()

    # This only extracts the first SELECT-like statement; deeper validation happens elsewhere.
    sql = _extract_first_select(text)
    if not sql:
        return None, "no_select"

    sql = sql.strip()
    if ";" in sql:
        sql = sql.split(";", 1)[0].strip() + ";"
    else:
        sql = sql.rstrip(";") + ";"

    # DEFAULT_FORBIDDEN_TOKENS tokens have trailing spaces ("drop "); strip for matching.
    low = sql.lower()
    if any(tok.strip() in low for tok in DEFAULT_FORBIDDEN_TOKENS):
        return None, "forbidden_sql"

    return sql, "ok"
