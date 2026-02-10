"""
Lightweight SQL text guardrails for candidate cleanup.
"""

from __future__ import annotations

import re
from typing import Optional


def _normalize_spaced_keywords(text: str) -> str:
    """Fix tokenized SQL keywords like 'S E L E C T'."""
    keywords = [
        "select",
        "from",
        "where",
        "group",
        "by",
        "order",
        "limit",
        "join",
        "inner",
        "left",
        "right",
        "on",
        "having",
        "distinct",
    ]
    for kw in keywords:
        pattern = r"\\b" + "\\s*".join(list(kw)) + r"\\b"
        text = re.sub(pattern, kw.upper(), text, flags=re.I)
    return text


_ECHO_CUTOFF_RE = re.compile(
    r"(?is)\b("
    r"output\s+only|"
    r"no\s+explanation|"
    r"no\s+markdown|"
    r"respond\s+with|"
    r"show\s+output|"
    r"outputformatting|"
    r"output\s+formatting|"
    r"return\s+a\s+corrected\s+single\s+sql\s+select|"
    r"error\s*:|"
    r"y/n"
    r")\b"
)

_SQL_KEYWORDS = {
    "select",
    "from",
    "where",
    "group",
    "by",
    "order",
    "limit",
    "having",
    "join",
    "left",
    "right",
    "inner",
    "outer",
    "on",
    "as",
    "distinct",
    "union",
    "all",
    "exists",
    "in",
    "and",
    "or",
    "not",
    "case",
    "when",
    "then",
    "else",
    "end",
    "asc",
    "desc",
    "like",
    "between",
    "is",
    "null",
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "show",
    "explain",
    "analyze",
    "optimize",
    "repair",
    "checksum",
    "procedure",
    "call",
    "row",
    "rows",
}


def clean_candidate_with_reason(raw: str) -> tuple[Optional[str], str]:
    """
    Extract a single executable SELECT statement (or explain rejection reason).
    """
    if not raw:
        return None, "empty"

    text = _normalize_spaced_keywords(raw)
    # Defensive local import: notebook kernels may hold stale module globals.
    from nl2sql.llm import extract_first_select as _extract_first_select

    sql = _extract_first_select(text) or text
    sql = (sql or "").strip()
    lower = sql.lower()

    idx = lower.find("select")
    if idx == -1:
        return None, "no_select"
    sql = sql[idx:].strip()
    lower = sql.lower()

    m = _ECHO_CUTOFF_RE.search(sql)
    if m:
        sql = sql[: m.start()].strip()
        lower = sql.lower()

    if ";" in sql:
        sql = sql.split(";", 1)[0].strip()
        lower = sql.lower()

    if not re.search(r"\bfrom\b", lower):
        return None, "no_from"

    m = re.search(r"(?is)^\s*select\s+(.*?)\s+from\s+", sql)
    if m:
        select_part = m.group(1).strip()
        if "*" not in select_part and "(" not in select_part:
            tokens = re.findall(r"[a-zA-Z_][\w$]*", select_part)
            if not tokens:
                return None, "no_select_fields"
            if all(t.lower() in _SQL_KEYWORDS for t in tokens):
                return None, "no_select_fields"

    if not re.search(r"(?is)\bfrom\s*\(", sql):
        m = re.search(r"(?is)\bfrom\s+([a-zA-Z_][\w$\.]*)(?:\s|$)", sql)
        if not m:
            return None, "no_from"
        if m.group(1).lower() in _SQL_KEYWORDS:
            return None, "no_from_table"

    tokens = re.findall(r"[a-zA-Z_][\w$]*", sql)
    if tokens:
        kw = sum(1 for t in tokens if t.lower() in _SQL_KEYWORDS)
        ident = sum(1 for t in tokens if t.lower() not in _SQL_KEYWORDS)
        if ident == 0:
            return None, "no_identifiers"
        if len(tokens) >= 12 and kw >= 3 * ident:
            return None, "keyword_soup"

    bad_phrases = ("```", "answer:", "explanation")
    if any(bp in lower for bp in bad_phrases):
        return None, "bad_phrase"

    if re.search(r"\bselect\s+(query|statement)\b", lower):
        return None, "instruction_echo"

    return sql.rstrip(";") + ";", "ok"

