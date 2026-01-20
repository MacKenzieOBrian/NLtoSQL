"""
Post-processing for model SQL text.
Refs: common NL→SQL cleanup after generation; HF generation tips:
https://huggingface.co/docs/transformers/en/main_classes/text_generation

# Used here: normalize whitespace/case, grab the first SELECT, and optionally enforce
# minimal projection on list-style queries. All heuristics are ours.
"""

from __future__ import annotations

import re


def normalize_sql(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\\s+", " ", s)
    s = s.rstrip(";")
    return s.lower()


LIST_ALL_RE = re.compile(r"(?is)^\\s*list\\s+all\\s+")
SELECT_LIST_RE = re.compile(r"(?is)^\\s*select\\s+(.*?)\\s+from\\s+", re.DOTALL)


def enforce_minimal_projection(sql: str, nlq: str) -> str:
    if not sql or not nlq:
        return sql
    if not LIST_ALL_RE.search(nlq.strip()):
        return sql
    m = SELECT_LIST_RE.search(sql)
    if not m:
        return sql
    select_part = m.group(1).strip()
    if "*" in select_part:
        return sql
    first_expr = select_part.split(",")[0].strip()
    rebuilt = re.sub(SELECT_LIST_RE, lambda _: f"SELECT {first_expr} FROM ", sql, count=1)
    return rebuilt
