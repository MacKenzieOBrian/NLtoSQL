"""
Intent classification and intent-to-SQL structural checks.
"""

from __future__ import annotations

import re


def classify_intent(nlq: str) -> str:
    nl = (nlq or "").lower()
    if re.search(r"\b(top|highest|lowest|first|last|most|least)\b", nl):
        return "topk"
    has_agg_cue = bool(
        re.search(
            r"\b(how many|number of|count|sum|average|avg|total|how much|minimum|min|maximum|max)\b",
            nl,
        )
    )
    if has_agg_cue and re.search(r"\b(per|by|each|for each)\b", nl):
        return "grouped_aggregate"
    if has_agg_cue:
        return "aggregate"
    return "lookup"


def intent_constraints(nlq: str, sql: str) -> tuple[bool, str]:
    intent = classify_intent(nlq)
    s = (sql or "").lower()

    has_agg = re.search(r"(?is)\b(sum|count|avg|min|max)\s*\(", s) is not None
    has_group = re.search(r"(?is)\bgroup\s+by\b", s) is not None
    has_order = re.search(r"(?is)\border\s+by\b", s) is not None
    has_limit = re.search(r"(?is)\blimit\b", s) is not None

    if intent == "lookup":
        if has_agg or has_group:
            return False, "lookup_disallows_aggregate"
    if intent == "aggregate":
        if not has_agg:
            return False, "aggregate_requires_fn"
        if has_group:
            return False, "aggregate_disallows_group_by"
    if intent == "grouped_aggregate":
        if not has_agg:
            return False, "grouped_requires_aggregate"
        if not has_group:
            return False, "grouped_requires_group_by"
    if intent == "topk":
        if not (has_order and has_limit):
            return False, "topk_requires_order_limit"
    return True, "ok"

