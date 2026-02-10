"""
Deterministic NLQ constraint extraction policy.

This module contains heuristic rules and lightweight intent signals that
describe what the SQL should structurally contain (aggregation, required
tables, output fields, location constraints, etc.).
"""

from __future__ import annotations

import re
from typing import Any

from .agent_schema_linking import _parse_schema_summary
from .constraint_hints import (
    _extract_required_columns,
    _extract_value_hints,
    _projection_hints,
    _value_linked_columns_from_tables,
)
from .validation import parse_schema_text


def _extend_unique(dst: list[str], values: list[str]) -> None:
    for v in values:
        if v and v not in dst:
            dst.append(v)


def _match_all(nl: str, patterns: list[str]) -> bool:
    return all(re.search(p, nl) for p in patterns)


def _match_any(nl: str, patterns: list[str]) -> bool:
    return any(re.search(p, nl) for p in patterns)


_CONSTRAINT_RULES: list[dict[str, Any]] = [
    {
        "tag": "template:order_totals_from_orderdetails",
        "all_patterns": [r"\border number\b"],
        "any_patterns": [r"\btotal\b", r"\bamount\b", r"\bsum\b", r"\bavg\b"],
        "tables": ["orderdetails"],
        "required_output_fields": ["orderNumber"],
        "require_all_tables": False,
    },
    {
        "tag": "template:payments_by_country_requires_join",
        "all_patterns": [r"\bpayments?\b", r"\b(per|by)\s+country\b"],
        "tables": ["payments", "customers"],
        "required_output_fields": ["country"],
        "require_all_tables": True,
    },
    {
        "tag": "template:top_customers_by_payments",
        "all_patterns": [r"\btop\s+\d+\s+customers?\b", r"\bpayments?\b"],
        "tables": ["payments", "customers"],
        "required_output_fields": ["customerName"],
        "require_all_tables": True,
    },
    {
        "tag": "template:avg_payment_by_country",
        "all_patterns": [r"\baverage\b", r"\bpayments?\b", r"\b(per|by)\s+country\b"],
        "tables": ["payments", "customers"],
        "required_output_fields": ["country"],
        "require_all_tables": True,
    },
    {
        "tag": "template:avg_msrp_by_product_line",
        "all_patterns": [r"\baverage\b", r"\bmsrp\b", r"\b(per|by)\s+product\s+line\b"],
        "tables": ["products"],
        "required_output_fields": ["productLine"],
        "require_all_tables": False,
    },
]


def build_constraints(nlq: str, schema_text: str) -> dict:
    """Build structural SQL constraints for an NLQ against a schema summary."""
    nl = (nlq or "").lower()

    agg = None
    if re.search(r"\bcount\b|how many|number of|total number of|count of", nl):
        agg = "COUNT"
    elif re.search(r"\baverage\b|\bavg\b|mean", nl):
        agg = "AVG"
    elif re.search(r"\btotal\b|\bsum\b|how much", nl):
        agg = "SUM"
    elif re.search(r"\bmaximum\b|\bmax\b|highest|most", nl):
        agg = "MAX"
    elif re.search(r"\bminimum\b|\bmin\b|lowest|least", nl):
        agg = "MIN"

    needs_group_by = bool(agg and re.search(r"\bper\b|\bby\b|for each|each", nl))
    needs_order_by = bool(
        re.search(r"\btop\b|\bhighest\b|\blowest\b|\bmost\b|\bleast\b|sorted|ranked|order by", nl)
    )

    limit = None
    m = re.search(r"\b(top|first|last)\s+(\d+)\b", nl)
    if m:
        try:
            limit = int(m.group(2))
        except ValueError:
            limit = None

    distinct = bool(re.search(r"\b(unique|distinct|different)\b", nl))

    value_hints = _extract_value_hints(nlq)
    explicit_fields = _extract_required_columns(nlq)
    projection_hints = _projection_hints(nlq)
    entity_hints: list[str] = []
    entity_identifiers: list[str] = []
    required_output_fields = list(dict.fromkeys(explicit_fields))
    rule_tags: list[str] = []
    explicit_projection = bool(
        explicit_fields
        and ("," in nl or " and " in nl or nl.strip().startswith(("show", "list", "give", "display")))
    )
    value_columns = _value_linked_columns_from_tables(nlq, _parse_schema_summary(schema_text))
    needs_location = bool(value_hints and re.search(r"\b(in|from|located|based|office)\b", nl))

    required_tables: list[str] = []
    required_tables_all = False

    for rule in _CONSTRAINT_RULES:
        all_patterns = rule.get("all_patterns") or []
        any_patterns = rule.get("any_patterns") or []
        if all_patterns and not _match_all(nl, all_patterns):
            continue
        if any_patterns and not _match_any(nl, any_patterns):
            continue
        _extend_unique(required_tables, rule.get("tables") or [])
        _extend_unique(required_output_fields, rule.get("required_output_fields") or [])
        if rule.get("require_all_tables"):
            required_tables_all = True
        rule_tags.append(str(rule.get("tag") or "rule"))

    if re.search(r"\b(revenue|sales)\b", nl):
        _extend_unique(required_tables, ["orders", "orderdetails"])
        required_tables_all = True
        rule_tags.append("template:revenue_sales_requires_orderdetails")

    if needs_location:
        if re.search(r"\boffice(s)?\b", nl) or re.search(r"\bemployees?\b", nl):
            _extend_unique(required_tables, ["offices"])
        if re.search(r"\bcustomers?\b", nl):
            _extend_unique(required_tables, ["customers"])
        rule_tags.append("template:location_requires_location_table")

    if re.search(r"\bemployees?\b", nl) and (needs_location or re.search(r"\boffice(s)?\b", nl)):
        _extend_unique(required_tables, ["employees", "offices"])
        required_tables_all = True
        rule_tags.append("template:employees_offices_pair")

    location_value_signal = bool(value_hints and re.search(r"\b(in|from|located|based)\b", nl))
    if agg == "COUNT" and re.search(r"\bemployees?\b", nl) and (
        re.search(r"\boffice(s)?\b", nl) or needs_location or location_value_signal
    ):
        _extend_unique(required_tables, ["employees", "offices"])
        required_tables_all = True
        rule_tags.append("template:employee_count_by_office_location")

    needs_self_join = False
    self_join_table = None
    if re.search(r"\bmanagers?\b", nl) and re.search(r"\bemployees?\b", nl):
        needs_self_join = True
        self_join_table = "employees"

    _, table_cols = parse_schema_text(schema_text)
    location_cols = {"city", "country", "state", "territory", "region"}
    location_tables = sorted([t for t, cols in table_cols.items() if cols & location_cols])

    return {
        "agg": agg,
        "needs_group_by": needs_group_by,
        "needs_order_by": needs_order_by,
        "limit": limit,
        "distinct": distinct,
        "value_hints": value_hints,
        "explicit_fields": explicit_fields,
        "required_output_fields": required_output_fields,
        "explicit_projection": explicit_projection,
        "projection_hints": projection_hints,
        "entity_hints": entity_hints,
        "entity_identifiers": entity_identifiers,
        "value_columns": value_columns,
        "required_tables": required_tables,
        "required_tables_all": required_tables_all,
        "needs_self_join": needs_self_join,
        "self_join_table": self_join_table,
        "needs_location": needs_location,
        "location_tables": location_tables,
        "rule_tags": list(dict.fromkeys(rule_tags)),
    }
