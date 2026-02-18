"""
Deterministic NLQ -> SQL-structure constraints.

How to read this file:
1) Detect intent shape (aggregate/group/order/limit/distinct).
2) Add minimal required tables for known NL patterns.
3) Return one constraints dict used by generator + validator.

References:
- SQL SELECT clauses (GROUP BY / ORDER BY / LIMIT): https://dev.mysql.com/doc/refman/8.0/en/select.html
- Python regex docs: https://docs.python.org/3/library/re.html
"""

from __future__ import annotations

import re
from typing import Any

from .agent_schema_linking import _parse_schema_summary
from .constraint_hints import (
    _entity_identifier_fields,
    _entity_projection_hints,
    _extract_required_columns,
    _extract_value_hints,
    _projection_hints,
    _value_linked_columns_from_tables,
)


def _extend_unique(dst: list[str], values: list[str]) -> None:
    """Append values once, preserving order."""
    for value in values:
        if value and value not in dst:
            dst.append(value)


def _detect_agg(nl: str) -> str | None:
    """Map common NL aggregate cues to SQL aggregate functions."""
    if re.search(r"\b(count|how many|number of|total number)\b", nl):
        return "COUNT"
    if re.search(r"\b(average|avg|mean)\b", nl):
        return "AVG"
    if re.search(r"\b(total|sum|how much)\b", nl):
        return "SUM"
    if re.search(r"\b(maximum|max|highest|most)\b", nl):
        return "MAX"
    if re.search(r"\b(minimum|min|lowest|least)\b", nl):
        return "MIN"
    return None


def build_constraints(nlq: str, schema_text: str) -> dict[str, Any]:
    """Build a compact constraints object used by generation and validation."""
    nl = (nlq or "").lower()

    agg = _detect_agg(nl)
    needs_group_by = bool(agg and re.search(r"\b(per|by|each|for each)\b", nl))
    needs_order_by = bool(re.search(r"\b(top|highest|lowest|most|least|sorted|ranked|first|last)\b", nl))

    limit = None
    m = re.search(r"\b(top|first|last)\s+(\d+)\b", nl)
    if m:
        try:
            limit = int(m.group(2))
        except ValueError:
            limit = None

    distinct = bool(re.search(r"\b(unique|distinct|different)\b", nl))

    explicit_fields = _extract_required_columns(nlq)
    projection_hints = _projection_hints(nlq)
    entity_hints = _entity_projection_hints(nlq)
    entity_identifiers = _entity_identifier_fields(nlq)
    value_hints = _extract_value_hints(nlq)

    tables = _parse_schema_summary(schema_text)
    value_columns = _value_linked_columns_from_tables(nlq, tables)

    # Required output fields should be explicit and stable.
    required_output_fields = list(dict.fromkeys(explicit_fields))

    # Minimal table requirements by simple lexical triggers.
    required_tables: list[str] = []
    required_tables_all = False
    rule_tags: list[str] = []

    if re.search(r"\b(revenue|sales)\b", nl):
        _extend_unique(required_tables, ["orders", "orderdetails"])
        required_tables_all = True
        rule_tags.append("sales_requires_orderdetails")

    if re.search(r"\bpayment(s)?\b", nl) and re.search(r"\b(country|city|state)\b", nl):
        _extend_unique(required_tables, ["payments", "customers"])
        required_tables_all = True
        rule_tags.append("payments_location_requires_customers")

    if re.search(r"\bemployees?\b", nl) and re.search(r"\boffices?\b", nl):
        _extend_unique(required_tables, ["employees", "offices"])
        required_tables_all = True
        rule_tags.append("employees_offices_pair")

    if re.search(r"\bproducts?\b", nl) and re.search(r"\bproduct\s*line\b", nl):
        _extend_unique(required_tables, ["products", "productlines"])
        rule_tags.append("productline_pair")

    if re.search(r"\borders?\b", nl) and re.search(r"\bcustomers?\b", nl):
        _extend_unique(required_tables, ["orders", "customers"])
        rule_tags.append("orders_customers_pair")

    # Location requirement drives stricter downstream validation when values are present.
    needs_location = bool(
        re.search(r"\b(city|country|state|region|territory|office)\b", nl)
        or (value_hints and re.search(r"\b(in|from|located|based)\b", nl))
    )

    # Detect location-capable tables from schema text.
    location_cols = {"city", "country", "state", "territory", "region"}
    location_tables = sorted(
        [
            table
            for table, cols in tables.items()
            if set(c.lower() for c in cols) & location_cols
        ]
    )

    needs_self_join = bool(re.search(r"\bmanagers?\b", nl) and re.search(r"\bemployees?\b", nl))
    self_join_table = "employees" if needs_self_join else None

    explicit_projection = bool(explicit_fields)

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
