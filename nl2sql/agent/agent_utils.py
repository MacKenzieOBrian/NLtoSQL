"""
Backward-compatible facade for legacy imports.

This module only re-exports helper functions so old notebook imports still work.
If you are starting fresh, import from the focused modules directly.
"""

from __future__ import annotations

from .agent_schema_linking import (
    _build_fk_graph,
    _parse_schema_summary,
    _tables_connected,
    build_schema_subset,
    validate_join_hints,
)
from .constraint_hints import (
    _entity_identifier_fields,
    _entity_projection_hints,
    _explicit_field_list,
    _extract_required_columns,
    _extract_value_hints,
    _looks_like_date,
    _projection_hints,
    _value_linked_columns_from_tables,
)
from .intent_rules import classify_intent, intent_constraints
from ..core.sql_guardrails import _normalize_spaced_keywords, clean_candidate_with_reason


__all__ = [
    "_build_fk_graph",
    "_parse_schema_summary",
    "_tables_connected",
    "build_schema_subset",
    "validate_join_hints",
    "_entity_identifier_fields",
    "_entity_projection_hints",
    "_explicit_field_list",
    "_extract_required_columns",
    "_extract_value_hints",
    "_looks_like_date",
    "_projection_hints",
    "_value_linked_columns_from_tables",
    "classify_intent",
    "intent_constraints",
    "_normalize_spaced_keywords",
    "clean_candidate_with_reason",
]
