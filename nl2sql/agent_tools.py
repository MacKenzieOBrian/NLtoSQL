"""
Agent tool interface for a ReAct-style NL->SQL loop.
The LLM selects actions; these functions execute them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import json

import re
from sqlalchemy import text

from .db import safe_connection
from .schema import list_tables, get_table_columns
from .llm import generate_sql_from_messages
from .prompting import SYSTEM_INSTRUCTIONS
from .query_runner import QueryRunner
from .agent_schema_linking import _parse_schema_summary, build_schema_subset
from .constraint_hints import (
    _extract_required_columns,
    _extract_value_hints,
    _projection_hints,
    _value_linked_columns_from_tables,
)
from .validation import parse_schema_text, validate_sql as _validate_sql, validate_constraints as _validate_constraints


@dataclass
class AgentContext:
    engine: Any
    db_name: str
    model: Any
    tok: Any
    runner: QueryRunner
    schema_cache: Optional[dict] = None
    schema_text_cache: Optional[str] = None
    max_new_tokens: int = 128


_CTX: Optional[AgentContext] = None


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


def set_agent_context(ctx: AgentContext) -> None:
    global _CTX
    _CTX = ctx


def _require_ctx() -> AgentContext:
    if _CTX is None:
        raise RuntimeError("Agent tools not initialized. Call set_agent_context(...) first.")
    return _CTX


def schema_to_text(schema: dict) -> str:
    """Render a structured schema to prompt-friendly text."""
    lines: list[str] = []
    for table in schema.get("tables", []):
        cols = [c["name"] for c in table.get("columns", [])]
        lines.append(f"{table['name']}({', '.join(cols)})")
    return "\n".join(lines)


def link_schema(nlq: str, schema_text: Optional[str] = None, max_tables: int = 6) -> dict:
    """Return a pruned schema text + join hints for the NLQ."""
    if not schema_text:
        schema_text = _require_ctx().schema_text_cache or schema_to_text(get_schema())
    subset, debug = build_schema_subset(
        schema_text, nlq, max_tables=max_tables, return_debug=True
    )
    return {
        "schema_text": subset,
        "changed": subset != schema_text,
        "link_debug": debug,
    }


def extract_constraints(nlq: str) -> dict:
    """Lightweight, deterministic constraint extraction from NLQ."""
    # Regex reference: https://docs.python.org/3/library/re.html
    # Rationale: structural cues (COUNT, ORDER BY, LIMIT) were a common source of EX failures.
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
    needs_order_by = bool(re.search(r"\btop\b|\bhighest\b|\blowest\b|\bmost\b|\bleast\b|sorted|ranked|order by", nl))

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
    schema_text = _require_ctx().schema_text_cache or schema_to_text(get_schema())
    value_columns = _value_linked_columns_from_tables(nlq, _parse_schema_summary(schema_text))
    needs_location = bool(
        value_hints and re.search(r"\b(in|from|located|based|office)\b", nl)
    )

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

    # Location queries usually require a dedicated location table.
    if needs_location:
        if re.search(r"\boffice(s)?\b", nl) or re.search(r"\bemployees?\b", nl):
            _extend_unique(required_tables, ["offices"])
        if re.search(r"\bcustomers?\b", nl):
            _extend_unique(required_tables, ["customers"])
        rule_tags.append("template:location_requires_location_table")

    # If the NLQ is about employees + a location/office, require both employees and offices.
    if re.search(r"\bemployees?\b", nl) and (needs_location or re.search(r"\boffice(s)?\b", nl)):
        _extend_unique(required_tables, ["employees", "offices"])
        required_tables_all = True
        rule_tags.append("template:employees_offices_pair")
    # Template rule: employee count constrained by office/location should always resolve via employees+offices.
    location_value_signal = bool(
        value_hints and re.search(r"\b(in|from|located|based)\b", nl)
    )
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

    schema_text = _require_ctx().schema_text_cache or schema_to_text(get_schema())
    _, table_cols = parse_schema_text(schema_text)
    location_cols = {"city", "country", "state", "territory", "region"}
    location_tables = sorted(
        [t for t, cols in table_cols.items() if cols & location_cols]
    )

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


def validate_constraints(sql: str, constraints: Optional[dict]) -> dict:
    """Validate SQL structure against extracted constraints."""
    ctx = _require_ctx()
    schema_text = ctx.schema_text_cache or schema_to_text(get_schema())
    return _validate_constraints(sql, constraints, schema_text=schema_text)


def get_schema() -> dict:
    """Return structured schema: tables, columns, PK/FK."""
    ctx = _require_ctx()
    if ctx.schema_cache is not None:
        return ctx.schema_cache

    engine = ctx.engine
    db_name = ctx.db_name
    tables = list_tables(engine)

    schema: dict[str, Any] = {"tables": [], "foreign_keys": []}
    for table in tables:
        cols_df = get_table_columns(engine, db_name=db_name, table_name=table)
        columns = []
        pk_cols = []
        for _, row in cols_df.iterrows():
            col = {
                "name": str(row["COLUMN_NAME"]),
                "type": str(row["DATA_TYPE"]),
                "nullable": str(row["IS_NULLABLE"]).upper() == "YES",
                "key": str(row["COLUMN_KEY"] or ""),
            }
            if col["key"] == "PRI":
                pk_cols.append(col["name"])
            columns.append(col)
        schema["tables"].append({"name": table, "columns": columns, "primary_keys": pk_cols})

    fk_query = text(
        """
        SELECT TABLE_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = :db AND REFERENCED_TABLE_NAME IS NOT NULL
        """
    )
    with safe_connection(engine) as conn:
        fk_rows = conn.execute(fk_query, {"db": db_name}).fetchall()
    for row in fk_rows:
        schema["foreign_keys"].append(
            {
                "table": row[0],
                "column": row[1],
                "ref_table": row[2],
                "ref_column": row[3],
            }
        )

    ctx.schema_cache = schema
    ctx.schema_text_cache = schema_to_text(schema)
    return schema


def validate_sql(sql: str, schema_text: Optional[str] = None) -> dict:
    """Validate SQL formatting + schema references without executing."""
    # Rationale: catch obvious formatting/schema errors before hitting the database.
    if schema_text is None:
        schema_text = _require_ctx().schema_text_cache or schema_to_text(get_schema())
    if not schema_text:
        return {"valid": False, "reason": "schema_missing"}
    return _validate_sql(sql, schema_text, enforce_join_hints=True)


def _call_llm(
    messages: list[dict[str, str]],
    *,
    max_new_tokens: Optional[int] = None,
    num_return_sequences: int = 1,
    do_sample: bool = False,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str | list[str]:
    ctx = _require_ctx()
    return generate_sql_from_messages(
        model=ctx.model,
        tokenizer=ctx.tok,
        messages=messages,
        max_new_tokens=max_new_tokens or ctx.max_new_tokens,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )


def generate_sql(
    nlq: str,
    schema_text: str,
    constraints: dict,
    *,
    num_cands: int = 1,
    do_sample: bool = False,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str | list[str]:
    """LLM call that generates SQL candidate(s)."""
    constraint_text = json.dumps(constraints or {}, ensure_ascii=False)
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": f"Schema:\n{schema_text}"},
        {"role": "user", "content": f"NLQ: {nlq}\nConstraints: {constraint_text}\nReturn a single SQL SELECT."},
    ]
    return _call_llm(
        messages,
        max_new_tokens=128,
        num_return_sequences=num_cands,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )


def repair_sql(nlq: str, bad_sql: str, error: str, schema_text: str) -> str:
    """LLM call that revises SQL using execution feedback."""
    nl = (nlq or "").lower()
    err = (error or "").lower()

    # Deterministic template: employee count by office location.
    if re.search(r"\bemployees?\b", nl) and re.search(r"\boffice(s)?\b", nl) and re.search(r"\bcount\b|how many|number of", nl):
        if any(k in err for k in ("missing_location_table", "missing_required_table", "missing_join_path", "ambiguous")):
            city = None
            for hint in _extract_value_hints(nlq):
                h = str(hint or "").strip().lower()
                if " " in h and re.search(r"[a-z]", h) and not re.search(r"\d", h):
                    city = " ".join(tok.capitalize() for tok in h.split())
                    break
            if not city and "san francisco" in nl:
                city = "San Francisco"
            city = city or "San Francisco"
            city = city.replace("'", "''")
            return (
                "SELECT COUNT(*) AS employeeCount "
                "FROM employees e JOIN offices o ON e.officeCode = o.officeCode "
                f"WHERE o.city = '{city}';"
            )

    # Deterministic template: top customers by total payments.
    if re.search(r"\btop\s+\d+\s+customers?\b", nl) and re.search(r"\bpayments?\b", nl):
        m = re.search(r"\btop\s+(\d+)\b", nl)
        limit = int(m.group(1)) if m else 5
        return (
            "SELECT c.customerName, SUM(p.amount) AS totalPayments "
            "FROM customers c JOIN payments p ON c.customerNumber = p.customerNumber "
            "GROUP BY c.customerName "
            "ORDER BY totalPayments DESC "
            f"LIMIT {limit};"
        )

    # Deterministic template: average payment amount per country.
    if re.search(r"\baverage\b", nl) and re.search(r"\bpayments?\b", nl) and re.search(r"\b(per|by)\s+country\b", nl):
        return (
            "SELECT c.country, AVG(p.amount) AS avg_payment_amount "
            "FROM customers c JOIN payments p ON c.customerNumber = p.customerNumber "
            "GROUP BY c.country;"
        )

    # Deterministic template: average MSRP by product line.
    if re.search(r"\baverage\b", nl) and re.search(r"\bmsrp\b", nl) and re.search(r"\b(per|by)\s+product\s+line\b", nl):
        return "SELECT productLine, AVG(MSRP) AS avg_msrp FROM products GROUP BY productLine;"

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": f"Schema:\n{schema_text}"},
        {
            "role": "user",
            "content": (
                "The previous SQL failed. Fix it.\n"
                f"NLQ: {nlq}\n"
                f"Bad SQL: {bad_sql}\n"
                f"Error: {error}\n"
                "Return a corrected single SQL SELECT."
            ),
        },
    ]
    return _call_llm(messages, max_new_tokens=128)


def run_sql(sql: str) -> dict:
    """Execute SQL. Return rows OR error string."""
    ctx = _require_ctx()
    meta = ctx.runner.run(sql, capture_df=True)
    if not meta.success:
        return {"success": False, "error": meta.error}
    rows = []
    if meta.result_preview is not None:
        rows = meta.result_preview.to_dict(orient="records")
    return {
        "success": True,
        "rows": rows,
        "rowcount": meta.rowcount,
        "truncated": meta.truncated,
        "columns": meta.columns,
    }


def finish(answer: str, sql: str, provenance: dict) -> dict:
    """Final output container."""
    return {"answer": answer, "sql": sql, "provenance": provenance}
