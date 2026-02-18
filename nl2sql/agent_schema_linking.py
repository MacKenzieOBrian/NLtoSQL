"""
Schema-linking and join-structure helpers for the ReAct NL->SQL agent.

How to read this file:
1) Rank tables with simple lexical signals from the NLQ.
2) Keep required columns and join key columns.
3) Emit join hints and validate join-key usage.

References:
- Python regex docs: https://docs.python.org/3/library/re.html
"""

from __future__ import annotations

import re

from .constraint_hints import (
    _extract_required_columns,
    _extract_value_hints,
    _projection_hints,
    _value_linked_columns_from_tables,
)


# NL words -> likely tables.
_TABLE_HINTS = {
    "customer": ["customers"],
    "client": ["customers"],
    "order": ["orders", "orderdetails"],
    "purchase": ["orders", "orderdetails"],
    "product": ["products", "productlines"],
    "vendor": ["products"],
    "payment": ["payments"],
    "office": ["offices"],
    "employee": ["employees"],
    "sales rep": ["employees"],
    "product line": ["productlines"],
}

# Canonical ClassicModels joins.
_JOIN_HINTS = [
    "orders.customerNumber = customers.customerNumber",
    "orderdetails.orderNumber = orders.orderNumber",
    "orderdetails.productCode = products.productCode",
    "products.productLine = productlines.productLine",
    "payments.customerNumber = customers.customerNumber",
    "employees.officeCode = offices.officeCode",
    "customers.salesRepEmployeeNumber = employees.employeeNumber",
]


def get_join_hints(tables: set[str] | None = None) -> list[str]:
    """Return join hints, filtered to the selected tables when provided."""
    if not tables:
        return list(_JOIN_HINTS)

    filtered: list[str] = []
    for hint in _JOIN_HINTS:
        parts = re.findall(r"([a-zA-Z_][\w$]*)\.", hint)
        if len(parts) < 2:
            continue
        if parts[0] in tables and parts[1] in tables:
            filtered.append(hint)
    return filtered or list(_JOIN_HINTS)


def format_join_hints(tables: set[str] | None = None) -> str:
    return "Join hints: " + "; ".join(get_join_hints(tables))


_SQL_ALIAS_KEYWORDS = {
    "select",
    "from",
    "where",
    "group",
    "by",
    "order",
    "join",
    "on",
    "as",
    "left",
    "right",
    "inner",
    "outer",
    "limit",
}


def _parse_join_hints() -> list[tuple[str, str, str, str]]:
    pairs: list[tuple[str, str, str, str]] = []
    for hint in _JOIN_HINTS:
        m = re.match(
            r"(?is)^\s*([a-zA-Z_][\w$]*)\.([a-zA-Z_][\w$]*)\s*=\s*([a-zA-Z_][\w$]*)\.([a-zA-Z_][\w$]*)\s*$",
            hint,
        )
        if not m:
            continue
        pairs.append((m.group(1).lower(), m.group(2).lower(), m.group(3).lower(), m.group(4).lower()))
    return pairs


_JOIN_HINT_PAIRS = _parse_join_hints()


def validate_join_hints(sql: str) -> tuple[bool, str, dict]:
    """Check that known table pairs use expected join key columns."""
    if not sql or not sql.strip():
        return False, "empty_sql", {}

    sql_low = sql.lower()

    alias_to_table: dict[str, str] = {}
    tables_in_query: set[str] = set()
    for m in re.finditer(
        r"(?is)\b(from|join)\s+([a-zA-Z_][\w$]*)(?:\s+(?:as\s+)?([a-zA-Z_][\w$]*))?",
        sql_low,
    ):
        table = m.group(2)
        if sql_low[m.end() : m.end() + 1] == "(":
            continue
        tables_in_query.add(table)

        alias = (m.group(3) or "").strip()
        if alias and alias not in _SQL_ALIAS_KEYWORDS:
            alias_to_table[alias] = table
        alias_to_table[table] = table

    if len(tables_in_query) < 2:
        return True, "ok", {}

    observed: set[tuple[str, str, str, str]] = set()
    for m in re.finditer(
        r"(?is)\b([a-zA-Z_][\w$]*)\.([a-zA-Z_][\w$]*)\s*=\s*([a-zA-Z_][\w$]*)\.([a-zA-Z_][\w$]*)",
        sql_low,
    ):
        t1 = alias_to_table.get(m.group(1), m.group(1))
        c1 = m.group(2)
        t2 = alias_to_table.get(m.group(3), m.group(3))
        c2 = m.group(4)
        observed.add((t1, c1, t2, c2))

    missing: list[str] = []
    for t1, c1, t2, c2 in _JOIN_HINT_PAIRS:
        if t1 in tables_in_query and t2 in tables_in_query:
            if (t1, c1, t2, c2) in observed or (t2, c2, t1, c1) in observed:
                continue
            missing.append(f"{t1}.{c1}={t2}.{c2}")

    if missing:
        return False, "missing_join_hint", {"missing": missing}
    return True, "ok", {}


def _parse_schema_summary(schema_summary: str) -> dict[str, list[str]]:
    tables: dict[str, list[str]] = {}
    for line in (schema_summary or "").splitlines():
        if "(" not in line:
            continue
        name, cols = line.split("(", 1)
        tables[name.strip()] = [c.strip() for c in cols.rstrip(")").split(",") if c.strip()]
    return tables


def _build_fk_graph(schema_text: str) -> dict[str, set[str]]:
    """Build an undirected table graph from join-hint lines."""
    graph: dict[str, set[str]] = {}
    for line in (schema_text or "").splitlines():
        if not line.lower().startswith("join hints:"):
            continue
        for chunk in line.split(":", 1)[1].split(";"):
            if "=" not in chunk:
                continue
            left, right = [x.strip() for x in chunk.split("=", 1)]
            if "." not in left or "." not in right:
                continue
            lt = left.split(".", 1)[0].strip()
            rt = right.split(".", 1)[0].strip()
            if not lt or not rt:
                continue
            graph.setdefault(lt, set()).add(rt)
            graph.setdefault(rt, set()).add(lt)
    return graph


def _tables_connected(schema_text: str, tables: set[str]) -> bool:
    if not tables or len(tables) == 1:
        return True
    graph = _build_fk_graph(schema_text)
    if not graph:
        return False
    start = next(iter(tables))
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for nxt in graph.get(node, set()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return tables.issubset(seen)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]+", (text or "").lower()))


def _split_ident(name: str) -> set[str]:
    # Split camel/snake-ish identifiers into lowercase tokens.
    parts = re.findall(r"[A-Z]?[a-z]+|[0-9]+", name or "")
    if not parts:
        parts = re.split(r"_+", name or "")
    tokens = [p.lower() for p in parts if p]
    low = (name or "").lower()
    if low:
        tokens.append(low)
        if low.endswith("s"):
            tokens.append(low[:-1])
    return set(t for t in tokens if t)


def build_schema_subset(
    schema_summary: str,
    nlq: str,
    max_tables: int = 6,
    max_cols_per_table: int = 8,
    return_debug: bool = False,
) -> str | tuple[str, dict]:
    """Return a reduced schema summary + join hints for the NLQ."""
    tables = _parse_schema_summary(schema_summary)
    if not tables:
        if return_debug:
            return schema_summary, {"selected_tables": [], "table_scores": {}, "join_hints": []}
        return schema_summary

    nl = (nlq or "").lower()
    nl_tokens = _tokenize(nlq)

    explicit_fields = _extract_required_columns(nlq)
    projection_hints = _projection_hints(nlq)
    value_hints = _extract_value_hints(nlq)
    value_columns = _value_linked_columns_from_tables(nlq, tables)

    explicit_set = {c.lower() for c in explicit_fields}
    projection_set = {c.lower() for c in projection_hints}
    value_set = {c.lower() for c in value_columns}

    table_scores: dict[str, float] = {}
    table_reasons: dict[str, list[str]] = {}

    for table, cols in tables.items():
        score = 0.0
        reasons: list[str] = []

        # Direct lexical table hints.
        for key, hinted_tables in _TABLE_HINTS.items():
            if key in nl and table in hinted_tables:
                score += 2.0
                reasons.append(f"hint:{key}")

        # Table-name overlap with question words.
        if _split_ident(table) & nl_tokens:
            score += 1.5
            reasons.append("table_overlap")

        col_lows = {c.lower() for c in cols}

        # Support explicit/projection/value cues.
        if explicit_set & col_lows:
            score += 2.0
            reasons.append("explicit_fields")
        if projection_set & col_lows:
            score += 1.5
            reasons.append("projection_hints")
        if value_set & col_lows:
            score += 1.5
            reasons.append("value_columns")

        # Small score for NL-token overlap with columns.
        col_hits = sum(1 for col in cols if _split_ident(col) & nl_tokens)
        if col_hits:
            score += min(2.0, 0.5 * col_hits)
            reasons.append(f"col_hits:{col_hits}")

        if score > 0:
            table_scores[table] = score
            table_reasons[table] = reasons

    # Fallback: if no hints fire, keep schema head tables for stability.
    if not table_scores:
        picked = list(tables.keys())[:max_tables]
    else:
        # Light relation boost: tables neighboring strong tables get a bump.
        adjacency: dict[str, set[str]] = {}
        for t1, _c1, t2, _c2 in _JOIN_HINT_PAIRS:
            adjacency.setdefault(t1, set()).add(t2)
            adjacency.setdefault(t2, set()).add(t1)

        lower_to_actual = {t.lower(): t for t in tables.keys()}
        boosts: dict[str, float] = {}
        for table, score in table_scores.items():
            del score
            for neighbor in adjacency.get(table.lower(), set()):
                actual = lower_to_actual.get(neighbor)
                if actual:
                    boosts[actual] = boosts.get(actual, 0.0) + 0.5
        for table, boost in boosts.items():
            table_scores[table] = table_scores.get(table, 0.0) + boost
            table_reasons.setdefault(table, []).append(f"relation_boost:{boost:.2f}")

        picked = [
            t for t, _ in sorted(table_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        ][:max_tables]

    picked_set = set(picked)

    # Keep join key columns for selected table pairs.
    lower_to_actual = {t.lower(): t for t in tables.keys()}
    picked_lower = {t.lower() for t in picked}
    join_cols_by_table: dict[str, set[str]] = {t: set() for t in picked}
    for t1, c1, t2, c2 in _JOIN_HINT_PAIRS:
        if t1 in picked_lower and t2 in picked_lower:
            a = lower_to_actual.get(t1)
            b = lower_to_actual.get(t2)
            if a:
                join_cols_by_table[a].add(c1)
            if b:
                join_cols_by_table[b].add(c2)

    selected_columns: dict[str, list[str]] = {}
    for table in picked:
        cols = tables[table]
        if max_cols_per_table is None or max_cols_per_table <= 0:
            selected = set(cols)
        else:
            selected = set(cols[:max_cols_per_table])

        # Always keep explicit/projection/value and join key columns.
        for col in cols:
            col_low = col.lower()
            if col_low in explicit_set or col_low in projection_set or col_low in value_set:
                selected.add(col)
            if col_low in join_cols_by_table.get(table, set()):
                selected.add(col)

        selected_columns[table] = [c for c in cols if c in selected]

    subset_lines = [f"{t}({', '.join(selected_columns[t])})" for t in picked]

    join_hints = get_join_hints(picked_set)

    subset = "\n".join(subset_lines + ["Join hints: " + "; ".join(join_hints)])

    if return_debug:
        return subset, {
            "selected_tables": picked,
            "table_scores": table_scores,
            "table_reasons": table_reasons,
            "explicit_fields": explicit_fields,
            "projection_hints": projection_hints,
            "value_hints": value_hints,
            "value_columns": value_columns,
            "join_hints": join_hints,
            "selected_columns": selected_columns,
        }
    return subset
