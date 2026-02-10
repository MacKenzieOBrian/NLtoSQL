"""
Schema-linking and join-structure helpers for the ReAct NL->SQL agent.
"""

from __future__ import annotations

import re

from .constraint_hints import (
    _extract_required_columns,
    _extract_value_hints,
    _projection_hints,
    _value_linked_columns_from_tables,
)


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
    """Return join hints, optionally filtered to tables present in the schema."""
    if not tables:
        return list(_JOIN_HINTS)
    filtered: list[str] = []
    for hint in _JOIN_HINTS:
        parts = re.findall(r"([a-zA-Z_][\\w$]*)\\.", hint)
        if len(parts) >= 2 and parts[0] in tables and parts[1] in tables:
            filtered.append(hint)
    return filtered or list(_JOIN_HINTS)


def format_join_hints(tables: set[str] | None = None) -> str:
    hints = get_join_hints(tables)
    return "Join hints: " + "; ".join(hints)

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
        t1, c1, t2, c2 = (m.group(1).lower(), m.group(2).lower(), m.group(3).lower(), m.group(4).lower())
        pairs.append((t1, c1, t2, c2))
    return pairs


_JOIN_HINT_PAIRS = _parse_join_hints()


def validate_join_hints(sql: str) -> tuple[bool, str, dict]:
    """Validate that joins between known table pairs use expected key columns."""
    if not sql or not sql.strip():
        return False, "empty_sql", {}

    sql_low = sql.lower()
    tables_in_query: set[str] = set()
    alias_to_table: dict[str, str] = {}
    for m in re.finditer(
        r"(?is)\b(from|join)\s+([a-zA-Z_][\w$]*)(?:\s+(?:as\s+)?([a-zA-Z_][\w$]*))?",
        sql_low,
    ):
        table = m.group(2)
        after = sql_low[m.end() : m.end() + 1]
        if after == "(":
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
        t1, c1, t2, c2 = m.group(1), m.group(2), m.group(3), m.group(4)
        t1 = alias_to_table.get(t1, t1)
        t2 = alias_to_table.get(t2, t2)
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
        cols = cols.rstrip(")")
        tables[name.strip()] = [c.strip() for c in cols.split(",") if c.strip()]
    return tables


def _build_fk_graph(schema_text: str) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    if not schema_text:
        return graph
    for line in schema_text.splitlines():
        if not line.lower().startswith("join hints:"):
            continue
        hints = line.split(":", 1)[1]
        for chunk in hints.split(";"):
            chunk = chunk.strip()
            if not chunk or "=" not in chunk:
                continue
            left, right = chunk.split("=", 1)
            left = left.strip()
            right = right.strip()
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


def build_schema_subset(
    schema_summary: str,
    nlq: str,
    max_tables: int = 6,
    max_cols_per_table: int = 8,
    return_debug: bool = False,
) -> str | tuple[str, dict]:
    """
    Return a reduced schema summary + join hints for the NLQ.
    """
    tables = _parse_schema_summary(schema_summary)
    nl = (nlq or "").lower()

    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z]+", (text or "").lower()))

    def _split_ident(name: str) -> set[str]:
        parts = re.findall(r"[A-Z]?[a-z]+|[0-9]+", name or "")
        if not parts:
            parts = re.split(r"_+", name or "")
        tokens = [p.lower() for p in parts if p]
        low = (name or "").lower()
        if low.endswith("s"):
            tokens.append(low[:-1])
        tokens.append(low)
        return set(t for t in tokens if t)

    nl_tokens = _tokenize(nlq)
    value_hints = _extract_value_hints(nlq)
    explicit_fields = _extract_required_columns(nlq)
    projection_hints = _projection_hints(nlq)
    value_columns = _value_linked_columns_from_tables(nlq, tables)
    explicit_set = {c.lower() for c in explicit_fields}
    projection_set = {c.lower() for c in projection_hints}
    value_set = {c.lower() for c in value_columns}

    location_cols = {"city", "country", "state", "territory", "region"}
    location_tables = sorted([t for t, cols in tables.items() if set(c.lower() for c in cols) & location_cols])

    table_scores: dict[str, float] = {}
    table_reasons: dict[str, list[str]] = {}
    for t, cols in tables.items():
        score = 0.0
        reasons: list[str] = []
        for key, tbls in _TABLE_HINTS.items():
            if key in nl and t in tbls:
                score += 3.0
                reasons.append(f"hint:{key}")
        t_tokens = _split_ident(t)
        if t_tokens & nl_tokens:
            score += 2.0
            reasons.append("table_overlap")
        col_hits = 0
        col_lows = {c.lower() for c in cols}
        for col in cols:
            if _split_ident(col) & nl_tokens:
                col_hits += 1
        if col_hits:
            score += min(3.0, float(col_hits))
            reasons.append(f"col_hits:{col_hits}")
        if explicit_set & col_lows:
            score += 2.5
            reasons.append("explicit_fields")
        if projection_set & col_lows:
            score += 1.5
            reasons.append("projection_hints")
        if value_set & col_lows:
            score += 1.5
            reasons.append("value_columns")
        if value_hints and ({c.lower() for c in cols} & location_cols):
            score += 1.5
            reasons.append("location_cols")
        if score > 0:
            table_scores[t] = score
            table_reasons[t] = reasons

    rel_boost = 0.75
    adjacency: dict[str, set[str]] = {}
    for t1, _c1, t2, _c2 in _JOIN_HINT_PAIRS:
        adjacency.setdefault(t1, set()).add(t2)
        adjacency.setdefault(t2, set()).add(t1)
    relation_boosts: dict[str, float] = {}
    for t in table_scores:
        for nb in adjacency.get(t, set()):
            relation_boosts[nb] = relation_boosts.get(nb, 0.0) + rel_boost
    for nb, boost in relation_boosts.items():
        if nb in table_scores:
            table_scores[nb] += boost
            table_reasons[nb].append(f"relation_boost:{boost:.2f}")
        else:
            table_scores[nb] = boost
            table_reasons[nb] = [f"relation_boost:{boost:.2f}"]

    picked = [t for t, _ in sorted(table_scores.items(), key=lambda kv: (-kv[1], kv[0]))][:max_tables]
    if not picked:
        if return_debug:
            return schema_summary, {
                "selected_tables": [],
                "table_scores": {},
                "table_reasons": {},
                "value_hints": value_hints,
                "explicit_fields": explicit_fields,
                "projection_hints": projection_hints,
                "value_columns": value_columns,
                "location_tables": location_tables,
                "join_hints": [],
            }
        return schema_summary

    table_lower_map = {t.lower(): t for t in tables.keys()}
    picked_lower = {t.lower() for t in picked}
    join_cols_by_table: dict[str, set[str]] = {t: set() for t in picked}
    for t1, c1, t2, c2 in _JOIN_HINT_PAIRS:
        if t1 in picked_lower and t2 in picked_lower:
            join_cols_by_table[table_lower_map[t1]].add(c1)
            join_cols_by_table[table_lower_map[t2]].add(c2)

    selected_columns: dict[str, list[str]] = {}
    column_scores: dict[str, dict[str, float]] = {}

    def _score_column(col: str, idx: int) -> float:
        score = 0.0
        col_low = col.lower()
        if col_low in explicit_set:
            score += 5.0
        if col_low in projection_set:
            score += 3.0
        if col_low in value_set:
            score += 3.5
        overlap = _split_ident(col) & nl_tokens
        if overlap:
            score += min(2.5, float(len(overlap)))
        if value_hints and col_low in location_cols:
            score += 1.0
        return score

    for t in picked:
        cols = tables[t]
        col_index = {c: i for i, c in enumerate(cols)}
        scores = {c: _score_column(c, col_index[c]) for c in cols}
        column_scores[t] = scores
        ranked = sorted(cols, key=lambda c: (-scores[c], col_index[c]))

        if max_cols_per_table is None or max_cols_per_table <= 0:
            selected = set(cols)
        else:
            selected = set(ranked[:max_cols_per_table])

        forced = set()
        for c in cols:
            c_low = c.lower()
            if c_low in explicit_set:
                forced.add(c)
            if c_low in join_cols_by_table.get(t, set()):
                forced.add(c)
        selected |= forced
        selected_columns[t] = [c for c in cols if c in selected]

    subset_lines = [f"{t}({', '.join(selected_columns[t])})" for t in picked]

    join_hints = []
    for hint in _JOIN_HINTS:
        parts = re.findall(r"([a-zA-Z_][\\w$]*)\\.", hint)
        if len(parts) >= 2:
            left, right = parts[0], parts[1]
            if left in picked and right in picked:
                join_hints.append(hint)
    if not join_hints:
        join_hints = _JOIN_HINTS[:]
    join_hint_text = "Join hints: " + "; ".join(join_hints)

    subset = "\n".join(subset_lines + [join_hint_text])
    if return_debug:
        return subset, {
            "selected_tables": picked,
            "table_scores": table_scores,
            "table_reasons": table_reasons,
            "value_hints": value_hints,
            "explicit_fields": explicit_fields,
            "projection_hints": projection_hints,
            "value_columns": value_columns,
            "location_tables": location_tables,
            "join_hints": join_hints,
            "selected_columns": selected_columns,
            "column_scores": column_scores,
        }
    return subset
