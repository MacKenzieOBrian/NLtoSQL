"""
Utilities to strengthen the ReAct-style NL->SQL agent without rewriting notebooks.

Provides:
- clean_candidate: strict SELECT-only filter to drop junk "Show SQL..." outputs.
- vanilla_candidate: deterministic few-shot baseline candidate (for fallback/rerank).
- semantic_score: lightweight lexical heuristic to rerank executable candidates.
- projection/intent helpers: enforce minimal projections and basic intent constraints.
- schema subset helpers: lightweight schema-linking for prompt reduction.

These helpers are intentionally lightweight and dependency-free so they can be
imported directly in notebooks (`notebooks/03_agentic_eval.ipynb`) or scripts.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from nl2sql.llm import extract_first_select
from nl2sql.prompting import make_few_shot_messages


def _normalize_spaced_keywords(text: str) -> str:
    # Regex reference: https://docs.python.org/3/library/re.html
    # Rationale: early traces showed "S E L E C T" / spaced keywords that break parsing.
    # Fix outputs like "S E L E C T" / "F R O M".
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


# --- Lightweight schema-linking helpers ---
# Justification: survey work repeatedly notes schema linking as the dominant NL->SQL
# bottleneck. We use a simple, transparent keyword-to-table map to reduce the
# prompt scope without injecting answer logic.
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
    """
    Validate that joins between known table pairs use expected key columns.
    Returns (ok, reason, detail). If no applicable join hints exist, returns ok.
    """
    if not sql or not sql.strip():
        return False, "empty_sql", {}

    sql_low = sql.lower()

    # Collect tables and aliases from FROM/JOIN.
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
        if alias and alias not in _SQL_KEYWORDS:
            alias_to_table[alias] = table
        alias_to_table[table] = table

    if len(tables_in_query) < 2:
        return True, "ok", {}

    # Gather observed join predicates table.col = table.col.
    observed: set[tuple[str, str, str, str]] = set()
    for m in re.finditer(
        r"(?is)\b([a-zA-Z_][\w$]*)\.([a-zA-Z_][\w$]*)\s*=\s*([a-zA-Z_][\w$]*)\.([a-zA-Z_][\w$]*)",
        sql_low,
    ):
        t1, c1, t2, c2 = m.group(1), m.group(2), m.group(3), m.group(4)
        t1 = alias_to_table.get(t1, t1)
        t2 = alias_to_table.get(t2, t2)
        observed.add((t1, c1, t2, c2))

    # For any table pair with a known join hint, require the key join.
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
            if not chunk:
                continue
            # expected pattern: a.b = c.d
            if "=" not in chunk:
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
    This is a light schema-linking step: it narrows the prompt scope,
    which reduces wrong-table selection without hardcoding answers.
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

        # Keyword-to-table mapping (auditable, deterministic).
        for key, tbls in _TABLE_HINTS.items():
            if key in nl and t in tbls:
                score += 3.0
                reasons.append(f"hint:{key}")

        # Table name overlap.
        t_tokens = _split_ident(t)
        overlap = t_tokens & nl_tokens
        if overlap:
            score += 2.0
            reasons.append("table_overlap")

        # Column name overlap (cap to avoid over-weighting).
        col_hits = 0
        col_lows = set(c.lower() for c in cols)
        for col in cols:
            c_tokens = _split_ident(col)
            if c_tokens & nl_tokens:
                col_hits += 1
        if col_hits:
            score += min(3.0, float(col_hits))
            reasons.append(f"col_hits:{col_hits}")

        # Explicit/projection hints: boost tables containing requested columns.
        if explicit_set & col_lows:
            score += 2.5
            reasons.append("explicit_fields")
        if projection_set & col_lows:
            score += 1.5
            reasons.append("projection_hints")
        if value_set & col_lows:
            score += 1.5
            reasons.append("value_columns")

        # If NLQ has explicit values and this table contains location columns, boost.
        if value_hints and (set(c.lower() for c in cols) & location_cols):
            score += 1.5
            reasons.append("location_cols")

        if score > 0:
            table_scores[t] = score
            table_reasons[t] = reasons

    # Relation-aware boost: include tables directly connected to relevant tables.
    # Rationale: join mistakes often occur when the linked schema omits a needed neighbor.
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

    # Rank tables by score.
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

    # Pre-compute join-key columns for selected tables (force include).
    table_lower_map = {t.lower(): t for t in tables.keys()}
    picked_lower = {t.lower() for t in picked}
    join_cols_by_table: dict[str, set[str]] = {t: set() for t in picked}
    for t1, c1, t2, c2 in _JOIN_HINT_PAIRS:
        if t1 in picked_lower and t2 in picked_lower:
            join_cols_by_table[table_lower_map[t1]].add(c1)
            join_cols_by_table[table_lower_map[t2]].add(c2)

    # Rank columns within each selected table (schema item ranking).
    # Rationale: relation-aware schema selection (RAT-SQL/RESDSQL) improves accuracy by
    # narrowing candidate columns before decoding.
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
        c_tokens = _split_ident(col)
        overlap = c_tokens & nl_tokens
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

        # Force include explicit fields and join keys for selected tables.
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

    # Filter join hints to selected tables for clarity.
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


# --- Projection contract + intent constraints ---
# Justification: execution validity (VA) is not sufficient for semantic correctness.
# We enforce only *output shape* constraints implied by the NLQ (not answers),
# which is consistent with constrained decoding/validation literature.
_FIELD_SYNONYMS = {
    "msrp": "MSRP",
    "msrps": "MSRP",
    "product code": "productCode",
    "product codes": "productCode",
    "product name": "productName",
    "product names": "productName",
    "product line": "productLine",
    "order number": "orderNumber",
    "order date": "orderDate",
    "order dates": "orderDate",
    "customer name": "customerName",
    "customer number": "customerNumber",
    "credit limit": "creditLimit",
    "phone": "phone",
    "city": "city",
    "country": "country",
    "state": "state",
    "postal code": "postalCode",
    "zip": "postalCode",
    "payment date": "paymentDate",
    "payment dates": "paymentDate",
    "check number": "checkNumber",
    "check numbers": "checkNumber",
    "office code": "officeCode",
    "employee number": "employeeNumber",
    "first name": "firstName",
    "last name": "lastName",
    "status": "status",
    "comments": "comments",
    "amount": "amount",
}

# Some terms are too ambiguous alone (e.g., "codes"). Use a context check.
_SPECIAL_FIELD_HINTS = {
    "codes": ("productCode", ["product"]),
}


def _explicit_field_list(nlq: str) -> list[str]:
    """
    Extract an explicit field list in NLQ order when the question enumerates fields
    (e.g., \"names, codes, and MSRPs\" or \"with city and country\").
    """
    nl = (nlq or "").lower()
    # Require an enumeration cue to avoid treating filter fields as projections.
    if not ("," in nl or " and " in nl or nl.startswith(("show", "list", "give", "display"))):
        return []

    hits = []
    for k, col in _FIELD_SYNONYMS.items():
        idx = nl.find(k)
        if idx != -1:
            hits.append((idx, col))
    for k, (col, ctx) in _SPECIAL_FIELD_HINTS.items():
        idx = nl.find(k)
        if idx != -1 and any(c in nl for c in ctx):
            hits.append((idx, col))
    if not hits:
        return []
    hits.sort(key=lambda x: x[0])
    ordered = []
    for _, col in hits:
        if col not in ordered:
            ordered.append(col)
    return ordered


_ENTITY_PROJECTION_HINTS = {
    "product line": ["productLine"],
    "product": ["productName", "productCode"],
    "customer": ["customerName"],
    "order": ["orderNumber", "orderDate"],
    "payment": ["checkNumber", "paymentDate", "amount"],
    "office": ["city", "country"],
    "employee": ["firstName", "lastName"],
}


_ENTITY_IDENTIFIER_FIELDS = {
    "customer": ["customerName"],
    "order": ["orderNumber"],
    "product": ["productCode"],
    "payment": ["checkNumber"],
}


def _projection_hints(nlq: str) -> list[str]:
    """
    Soft projection hints for schema item ranking.
    These are not enforced; they only bias column selection in the schema subset.
    """
    nl = (nlq or "").lower()
    hints = _explicit_field_list(nlq)

    # Only add entity-level hints when the NLQ asks to list entities.
    listing = bool(re.search(r"\b(list|show|which|display|give|find)\b", nl))
    listing = listing or bool(re.search(r"\b(top|highest|lowest|most|least|first|last)\b", nl))
    listing = listing or bool(re.search(r"\b(with|who|that)\b", nl))
    if not listing:
        return list(dict.fromkeys(hints))

    # Multi-word phrases first (e.g., "product line").
    for phrase, cols in _ENTITY_PROJECTION_HINTS.items():
        if " " in phrase and re.search(rf"\b{re.escape(phrase)}s?\b", nl):
            for c in cols:
                if c not in hints:
                    hints.append(c)

    for phrase, cols in _ENTITY_PROJECTION_HINTS.items():
        if " " in phrase:
            continue
        if re.search(rf"\b{re.escape(phrase)}s?\b", nl):
            for c in cols:
                if c not in hints:
                    hints.append(c)

    return list(dict.fromkeys(hints))


def _entity_identifier_fields(nlq: str) -> list[str]:
    """
    Entity identifiers to keep listings anchored (e.g., orderNumber, productCode).
    Conservative: only fire on listing-style queries to avoid over-blocking.
    """
    nl = (nlq or "").lower().strip()
    if not nl:
        return []

    listing = bool(re.search(r"\b(list|show|which|display|give|find)\b", nl))
    listing = listing or bool(re.search(r"\b(top|highest|lowest|most|least|first|last)\b", nl))
    listing = listing or bool(re.search(r"\b(with|who|that)\b", nl))
    if not listing:
        for phrase in _ENTITY_IDENTIFIER_FIELDS:
            if re.match(rf"^{re.escape(phrase)}s?\b", nl):
                listing = True
                break
    if not listing:
        return []

    hints: list[str] = []
    for phrase, cols in _ENTITY_IDENTIFIER_FIELDS.items():
        if re.search(rf"\b{re.escape(phrase)}s?\b", nl):
            for c in cols:
                if c not in hints:
                    hints.append(c)

    return list(dict.fromkeys(hints))


def _entity_projection_hints(nlq: str) -> list[str]:
    """
    Entity-level projection hints (no explicit field list).
    Used to ensure at least one entity identifier is returned for listing-style NLQs.
    """
    nl = (nlq or "").lower().strip()
    if not nl:
        return []

    listing = bool(re.search(r"\b(list|show|which|display|give|find)\b", nl))
    listing = listing or bool(re.search(r"\b(top|highest|lowest|most|least|first|last)\b", nl))
    listing = listing or bool(re.search(r"\b(with|who|that)\b", nl))
    if not listing:
        # If the NLQ starts with an entity term (e.g., "payments made by ..."),
        # treat it as listing even without explicit listing verbs.
        for phrase in _ENTITY_PROJECTION_HINTS:
            if re.match(rf"^{re.escape(phrase)}s?\b", nl):
                listing = True
                break

    if not listing:
        return []

    hints: list[str] = []
    # Multi-word phrases first (e.g., "product line").
    for phrase, cols in _ENTITY_PROJECTION_HINTS.items():
        if " " in phrase and re.search(rf"\b{re.escape(phrase)}s?\b", nl):
            for c in cols:
                if c not in hints:
                    hints.append(c)

    for phrase, cols in _ENTITY_PROJECTION_HINTS.items():
        if " " in phrase:
            continue
        if re.search(rf"\b{re.escape(phrase)}s?\b", nl):
            for c in cols:
                if c not in hints:
                    hints.append(c)

    return list(dict.fromkeys(hints))

_VALUE_COLUMN_HINTS = {
    "orderNumber": ["order number", "order id", "order #"],
    "customerNumber": ["customer number", "customer id", "customer #"],
    "employeeNumber": ["employee number", "employee id", "employee #"],
    "officeCode": ["office code"],
    "productCode": ["product code"],
    "checkNumber": ["check number", "check #"],
    "creditLimit": ["credit limit"],
    "amount": ["amount", "total payment", "payment amount"],
    "priceEach": ["price each", "price"],
    "buyPrice": ["buy price", "cost price"],
    "MSRP": ["msrp", "list price"],
    "quantityOrdered": ["quantity ordered", "qty ordered", "quantity"],
    "quantityInStock": ["quantity in stock", "stock", "inventory"],
    "status": ["status", "cancelled", "shipped", "resolved", "on hold", "disputed", "in process"],
    "orderDate": ["order date", "order dates", "ordered on"],
    "requiredDate": ["required date", "required by"],
    "shippedDate": ["shipped date", "shipping date"],
    "paymentDate": ["payment date", "paid on"],
    "country": ["country", "nation"],
    "city": ["city", "town"],
    "state": ["state", "province"],
    "postalCode": ["postal", "zip"],
}


_DATE_VALUE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_MONTH_NAMES = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}


def _looks_like_date(value: str) -> bool:
    v = (value or "").lower().strip()
    if not v:
        return False
    if _DATE_VALUE_RE.search(v):
        return True
    if any(m in v for m in _MONTH_NAMES):
        return True
    return False


def _value_linked_columns_from_tables(nlq: str, tables: dict[str, list[str]]) -> list[str]:
    """
    Lightweight value-to-column linking using NLQ context + value patterns.
    Returns canonical column names present in the schema.
    """
    if not tables:
        return []

    nl = (nlq or "").lower()
    value_hints = _extract_value_hints(nlq)

    # Map lower-case column names -> canonical column names.
    col_map: dict[str, str] = {}
    for cols in tables.values():
        for c in cols:
            col_map.setdefault(c.lower(), c)

    def _add(col: str, out: set[str]) -> None:
        key = col.lower()
        if key in col_map:
            out.add(col_map[key])

    linked: set[str] = set()

    # Phrase-based cues (most reliable).
    for col, phrases in _VALUE_COLUMN_HINTS.items():
        if any(p in nl for p in phrases):
            _add(col, linked)

    # Date-like values imply date columns if present.
    if any(_looks_like_date(v) for v in value_hints):
        for col in ["orderDate", "paymentDate", "requiredDate", "shippedDate"]:
            _add(col, linked)

    # Country/state codes (e.g., USA, UK).
    if any(re.fullmatch(r"[a-z]{2,3}", v) for v in value_hints):
        for col in ["country", "state"]:
            _add(col, linked)

    # Location phrases with values.
    if value_hints and re.search(r"\b(in|from|located|based)\b", nl):
        for col in ["city", "country", "state", "territory", "region"]:
            _add(col, linked)

    return sorted(linked)


def missing_explicit_fields(nlq: str, sql: str) -> list[str]:
    """Return explicitly requested fields that are missing from the SQL."""
    required = _explicit_field_list(nlq)
    if not required:
        return []
    sql_low = (sql or "").lower()
    return [c for c in required if c.lower() not in sql_low]


def _extract_required_columns(nlq: str) -> list[str]:
    nl = (nlq or "").lower()
    cols = _explicit_field_list(nlq)
    if cols:
        return cols
    # default list-style questions imply name-only
    if re.search(r"\b(which|list)\s+customers\b", nl) and "customerName" not in cols:
        cols.append("customerName")
    if re.search(r"\bcustomers?\s+(with|who|that)\b", nl) and "customerName" not in cols:
        cols.append("customerName")
    if re.search(r"\b(which|list)\s+products\b", nl) and "productName" not in cols:
        cols.append("productName")
    return cols


def _split_select_items(select_part: str) -> list[str]:
    # lightweight split: assumes no nested commas outside functions
    items = []
    depth = 0
    current = []
    for ch in select_part:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            items.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        items.append(tail)
    return items


def classify_intent(nlq: str) -> str:
    nl = (nlq or "").lower()
    if re.search(r"\b(top|highest|lowest|first|last|most|least)\b", nl):
        return "topk"
    if re.search(r"\b(per|by|each)\b", nl):
        return "grouped_aggregate"
    if re.search(r"\b(how many|number of|count|sum|average|avg|total|how much)\b", nl):
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
            # grouped aggregates handled in next intent
            return False, "aggregate_without_grouping"
    if intent == "grouped_aggregate":
        if not has_agg:
            return False, "grouped_requires_aggregate"
        if not has_group:
            return False, "grouped_requires_group_by"
    if intent == "topk":
        if not (has_order and has_limit):
            return False, "topk_requires_order_limit"
    return True, "ok"


# Prompt-echo trimming: models sometimes repeat instruction text ("output only SQL"),
# which makes the candidate non-executable. Keep this deterministic and transparent.
# Regex reference: https://docs.python.org/3/library/re.html
# Rationale: seen in Jan traces when the model echoed system rules into the SQL.
_ECHO_CUTOFF_RE = re.compile(
    r"(?is)\b("
    r"output\s+only|"
    r"no\s+explanation|"
    r"no\s+markdown|"
    r"respond\s+with|"
    r"show\s+output|"
    r"outputformatting|"
    r"output\s+formatting|"
    r"y/n"
    r")\b"
)

# Lightweight keyword set for rejecting keyword-soup candidates.
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
    """Extract a single executable SELECT statement (or explain why it was rejected).

    Used by the ReAct loop to:
    - drop prompt echo / markdown / multi-statement outputs that cause VA=0
    - keep the trace explainable ("rejected because ...")
    """
    if not raw:
        return None, "empty"

    text = _normalize_spaced_keywords(raw)
    sql = extract_first_select(text) or text
    sql = (sql or "").strip()
    lower = sql.lower()

    idx = lower.find("select")
    if idx == -1:
        return None, "no_select"
    sql = sql[idx:].strip()
    lower = sql.lower()

    # Cut off instruction echo that sometimes appears inside the same text span.
    # Rationale: avoids VA=0 caused by prompt echo in early baseline runs.
    m = _ECHO_CUTOFF_RE.search(sql)
    if m:
        sql = sql[: m.start()].strip()
        lower = sql.lower()

    # Cut at the first ';' so trailing text does not poison execution.
    # Rationale: model often adds explanations after the SQL.
    if ";" in sql:
        sql = sql.split(";", 1)[0].strip()
        lower = sql.lower()

    # Must contain FROM (allowing newlines/whitespace).
    # Rationale: ensures a real SELECT statement instead of partial output.
    if not re.search(r"\bfrom\b", lower):
        return None, "no_from"

    # Basic select-list sanity: reject empty / keyword-only projections.
    # Rationale: prevents "SELECT FROM" or keyword-soup outputs observed in early runs.
    m = re.search(r"(?is)^\s*select\s+(.*?)\s+from\s+", sql)
    if m:
        select_part = m.group(1).strip()
        if "*" not in select_part and "(" not in select_part:
            tokens = re.findall(r"[a-zA-Z_][\w$]*", select_part)
            if not tokens:
                return None, "no_select_fields"
            if all(t.lower() in _SQL_KEYWORDS for t in tokens):
                return None, "no_select_fields"

    # Basic FROM sanity: reject keyword-only table tokens (allow subqueries).
    if not re.search(r"(?is)\bfrom\s*\(", sql):
        m = re.search(r"(?is)\bfrom\s+([a-zA-Z_][\w$\.]*)(?:\s|$)", sql)
        if not m:
            return None, "no_from"
        if m.group(1).lower() in _SQL_KEYWORDS:
            return None, "no_from_table"

    # Keyword-soup heuristic: too many keywords, too few identifiers.
    # Rationale: rejects degenerate outputs that look like SQL but contain no fields/tables.
    tokens = re.findall(r"[a-zA-Z_][\w$]*", sql)
    if tokens:
        kw = sum(1 for t in tokens if t.lower() in _SQL_KEYWORDS)
        ident = sum(1 for t in tokens if t.lower() not in _SQL_KEYWORDS)
        if ident == 0:
            return None, "no_identifiers"
        if len(tokens) >= 12 and kw >= 3 * ident:
            return None, "keyword_soup"

    # Lightweight junk filters on trimmed SQL.
    bad_phrases = (
        "```",
        "answer:",
        "explanation",
    )
    if any(bp in lower for bp in bad_phrases):
        return None, "bad_phrase"

    # Reject instruction echoes like "SELECT statement only"
    if re.search(r"\bselect\s+(query|statement)\b", lower):
        return None, "instruction_echo"

    return sql.rstrip(";") + ";", "ok"


def clean_candidate(raw: str) -> Optional[str]:
    sql, _ = clean_candidate_with_reason(raw)
    return sql


def vanilla_candidate(
    nlq: str,
    schema_summary,
    tok,
    model,
    exemplars: Optional[Iterable[dict]] = None,
    max_new_tokens: int = 256,
):
    """
    Produce a deterministic few-shot candidate using the baseline prompt.
    Useful as a fallback when the ReAct loop finds no valid SQL.
    """
    from nl2sql.postprocess import guarded_postprocess

    msgs = make_few_shot_messages(schema=schema_summary, exemplars=exemplars or [], nlq=nlq)
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    raw_sql = extract_first_select(text) or text
    explicit_fields = _extract_required_columns(nlq)
    sql = guarded_postprocess(raw_sql, nlq, explicit_fields=explicit_fields)
    return clean_candidate(sql)


# Simple schema keyword map for lexical reranking
SCHEMA_KEYWORDS = {
    "country": ["country"],
    "customer": ["customer", "client", "buyer"],
    "order": ["order", "purchase"],
    "product": ["product", "item"],
    "office": ["office", "city", "location"],
    "employee": ["employee", "sales rep", "salesperson"],
    "total": ["total", "sum", "amount", "revenue"],
    "average": ["average", "avg", "mean"],
    "count": ["how many", "number of", "count"],
    "date": ["date", "year", "month"],
}


def count_select_columns(sql: str) -> int:
    lower = sql.lower()
    if "select" not in lower:
        return 99
    if not re.search(r"\bfrom\b", lower):
        return 99
    select_part = re.split(r"\bfrom\b", lower, 1)[0]
    select_part = select_part.split("select", 1)[1]
    return select_part.count(",") + 1


def semantic_score(nlq: str, sql: str) -> float:
    """
    Lightweight lexical score to prefer candidates whose columns/aggregates
    align with the NLQ intent. Not a true semantic parser, but better than
    "fewest columns".
    """
    nlq_low = nlq.lower()
    sql_low = sql.lower()
    score = 0.0

    # Penalize missing explicitly requested fields (if NLQ enumerates them).
    required_fields = _explicit_field_list(nlq)
    if required_fields:
        missing = [c for c in required_fields if c.lower() not in sql_low]
        if missing:
            score -= 3.0 * len(missing)
    else:
        # Soft projection hints when the NLQ implies listing entities.
        projection_hints = _projection_hints(nlq)
        if projection_hints:
            if any(h.lower() in sql_low for h in projection_hints):
                score += 1.5
            else:
                score -= 1.5

    for key, aliases in SCHEMA_KEYWORDS.items():
        if any(a in nlq_low for a in aliases) and key in sql_low:
            score += 2.0

    # Regex reference: https://docs.python.org/3/library/re.html
    # Rationale: lightweight lexical overlap helped pick better candidates than
    # shortest/first SQL in Jan candidate-ranking tests.
    nl_tokens = set(re.findall(r"[a-zA-Z]+", nlq_low))
    sql_tokens = set(re.findall(r"[a-zA-Z]+", sql_low))
    overlap = len(nl_tokens & sql_tokens)
    score += 0.1 * overlap

    # Reward presence of explicit NLQ values (e.g., "USA", "San Francisco").
    # This helps prioritize candidates that include the correct filter literals.
    value_hints = _extract_value_hints(nlq)
    if value_hints:
        if any(v in sql_low for v in value_hints):
            score += 2.0
        else:
            score -= 2.0

    if any(w in nlq_low for w in ["total", "sum", "revenue", "amount"]):
        if re.search(r"\b(sum|count|avg|max|min)\s*\(", sql_low, re.IGNORECASE):
            score += 3.0
        else:
            score -= 5.0

    if "each" in nlq_low or "per " in nlq_low:
        if re.search(r"\b(sum|count|avg|max|min)\s*\(", sql_low, re.IGNORECASE):
            score -= 2.0

    return score


_VALUE_STOPWORDS = {
    "List",
    "Show",
    "Which",
    "What",
    "How",
    "Count",
    "Total",
    "Average",
    "Top",
    "Highest",
    "Lowest",
    "First",
    "Last",
    "Most",
    "Least",
    "Per",
    "By",
    "Each",
    "Find",
    "Give",
    "Display",
    "Name",
    "Names",
    "Number",
}


def _extract_value_hints(nlq: str) -> list[str]:
    """Extract likely literal values from NLQ for scoring (lowercased)."""
    text = nlq or ""
    hints: set[str] = set()

    # Quoted strings are strong signals.
    for m in re.findall(r"\"([^\"]+)\"|'([^']+)'", text):
        for group in m:
            if group:
                hints.add(group)

    # Uppercase abbreviations (e.g., USA, UK).
    hints.update(re.findall(r"\b[A-Z]{2,}\b", text))

    # Multi-word proper nouns (e.g., San Francisco).
    hints.update(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text))

    # Numeric literals (e.g., 100000, 12.5).
    hints.update(re.findall(r"\b\d+(?:\.\d+)?\b", text))

    # Single capitalized words (filter common question words).
    for w in re.findall(r"\b[A-Z][a-z]+\b", text):
        if w not in _VALUE_STOPWORDS:
            hints.add(w)

    return [h.lower() for h in hints if h.strip()]
