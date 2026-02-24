"""
Minimal NLQ hint extractors.

How to read this file:
1) Find phrase-level field hints (for SELECT columns).
2) Find value hints (for WHERE values).
3) Map value hints to likely schema columns.

References (project anchors):
- `REFERENCES.md#ref-li2023-resdsql`
- `REFERENCES.md#ref-zhu2024-survey`

Implementation docs:
- Python regex docs: https://docs.python.org/3/library/re.html
"""

from __future__ import annotations

import re


# common nl phrases -> schema columns.
_FIELD_PHRASES = {
    "customer name": "customerName",
    "customer number": "customerNumber",
    "order number": "orderNumber",
    "order date": "orderDate",
    "product name": "productName",
    "product code": "productCode",
    "product line": "productLine",
    "payment date": "paymentDate",
    "check number": "checkNumber",
    "first name": "firstName",
    "last name": "lastName",
    "office code": "officeCode",
    "employee number": "employeeNumber",
    "credit limit": "creditLimit",
    "postal code": "postalCode",
    "zip": "postalCode",
    "country": "country",
    "city": "city",
    "state": "state",
    "status": "status",
    "amount": "amount",
    "msrp": "MSRP",
}

# entity words -> useful output fields for listing questions.
_ENTITY_PROJECTION_FIELDS = {
    "customer": ["customerName"],
    "order": ["orderNumber", "orderDate"],
    "product": ["productName", "productCode"],
    "payment": ["checkNumber", "paymentDate", "amount"],
    "employee": ["firstName", "lastName"],
    "office": ["city", "country"],
}

# entity words -> primary identifier field.
_ENTITY_IDENTIFIER_FIELDS = {
    "customer": ["customerName"],
    "order": ["orderNumber"],
    "product": ["productCode"],
    "payment": ["checkNumber"],
    "employee": ["employeeNumber"],
    "office": ["officeCode"],
}

# column hints used when mapping nl values to likely where columns.
_VALUE_COLUMN_HINTS = {
    "country": ["country", "nation"],
    "city": ["city", "town"],
    "state": ["state", "province"],
    "postalCode": ["postal", "zip"],
    "status": ["status", "shipped", "cancelled", "on hold", "disputed", "resolved", "in process"],
    "paymentDate": ["payment date", "paid on"],
    "orderDate": ["order date", "ordered on"],
    "customerNumber": ["customer number", "customer id"],
    "orderNumber": ["order number", "order id"],
    "productCode": ["product code"],
    "checkNumber": ["check number"],
    "creditLimit": ["credit limit"],
    "amount": ["amount", "total payment", "payment amount"],
    "quantityOrdered": ["quantity", "qty"],
}

_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _unique(values: list[str]) -> list[str]:
    """Keep first-seen order while removing duplicates."""
    out: list[str] = []
    for v in values:
        if v and v not in out:
            out.append(v)
    return out


def _find_phrase_hits(nlq: str, mapping: dict[str, str]) -> list[str]:
    """Return mapped columns in the same order they appear in the NLQ."""
    nl = (nlq or "").lower()
    hits: list[tuple[int, str]] = []
    for phrase, col in mapping.items():
        idx = nl.find(phrase)
        if idx != -1:
            hits.append((idx, col))
    hits.sort(key=lambda x: x[0])
    return _unique([col for _, col in hits])


def _looks_like_date(value: str) -> bool:
    """Compatibility helper used by legacy imports."""
    if not value:
        return False
    return bool(_DATE_RE.search(str(value)))


def _explicit_field_list(nlq: str) -> list[str]:
    """Columns explicitly requested by NL wording."""
    return _find_phrase_hits(nlq, _FIELD_PHRASES)


def _extract_required_columns(nlq: str) -> list[str]:
    """Public alias for explicit requested output fields."""
    return _explicit_field_list(nlq)


def _entity_projection_hints(nlq: str) -> list[str]:
    """Entity-level projection hints for list/show questions."""
    nl = (nlq or "").lower()
    out: list[str] = []
    for entity, fields in _ENTITY_PROJECTION_FIELDS.items():
        if re.search(rf"\b{re.escape(entity)}s?\b", nl):
            out.extend(fields)
    return _unique(out)


def _entity_identifier_fields(nlq: str) -> list[str]:
    """Identifier-like fields to keep entity listing grounded."""
    nl = (nlq or "").lower()
    out: list[str] = []
    for entity, fields in _ENTITY_IDENTIFIER_FIELDS.items():
        if re.search(rf"\b{re.escape(entity)}s?\b", nl):
            out.extend(fields)
    return _unique(out)


def _projection_hints(nlq: str) -> list[str]:
    """Soft projection hints used by schema-linking ranking."""
    explicit = _extract_required_columns(nlq)
    entity = _entity_projection_hints(nlq)
    return _unique(explicit + entity)


def _extract_value_hints(nlq: str) -> list[str]:
    """
    Extract likely literal values from NLQ.
    Returns lowercase hints to match lowercase SQL text checks.
    """
    text = (nlq or "")
    out: list[str] = []

    # quoted literals are high-confidence value hints.
    out.extend(m.group(1).strip().lower() for m in re.finditer(r"'([^']+)'", text))
    out.extend(m.group(1).strip().lower() for m in re.finditer(r'"([^\"]+)"', text))

    # dates and years are common filter values.
    out.extend(m.group(0).lower() for m in _DATE_RE.finditer(text))
    out.extend(m.group(0).lower() for m in _YEAR_RE.finditer(text))

    # status literals often appear unquoted in nlq.
    status_tokens = ["shipped", "cancelled", "on hold", "disputed", "in process", "resolved"]
    nl = text.lower()
    for token in status_tokens:
        if token in nl:
            out.append(token)

    return _unique(out)


def _value_linked_columns_from_tables(nlq: str, tables: dict[str, list[str]]) -> list[str]:
    """
    Map NL value cues to columns that actually exist in provided tables.
    """
    nl = (nlq or "").lower()
    available = {c.lower(): c for cols in tables.values() for c in cols}

    out: list[str] = []
    for col, phrases in _VALUE_COLUMN_HINTS.items():
        if col.lower() not in available:
            continue
        if any(p in nl for p in phrases):
            out.append(available[col.lower()])

    # if a value + location wording appears, prefer location columns when present.
    value_hints = _extract_value_hints(nlq)
    if value_hints and re.search(r"\b(in|from|located|based)\b", nl):
        for location_col in ["city", "country", "state", "postalCode"]:
            if location_col.lower() in available:
                out.append(available[location_col.lower()])

    return _unique(out)
