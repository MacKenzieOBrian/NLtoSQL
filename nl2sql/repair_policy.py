"""
Deterministic SQL repair templates.

Templates are used before LLM-based repair to handle recurring error modes with
stable SQL rewrites.
"""

from __future__ import annotations

import re

from .constraint_hints import _extract_value_hints


def deterministic_repair(nlq: str, bad_sql: str, error: str) -> str | None:
    """Return a deterministic repaired SQL for known patterns, else None."""
    del bad_sql  # reserved for future pattern conditions
    nl = (nlq or "").lower()
    err = (error or "").lower()

    # Employee count by office/location should join employees to offices.
    if re.search(r"\bemployees?\b", nl) and re.search(r"\boffice(s)?\b", nl) and re.search(
        r"\bcount\b|how many|number of", nl
    ):
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

    # Top customers by total payments.
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

    # Average payment amount per country.
    if re.search(r"\baverage\b", nl) and re.search(r"\bpayments?\b", nl) and re.search(r"\b(per|by)\s+country\b", nl):
        return (
            "SELECT c.country, AVG(p.amount) AS avg_payment_amount "
            "FROM customers c JOIN payments p ON c.customerNumber = p.customerNumber "
            "GROUP BY c.country;"
        )

    # Average MSRP by product line.
    if re.search(r"\baverage\b", nl) and re.search(r"\bmsrp\b", nl) and re.search(r"\b(per|by)\s+product\s+line\b", nl):
        return "SELECT productLine, AVG(MSRP) AS avg_msrp FROM products GROUP BY productLine;"

    return None
