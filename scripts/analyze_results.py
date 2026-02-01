"""
Quick analysis of NL->SQL evaluation JSON to identify failing queries.

Usage:
    python scripts/analyze_results.py results/agent/results_react_200.json

If no path is provided, defaults to results/agent/results_react_200.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def is_agg(sql: str) -> bool:
    sql_low = sql.lower()
    return any(tok in sql_low for tok in ("sum(", "count(", "avg(", "group by"))


def is_join(sql: str) -> bool:
    return " join " in sql.lower()


def select_columns(sql: str) -> list[str]:
    """Parse the SELECT list in a lightweight way (best-effort)."""
    if not sql:
        return []
    m = re.search(r"(?is)^\s*select\s+(.*?)\s+from\s+", sql)
    if not m:
        return []
    sel = m.group(1)
    parts = [p.strip() for p in sel.split(",") if p.strip()]
    # drop aliases to compare core tokens
    cleaned = [re.sub(r"(?is)\s+as\s+.*$", "", p).strip() for p in parts]
    return cleaned


def quoted_literals(sql: str) -> set[str]:
    return set(re.findall(r"'([^']+)'", sql or ""))


def classify_error(gold: str, pred: str) -> list[str]:
    """Heuristic taxonomy for EX failures (simple and deterministic)."""
    labels: list[str] = []
    g = gold.lower() if gold else ""
    p = pred.lower() if pred else ""

    # projection mismatch
    g_cols = set(select_columns(gold))
    p_cols = set(select_columns(pred))
    if g_cols and p_cols and g_cols != p_cols:
        labels.append("projection_mismatch")

    # intent mismatch (aggregate vs non-aggregate)
    if is_agg(gold) != is_agg(pred):
        labels.append("intent_mismatch")

    # join mismatch
    if is_join(gold) != is_join(pred):
        labels.append("join_mismatch")

    # literal mismatch (e.g., USA, San Francisco)
    g_lit = quoted_literals(gold)
    p_lit = quoted_literals(pred)
    if g_lit and p_lit and g_lit != p_lit:
        labels.append("literal_mismatch")

    if not labels:
        labels.append("other")
    return labels


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/agent/results_react_200.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    items = data.get("items", [])
    total = len(items)
    failing_ex = [it for it in items if it.get("ex") == 0]
    failing_va = [it for it in items if it.get("va") == 0]

    print(f"File: {path}")
    print(f"Total items: {total}")
    print(f"EX failures: {len(failing_ex)}")
    print(f"VA failures: {len(failing_va)}")
    print()

    # simple taxonomy
    counts: dict[str, int] = {}
    examples: dict[str, list[dict]] = {}
    for it in failing_ex:
        gold = it.get("gold_sql", "")
        pred = it.get("pred_sql", "")
        labels = classify_error(gold, pred)
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
            examples.setdefault(lab, [])
            if len(examples[lab]) < 3:
                examples[lab].append(it)

    print("Error taxonomy (EX=0):")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"- {k}: {v}")
    print()

    print("Top failing queries (EX=0):")
    for it in failing_ex[:10]:
        nlq = it.get("nlq", "").strip()
        gold = it.get("gold_sql", "")
        pred = it.get("pred_sql", "")
        agg = is_agg(gold) or is_agg(pred)
        join = is_join(gold) or is_join(pred)
        flags = []
        if agg:
            flags.append("agg")
        if join:
            flags.append("join")
        flag_text = ",".join(flags) if flags else "simple"
        print(f"- NLQ: {nlq}")
        print(f"  flags: {flag_text}")
        print(f"  gold: {gold}")
        print(f"  pred: {pred}")
        print()

    print("Examples by error type:")
    for lab, items in examples.items():
        print(f"\n[{lab}]")
        for it in items:
            print(f"- NLQ: {it.get('nlq','').strip()}")
            print(f"  gold: {it.get('gold_sql','')}")
            print(f"  pred: {it.get('pred_sql','')}")


if __name__ == "__main__":
    main()
