"""
Quick analysis of NL->SQL evaluation JSON to identify failing queries,
with a focus on joins/aggregations.

Usage:
    python scripts/analyze_results.py results/agent/results_react_200.json

If no path is provided, defaults to results/agent/results_react_200.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def is_agg(sql: str) -> bool:
    sql_low = sql.lower()
    return any(tok in sql_low for tok in ("sum(", "count(", "avg(", "group by"))


def is_join(sql: str) -> bool:
    return " join " in sql.lower()


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

    print("Top failing queries (EX=0):")
    for it in failing_ex:
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


if __name__ == "__main__":
    main()
