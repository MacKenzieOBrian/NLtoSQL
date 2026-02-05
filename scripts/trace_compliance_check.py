#!/usr/bin/env python3
"""
Model-free compliance checks for the ReAct loop trace summaries.

This script synthesizes a few traces and verifies summarize_trace()
flags compliance correctly. It does not require model or DB access.
"""

from __future__ import annotations

import json


def summarize_trace(trace: list[dict]) -> dict:
    actions = [t.get("action") for t in trace if t.get("action")]
    forced_repairs = [t for t in trace if t.get("forced_action") == "repair_sql"]
    repair_count = sum(1 for t in trace if t.get("action") == "repair_sql")

    errors: list[str] = []
    for i, a in enumerate(actions):
        if a == "generate_sql" and "extract_constraints" not in actions[:i]:
            errors.append("generate_without_constraints")
        if a == "run_sql" and "validate_sql" not in actions[:i]:
            errors.append("run_without_validate")
        if a == "run_sql" and "validate_constraints" not in actions[:i]:
            errors.append("run_without_validate_constraints")
        if a == "finish" and "run_sql" not in actions[:i]:
            errors.append("finish_without_run")
    compliance_ok = len(errors) == 0
    return {
        "actions": actions,
        "repairs": repair_count,
        "forced_repairs": len(forced_repairs),
        "compliance_ok": compliance_ok,
        "compliance_errors": errors,
    }


SYNTHETIC_TRACES = [
    {
        "name": "happy_path",
        "trace": [
            {"action": "get_schema"},
            {"action": "link_schema"},
            {"action": "extract_constraints"},
            {"action": "generate_sql"},
            {"action": "validate_sql"},
            {"action": "validate_constraints"},
            {"action": "run_sql"},
            {"action": "finish"},
        ],
        "expect_ok": True,
    },
    {
        "name": "missing_validate",
        "trace": [
            {"action": "get_schema"},
            {"action": "link_schema"},
            {"action": "extract_constraints"},
            {"action": "generate_sql"},
            {"action": "run_sql"},
            {"action": "finish"},
        ],
        "expect_ok": False,
    },
    {
        "name": "missing_run",
        "trace": [
            {"action": "get_schema"},
            {"action": "link_schema"},
            {"action": "extract_constraints"},
            {"action": "generate_sql"},
            {"action": "validate_sql"},
            {"action": "finish"},
        ],
        "expect_ok": False,
    },
    {
        "name": "forced_repair_logged",
        "trace": [
            {"action": "extract_constraints"},
            {"action": "generate_sql"},
            {"action": "validate_sql"},
            {"action": "validate_constraints"},
            {"forced_action": "repair_sql"},
            {"action": "repair_sql"},
            {"action": "validate_sql"},
            {"action": "validate_constraints"},
            {"action": "run_sql"},
            {"action": "finish"},
        ],
        "expect_ok": True,
    },
]


def main() -> int:
    failed = False
    for case in SYNTHETIC_TRACES:
        summary = summarize_trace(case["trace"])
        ok = summary["compliance_ok"]
        expect = case["expect_ok"]
        status = "OK" if ok == expect else "FAIL"
        if status == "FAIL":
            failed = True
        print(f"[{status}] {case['name']}")
        print("  summary:", json.dumps(summary, ensure_ascii=False))
    print("\nCompliance check:", "PASS" if not failed else "FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
