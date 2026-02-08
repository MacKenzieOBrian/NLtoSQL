#!/usr/bin/env python3
"""
Preflight checks for the tool-driven ReAct loop (no model/DB required).

This validates that:
- required files exist
- prompt lists required tools and rules
- agent_tools defines required tool functions
- notebook contains validate_sql and run_sql gating
- technical reference exists (kept intentionally short) and mentions key loop guarantees
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "nl2sql/agent_tools.py",
    "nl2sql/prompts.py",
    "notebooks/03_agentic_eval.ipynb",
    "LOGBOOK.md",
    "TOOL_DRIVEN_REACT_LOOP_TECHNICAL_REFERENCE.md",
]

REQUIRED_TOOL_FUNCS = [
    "get_schema",
    "link_schema",
    "extract_constraints",
    "generate_sql",
    "validate_sql",
    "validate_constraints",
    "run_sql",
    "repair_sql",
    "finish",
]

REQUIRED_PROMPT_RULES = [
    "Only use tables and columns from get_schema",
    "After get_schema, call link_schema before generate_sql",
    "After link_schema, call extract_constraints before generate_sql",
    "After generate_sql or repair_sql, call validate_sql",
    "If validate_sql passes, call validate_constraints",
    "If validate_constraints fails, call repair_sql",
    "If validate_constraints passes, call run_sql",
    "If validate_sql fails, call repair_sql",
    "Always call run_sql before finish",
]


def _check_files() -> list[str]:
    errors = []
    for rel in REQUIRED_FILES:
        if not (ROOT / rel).exists():
            errors.append(f"Missing required file: {rel}")
    return errors


def _check_agent_tools() -> list[str]:
    errors = []
    text = (ROOT / "nl2sql/agent_tools.py").read_text(encoding="utf-8")
    for fn in REQUIRED_TOOL_FUNCS:
        if f"def {fn}(" not in text:
            errors.append(f"agent_tools missing function: {fn}")
    return errors


def _check_prompt_rules() -> list[str]:
    errors = []
    text = (ROOT / "nl2sql/prompts.py").read_text(encoding="utf-8")
    for rule in REQUIRED_PROMPT_RULES:
        if rule not in text:
            errors.append(f"Prompt missing rule: {rule}")
    for tool in REQUIRED_TOOL_FUNCS:
        if tool not in text:
            errors.append(f"Prompt missing tool mention: {tool}")
    return errors


def _check_notebook_loop() -> list[str]:
    errors = []
    nb_path = ROOT / "notebooks/03_agentic_eval.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code = "\n".join("".join(c.get("source", "")) for c in nb["cells"] if c.get("cell_type") == "code")
    if "def react_sql" not in code:
        errors.append("Notebook missing react_sql definition.")
    if '"validate_sql": validate_sql' not in code:
        errors.append("Notebook TOOLS missing validate_sql.")
    if '"extract_constraints": extract_constraints' not in code:
        errors.append("Notebook TOOLS missing extract_constraints.")
    if '"validate_constraints": validate_constraints' not in code:
        errors.append("Notebook TOOLS missing validate_constraints.")
    if '"link_schema": link_schema' not in code:
        errors.append("Notebook TOOLS missing link_schema.")
    if "Must call validate_sql before run_sql" not in code:
        errors.append("Notebook missing validate_sql -> run_sql gating.")
    if "Must call validate_constraints before run_sql" not in code:
        errors.append("Notebook missing validate_constraints -> run_sql gating.")
    if "_apply_guardrails" not in code:
        errors.append("Notebook missing guardrails application.")
    return errors


def _check_docs() -> list[str]:
    errors = []
    ref = (ROOT / "TOOL_DRIVEN_REACT_LOOP_TECHNICAL_REFERENCE.md").read_text(encoding="utf-8")
    required_mentions = ["react_sql", "validate_sql", "validate_constraints", "run_sql", "repair_sql", "trace", "decision_log"]
    for m in required_mentions:
        if m not in ref:
            errors.append(f"Technical reference missing mention: {m}")
    return errors


def main() -> int:
    errors = []
    errors.extend(_check_files())
    if errors:
        print("Preflight failed early:")
        for err in errors:
            print(" -", err)
        return 1

    checks = [
        ("agent_tools", _check_agent_tools()),
        ("prompt_rules", _check_prompt_rules()),
        ("notebook_loop", _check_notebook_loop()),
        ("docs", _check_docs()),
    ]
    failed = False
    for name, errs in checks:
        if errs:
            failed = True
            print(f"[FAIL] {name}")
            for err in errs:
                print(" -", err)
        else:
            print(f"[OK]   {name}")

    if failed:
        print("\nPreflight: FAIL")
        return 2
    print("\nPreflight: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
