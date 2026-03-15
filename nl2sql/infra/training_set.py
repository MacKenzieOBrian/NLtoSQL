"""
Training-set validation helpers for notebook 04.

This is notebook orchestration code rather than core NL-to-SQL logic.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine
from .notebook_utils import load_test_set, load_train_records

def filter_training_records(
    *,
    train_records: list[dict[str, Any]],
    test_nlqs: set[str],
) -> dict[str, Any]:
    """Remove obvious leakage and low-quality rows before QLoRA training."""
    # ai note copilot: scaffold block only, i edited final logic
    overlap: list[tuple[int, str]] = []
    non_select: list[tuple[int, str, str]] = []
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()

    for idx, row in enumerate(train_records):
        nlq = str(row["nlq"]).strip()
        sql = str(row["sql"]).strip()

        # drop exact nlq overlap with test set
        if nlq in test_nlqs:
            overlap.append((idx, nlq))
            continue

        # keep only select sql to match eval task
        if not sql.lower().lstrip().startswith("select"):
            non_select.append((idx, nlq, sql[:120]))
            continue

        # dedupe nlq so repeats dont stack
        if nlq in seen:
            continue
        seen.add(nlq)
        deduped.append({"nlq": nlq, "sql": sql if sql.endswith(";") else sql + ";"})

    return {
        "overlap": overlap,
        "non_select": non_select,
        "deduped": deduped,
    }


def validate_training_queries(
    *,
    engine: Engine,
    records: list[dict[str, str]],
) -> list[tuple[int, str, str, str]]:
    """Check whether the filtered training SQL still executes against the live DB."""
    # execute query like sqlalchemy docs
    # https://docs.sqlalchemy.org/en/20/core/connections.html#sqlalchemy.engine.Connection.execute
    failed: list[tuple[int, str, str, str]] = []
    with engine.connect() as conn:
        for idx, row in enumerate(records):
            sql = row["sql"]
            try:
                res = conn.execute(text(sql))
                res.fetchmany(1)
            except Exception as exc:
                failed.append((idx, row["nlq"], sql, repr(exc)))
    return failed


def run_training_set_validation(
    *,
    test_path: str | Path,
    train_path: str | Path,
    engine: Engine,
) -> dict[str, Any]:
    """Load train/test data, apply filtering, and run executability checks."""
    # ai note copilot: scaffold block only, i edited final logic
    test_items = load_test_set(test_path)
    train_records = load_train_records(train_path)
    test_nlqs = {str(item["nlq"]).strip() for item in test_items}

    filtered = filter_training_records(train_records=train_records, test_nlqs=test_nlqs)
    deduped = filtered["deduped"]
    failed = validate_training_queries(engine=engine, records=deduped)

    # ai note copilot: scaffold block only, i edited final logic
    return {
        "test_items": test_items,
        "train_records": train_records,
        "test_nlqs": test_nlqs,
        "overlap": filtered["overlap"],
        "non_select": filtered["non_select"],
        "deduped": deduped,
        "failed": failed,
    }


def print_training_set_validation_summary(result: dict[str, Any]) -> None:
    """Print the same high-level checks the notebook used inline before extraction."""
    test_items = result["test_items"]
    train_records = result["train_records"]
    overlap = result["overlap"]
    non_select = result["non_select"]
    deduped = result["deduped"]
    failed = result["failed"]

    print("Test items:", len(test_items))
    print("Train items:", len(train_records))
    print("Exact NLQ overlaps with test:", len(overlap))
    print("Non-SELECT rows:", len(non_select))
    print("After NLQ-dedup:", len(deduped))
    print("Executable (VA=True):", len(deduped) - len(failed), "/", len(deduped))
    print("Failed:", len(failed))

    if overlap[:5]:
        print("First overlaps:")
        for idx, nlq in overlap[:5]:
            print("  -", idx, nlq)

    if non_select[:5]:
        print("First non-SELECT rows:")
        for idx, nlq, sql_snip in non_select[:5]:
            print("  -", idx, nlq, "->", sql_snip)

    if failed[:5]:
        print("First failures:")
        for idx, nlq, sql, err in failed[:5]:
            print("---")
            print("row:", idx)
            print("nlq:", nlq)
            print("sql:", sql)
            print("err:", err)


def build_training_validation_report(result: dict[str, Any]) -> dict[str, Any]:
    """Convert the notebook validation result into one small JSON report."""
    test_items = result["test_items"]
    train_records = result["train_records"]
    overlap = result["overlap"]
    non_select = result["non_select"]
    deduped = result["deduped"]
    failed = result["failed"]

    return {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "test_items": len(test_items),
        "train_items_raw": len(train_records),
        "train_items_deduped": len(deduped),
        "overlap_count": len(overlap),
        "non_select_count": len(non_select),
        "va_pass_count": len(deduped) - len(failed),
        "va_fail_count": len(failed),
        "sample_failures": [
            {"row": idx, "nlq": nlq, "sql": sql, "error": err}
            for idx, nlq, sql, err in failed[:5]
        ],
    }


def save_training_validation_report(
    report: dict[str, Any],
    *,
    out_path: str | Path = "results/training_set_validation/validation_report.json",
) -> Path:
    """Persist the training-set validation summary to disk."""
    # save json file with pathlib
    # https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
    # https://docs.python.org/3/library/pathlib.html#pathlib.Path.write_text
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


__all__ = [
    "build_training_validation_report",
    "filter_training_records",
    "print_training_set_validation_summary",
    "run_training_set_validation",
    "save_training_validation_report",
    "validate_training_queries",
]
