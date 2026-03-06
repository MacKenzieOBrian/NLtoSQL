#!/usr/bin/env python3
"""Build raw analysis tables from saved run JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nl2sql.evaluation.research_comparison import (
    DEFAULT_OUT_DIR,
    DEFAULT_PER_ITEM_CSV,
    DEFAULT_PRIMARY_EVAL_PROFILE,
    DEFAULT_RUNS_ROOT,
    SUPPORTED_PRIMARY_EVAL_PROFILES,
    collect_analysis_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--per-item-csv", type=Path, default=DEFAULT_PER_ITEM_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--primary-eval-profile",
        choices=sorted(SUPPORTED_PRIMARY_EVAL_PROFILES),
        default=DEFAULT_PRIMARY_EVAL_PROFILE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = collect_analysis_inputs(
        runs_root=args.runs_root,
        per_item_csv=args.per_item_csv,
        out_dir=args.out_dir,
        primary_eval_profile=args.primary_eval_profile,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
