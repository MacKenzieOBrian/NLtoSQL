#!/usr/bin/env python3
"""Turn the raw per-item table into summary stats and paired-test tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nl2sql.evaluation.research_comparison import DEFAULT_OUT_DIR, DEFAULT_PER_ITEM_CSV, format_analysis_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-item-csv", type=Path, default=DEFAULT_PER_ITEM_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = format_analysis_outputs(
        per_item_csv=args.per_item_csv,
        out_dir=args.out_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
