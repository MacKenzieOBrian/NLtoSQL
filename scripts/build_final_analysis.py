#!/usr/bin/env python3
"""Build the official dissertation analysis tables from ``results/final_pack/``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from nl2sql.evaluation.final_pack import build_tables_from_pack
from nl2sql.evaluation.simple_stats import build_pairwise_tests, build_summary_by_condition


PACK_DIR = Path("results/final_pack")
OUT_DIR = Path("results/final_analysis")


def main() -> None:
    """Load the manual final pack and write the official dissertation CSV outputs."""
    # ai note copilot: scaffold block only, i edited final logic
    project_root = Path(__file__).resolve().parents[1]
    pack_dir = project_root / PACK_DIR
    out_dir = project_root / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    per_item_df, manifest_df = build_tables_from_pack(pack_dir)
    summary_by_condition = build_summary_by_condition(per_item_df)
    pairwise_tests = build_pairwise_tests(per_item_df)

    manifest_path = out_dir / "manifest.csv"
    per_item_path = out_dir / "per_item.csv"
    summary_by_condition_path = out_dir / "summary_by_condition.csv"
    pairwise_tests_path = out_dir / "pairwise_tests.csv"

    manifest_df.sort_values(["condition_id", "seed"]).to_csv(manifest_path, index=False)
    per_item_df.sort_values(["condition_id", "seed", "example_id"]).to_csv(per_item_path, index=False)
    summary_by_condition.sort_values(["condition_id"]).to_csv(summary_by_condition_path, index=False)
    if not pairwise_tests.empty:
        pairwise_tests = pairwise_tests.sort_values(["comparison"])
    pairwise_tests.to_csv(pairwise_tests_path, index=False)

    summary = {
        "files_loaded": int(len(manifest_df)),
        "conditions_loaded": sorted(manifest_df["condition_id"].dropna().astype(str).unique().tolist()),
        "outputs_written": [
            str(manifest_path),
            str(per_item_path),
            str(summary_by_condition_path),
            str(pairwise_tests_path),
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
