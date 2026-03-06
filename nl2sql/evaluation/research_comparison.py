"""Top-level generator for the research comparison outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .research_runs import (
    DEFAULT_PRIMARY_EVAL_PROFILE,
    RunSpec,
    SUPPORTED_PRIMARY_EVAL_PROFILES,
    build_tables_from_runs,
    discover_all_runs,
    discover_primary_runs,
    discover_react_runs,
)
from .research_stats import compute_mean_median_std, compute_paired_tests, planned_comparisons, prepare_per_item_table

DEFAULT_RUNS_ROOT = Path("results")
DEFAULT_PER_ITEM_CSV = Path("results/analysis/per_item_metrics_primary_raw.csv")
DEFAULT_OUT_DIR = Path("results/analysis")
MANIFEST_FILENAME = "run_manifest.csv"
MEANS_FILENAME = "stats_mean_median_std.csv"
PAIRED_FILENAME = "stats_paired_ttests.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Folder containing run JSON files.",
    )
    parser.add_argument(
        "--per-item-csv",
        type=Path,
        default=DEFAULT_PER_ITEM_CSV,
        help="Output path for per-item rows.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output folder for manifest and stats CSVs.",
    )
    parser.add_argument(
        "--primary-eval-profile",
        choices=sorted(SUPPORTED_PRIMARY_EVAL_PROFILES),
        default=DEFAULT_PRIMARY_EVAL_PROFILE,
        help="Primary baseline/QLoRA eval profile to include in the analysis.",
    )
    return parser.parse_args()


def _resolve_paths(
    *,
    runs_root: Path,
    per_item_csv: Path,
    out_dir: Path,
    project_root: Path,
) -> tuple[Path, Path, Path]:
    resolved_runs_root = runs_root if runs_root.is_absolute() else (project_root / runs_root)
    resolved_per_item_csv = per_item_csv if per_item_csv.is_absolute() else (project_root / per_item_csv)
    resolved_out_dir = out_dir if out_dir.is_absolute() else (project_root / out_dir)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)
    resolved_per_item_csv.parent.mkdir(parents=True, exist_ok=True)
    return resolved_runs_root, resolved_per_item_csv, resolved_out_dir


def _manifest_path(out_dir: Path) -> Path:
    return out_dir / MANIFEST_FILENAME


def _means_path(out_dir: Path) -> Path:
    return out_dir / MEANS_FILENAME


def _paired_path(out_dir: Path) -> Path:
    return out_dir / PAIRED_FILENAME


def collect_analysis_inputs(
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    per_item_csv: Path = DEFAULT_PER_ITEM_CSV,
    out_dir: Path = DEFAULT_OUT_DIR,
    primary_eval_profile: str = DEFAULT_PRIMARY_EVAL_PROFILE,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Build the raw tables used later by the stats step."""
    project_root = project_root or Path.cwd()
    runs_root, per_item_csv, out_dir = _resolve_paths(
        runs_root=runs_root,
        per_item_csv=per_item_csv,
        out_dir=out_dir,
        project_root=project_root,
    )

    all_specs, drops = discover_all_runs(
        project_root=project_root,
        runs_root=runs_root,
        primary_eval_profile=primary_eval_profile,
    )
    if not all_specs:
        raise FileNotFoundError(
            f"No valid run files found under: {runs_root}\n"
            "Expected files like: results_k0_seed7.json, results_k3_seed7.json"
        )

    items_df, manifest_df = build_tables_from_runs(all_specs)
    per_item = prepare_per_item_table(items_df)
    manifest_out = _manifest_path(out_dir)

    manifest_df.sort_values(["model_tag", "method", "k", "seed", "run_id"]).to_csv(manifest_out, index=False)
    per_item.sort_values(["condition_id", "seed", "example_id"]).to_csv(per_item_csv, index=False)

    react_specs = [spec for spec in all_specs if spec.method_tag == "react"]
    primary_count = len(list(runs_root.rglob("results_k*_seed*.json")))
    return {
        "stage": "collect_analysis_inputs",
        "runs_root": str(runs_root),
        "primary_eval_profile": primary_eval_profile,
        "run_files_found": int(primary_count),
        "react_files_found": int(len(react_specs)),
        "runs_included": int(len(all_specs)),
        "conditions_included": sorted(set(manifest_df["condition_id"].dropna().astype(str).tolist())),
        "dropped_files": drops,
        "per_item_csv": str(per_item_csv),
        "manifest_csv": str(manifest_out),
    }


def format_analysis_outputs(
    *,
    per_item_csv: Path = DEFAULT_PER_ITEM_CSV,
    out_dir: Path = DEFAULT_OUT_DIR,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Read the raw per-item table and write the formatted stats tables."""
    project_root = project_root or Path.cwd()
    _, per_item_csv, out_dir = _resolve_paths(
        runs_root=DEFAULT_RUNS_ROOT,
        per_item_csv=per_item_csv,
        out_dir=out_dir,
        project_root=project_root,
    )

    per_item = prepare_per_item_table(pd.read_csv(per_item_csv))
    condition_ids = set(per_item["condition_id"].dropna().astype(str).tolist())
    comparisons = planned_comparisons(condition_ids)
    means_out = _means_path(out_dir)
    paired_out = _paired_path(out_dir)

    compute_mean_median_std(per_item).sort_values(["condition_id", "seed", "metric"]).to_csv(means_out, index=False)
    compute_paired_tests(per_item, comparisons).sort_values(["comparison", "metric"]).to_csv(paired_out, index=False)

    return {
        "stage": "format_analysis_outputs",
        "per_item_csv": str(per_item_csv),
        "conditions_included": sorted(condition_ids),
        "comparisons_planned": int(len(comparisons)),
        "mean_stats_csv": str(means_out),
        "paired_stats_csv": str(paired_out),
        "outputs": [str(means_out), str(paired_out)],
    }


def _write_outputs(
    *,
    collect_summary: dict[str, Any],
    format_summary: dict[str, Any],
    per_item_csv: Path,
) -> list[str]:
    return [
        collect_summary["manifest_csv"],
        str(per_item_csv),
        format_summary["mean_stats_csv"],
        format_summary["paired_stats_csv"],
    ]


def generate(
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    per_item_csv: Path = DEFAULT_PER_ITEM_CSV,
    out_dir: Path = DEFAULT_OUT_DIR,
    primary_eval_profile: str = DEFAULT_PRIMARY_EVAL_PROFILE,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for the two-stage analysis flow."""
    project_root = project_root or Path.cwd()
    _, per_item_csv, out_dir = _resolve_paths(
        runs_root=runs_root,
        per_item_csv=per_item_csv,
        out_dir=out_dir,
        project_root=project_root,
    )

    collect_summary = collect_analysis_inputs(
        runs_root=runs_root,
        per_item_csv=per_item_csv,
        out_dir=out_dir,
        primary_eval_profile=primary_eval_profile,
        project_root=project_root,
    )
    format_summary = format_analysis_outputs(
        per_item_csv=per_item_csv,
        out_dir=out_dir,
        project_root=project_root,
    )
    outputs = _write_outputs(
        collect_summary=collect_summary,
        format_summary=format_summary,
        per_item_csv=per_item_csv,
    )
    return {
        "design": "two_stage_analysis_from_results_tree",
        "runs_root": collect_summary["runs_root"],
        "primary_eval_profile": primary_eval_profile,
        "run_files_found": collect_summary["run_files_found"],
        "react_files_found": collect_summary["react_files_found"],
        "runs_included": collect_summary["runs_included"],
        "conditions_included": collect_summary["conditions_included"],
        "comparisons_planned": format_summary["comparisons_planned"],
        "dropped_files": collect_summary["dropped_files"],
        "per_item_csv": collect_summary["per_item_csv"],
        "outputs": outputs,
        "collect": collect_summary,
        "format": format_summary,
    }


def main() -> None:
    args = parse_args()
    summary = generate(
        runs_root=args.runs_root,
        per_item_csv=args.per_item_csv,
        out_dir=args.out_dir,
        primary_eval_profile=args.primary_eval_profile,
    )
    print(json.dumps(summary, indent=2))


__all__ = [
    "RunSpec",
    "build_tables_from_runs",
    "collect_analysis_inputs",
    "compute_mean_median_std",
    "compute_paired_tests",
    "discover_all_runs",
    "discover_primary_runs",
    "discover_react_runs",
    "format_analysis_outputs",
    "generate",
    "main",
    "parse_args",
    "planned_comparisons",
    "prepare_per_item_table",
]
