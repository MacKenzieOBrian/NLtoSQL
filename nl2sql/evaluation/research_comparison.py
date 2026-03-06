"""Top-level generator for the research comparison outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("results"),
        help="Folder containing run JSON files.",
    )
    parser.add_argument(
        "--per-item-csv",
        type=Path,
        default=Path("results/analysis/per_item_metrics_primary_raw.csv"),
        help="Output path for per-item rows.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/analysis"),
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


def _write_outputs(
    *,
    manifest_df: Any,
    per_item: Any,
    comparisons: list[tuple[str, str, str]],
    per_item_csv: Path,
    out_dir: Path,
) -> list[str]:
    manifest_out = out_dir / "run_manifest.csv"
    means_out = out_dir / "stats_mean_median_std.csv"
    paired_out = out_dir / "stats_paired_ttests.csv"

    manifest_df.sort_values(["model_tag", "method", "k", "seed", "run_id"]).to_csv(manifest_out, index=False)
    per_item.sort_values(["condition_id", "seed", "example_id"]).to_csv(per_item_csv, index=False)
    compute_mean_median_std(per_item).sort_values(["condition_id", "seed", "metric"]).to_csv(means_out, index=False)
    compute_paired_tests(per_item, comparisons).sort_values(["comparison", "metric"]).to_csv(paired_out, index=False)
    return [str(manifest_out), str(means_out), str(paired_out)]


def generate(
    *,
    runs_root: Path = Path("results"),
    per_item_csv: Path = Path("results/analysis/per_item_metrics_primary_raw.csv"),
    out_dir: Path = Path("results/analysis"),
    primary_eval_profile: str = DEFAULT_PRIMARY_EVAL_PROFILE,
    project_root: Path | None = None,
) -> dict[str, Any]:
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
    condition_ids = set(manifest_df["condition_id"].dropna().astype(str).tolist())
    comparisons = planned_comparisons(condition_ids)
    outputs = _write_outputs(
        manifest_df=manifest_df,
        per_item=per_item,
        comparisons=comparisons,
        per_item_csv=per_item_csv,
        out_dir=out_dir,
    )

    react_specs = [spec for spec in all_specs if spec.method_tag == "react"]
    primary_count = len(list(runs_root.rglob("results_k*_seed*.json")))
    return {
        "design": "multi_model_multi_method_from_results_tree",
        "runs_root": str(runs_root),
        "primary_eval_profile": primary_eval_profile,
        "run_files_found": int(primary_count),
        "react_files_found": int(len(react_specs)),
        "runs_included": int(len(all_specs)),
        "conditions_included": sorted(condition_ids),
        "comparisons_planned": int(len(comparisons)),
        "dropped_files": drops,
        "per_item_csv": str(per_item_csv),
        "outputs": outputs,
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
    "compute_mean_median_std",
    "compute_paired_tests",
    "discover_all_runs",
    "discover_primary_runs",
    "discover_react_runs",
    "generate",
    "main",
    "parse_args",
    "planned_comparisons",
    "prepare_per_item_table",
]
