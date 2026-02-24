#!/usr/bin/env python3
"""
Minimal stats generator for dissertation hypothesis testing.

Supervisor-required outputs only:
1) Mean + median per run/metric.
2) Shapiro-Wilk normality per run/metric.
3) Paired t-tests on predefined run comparisons (H0: mean difference = 0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from scipy.stats import shapiro, ttest_rel

SHAPIRO_REF = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html"
TTEST_REF = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html"
METRICS = ("va", "em", "ex", "ts")

# right - left comparisons for paired t-tests
PAIRED_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("baseline_k0", "baseline_k3", "Baseline k0->k3"),
    ("qlora_k0", "qlora_k3", "QLoRA k0->k3"),
    ("baseline_k0", "qlora_k0", "Base->QLoRA @k0"),
    ("baseline_k3", "qlora_k3", "Base->QLoRA @k3"),
    ("llama_base_k0", "llama_base_k3", "Llama Base k0->k3"),
    ("qwen_base_k0", "qwen_base_k3", "Qwen Base k0->k3"),
    ("llama_qlora_k0", "llama_qlora_k3", "Llama QLoRA k0->k3"),
    ("qwen_qlora_k0", "qwen_qlora_k3", "Qwen QLoRA k0->k3"),
    ("llama_base_k0", "llama_qlora_k0", "Llama Base->QLoRA @k0"),
    ("llama_base_k3", "llama_qlora_k3", "Llama Base->QLoRA @k3"),
    ("qwen_base_k0", "qwen_qlora_k0", "Qwen Base->QLoRA @k0"),
    ("qwen_base_k3", "qwen_qlora_k3", "Qwen Base->QLoRA @k3"),
)


def _discover_run_jsons(project_root: Path) -> list[Path]:
    candidates: list[Path] = []
    patterns = (
        "results/baseline/results_zero_shot_200.json",
        "results/baseline/results_few_shot_k3_200.json",
        "results/qlora/results_zero_shot_200.json",
        "results/qlora/results_few_shot_k3_200.json",
        "results/baseline/model_family/*.json",
        "results/qlora/model_family/*.json",
        "results/baseline/runs/*/results_k*_seed*.json",
        "results/qlora/runs/*/results_k*_seed*.json",
    )
    for pat in patterns:
        candidates.extend(sorted(project_root.glob(pat)))

    # de-dupe while preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen and p.exists():
            seen.add(rp)
            out.append(p)
    return out


def _model_tag(model_alias: str) -> str:
    alias = (model_alias or "").lower()
    if "llama" in alias:
        return "llama"
    if "qwen" in alias:
        return "qwen"
    return alias or "model"


def _infer_method(path: Path, run_meta: dict[str, Any]) -> str:
    m = str(run_meta.get("method", "")).strip().lower()
    if m in {"baseline", "qlora"}:
        return m
    p = str(path).lower()
    if "/qlora/" in p:
        return "qlora"
    return "baseline"


def _infer_k(payload: dict[str, Any], run_meta: dict[str, Any], path: Path) -> int:
    k = payload.get("k", run_meta.get("k"))
    if isinstance(k, int):
        return k
    name = path.name.lower()
    if "k0" in name:
        return 0
    if "k3" in name:
        return 3
    if "k5" in name:
        return 5
    return -1


def _run_id_and_label(
    *,
    model_alias: str,
    method: str,
    k: int,
) -> tuple[str, str]:
    tag = _model_tag(model_alias)
    method_tag = "qlora" if method == "qlora" else "base"
    run_id = f"{tag}_{method_tag}_k{k}"
    method_label = "QLoRA" if method == "qlora" else "Base"
    run_label = f"{tag.capitalize()} {method_label} k={k}"
    return run_id, run_label


def _build_per_item_from_run_jsons(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        results = payload.get("results")
        if not isinstance(results, list):
            continue

        run_meta = payload.get("run_metadata") or {}
        method = _infer_method(path, run_meta)
        model_alias = str(run_meta.get("model_alias", ""))
        k = _infer_k(payload, run_meta, path)
        run_id, run_label = _run_id_and_label(model_alias=model_alias, method=method, k=k)

        for item in results:
            rows.append(
                {
                    "run_id": run_id,
                    "run_label": run_label,
                    "example_id": item.get("i"),
                    "nlq": item.get("nlq", ""),
                    "va": item.get("va"),
                    "em": item.get("em"),
                    "ex": item.get("ex"),
                    "ts": item.get("ts"),
                    "source_json": str(path),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["run_id", "run_label", "example_id", "nlq", "va", "em", "ex", "ts", "source_json"])
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-item-csv",
        type=Path,
        default=Path("results/analysis/per_item_metrics.csv"),
        help="Path to per_item_metrics.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/analysis"),
        help="Folder for output stats CSV files",
    )
    return parser.parse_args()


def _prepare_per_item(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path)
    required = {"run_id", "run_label", "nlq", "va", "em", "ex", "ts"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")
    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df


def compute_mean_median_shapiro(per_item: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_id, g in per_item.groupby("run_id", sort=False):
        run_label = str(g["run_label"].iloc[0])
        for metric in METRICS:
            s = g[metric].dropna()
            n = int(len(s))
            mean = float(s.mean()) if n else None
            median = float(s.median()) if n else None
            std = float(s.std(ddof=1)) if n > 1 else None

            if n < 3:
                w = None
                p = None
                decision = "insufficient_n"
                note = "need at least 3 samples for Shapiro-Wilk"
            else:
                w_val, p_val = shapiro(s.to_numpy())
                w = float(w_val)
                p = float(p_val)
                decision = "reject_normality" if p < 0.05 else "fail_to_reject_normality"
                note = ""

            rows.append(
                {
                    "run_id": run_id,
                    "run_label": run_label,
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "median": median,
                    "std": std,
                    "shapiro_w": w,
                    "shapiro_p": p,
                    "decision_alpha_0_05": decision,
                    "null_hypothesis": "data_is_normally_distributed",
                    "reference_url": SHAPIRO_REF,
                    "note": note,
                }
            )
    return pd.DataFrame(rows).sort_values(["run_id", "metric"]).reset_index(drop=True)


def _join_for_pair(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if "example_id" in left.columns and "example_id" in right.columns:
        left_ids = left["example_id"].notna().any()
        right_ids = right["example_id"].notna().any()
        if left_ids and right_ids:
            merged = left[["example_id", *METRICS]].merge(
                right[["example_id", *METRICS]],
                on="example_id",
                how="inner",
                suffixes=("_left", "_right"),
            )
            return merged, "example_id"

    merged = left[["nlq", *METRICS]].merge(
        right[["nlq", *METRICS]],
        on="nlq",
        how="inner",
        suffixes=("_left", "_right"),
    )
    return merged, "nlq"


def compute_paired_ttests(per_item: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_run = {rid: g.copy() for rid, g in per_item.groupby("run_id")}

    for left_id, right_id, label in PAIRED_COMPARISONS:
        if left_id not in by_run or right_id not in by_run:
            continue

        merged, pair_key = _join_for_pair(by_run[left_id], by_run[right_id])

        for metric in METRICS:
            col_l = f"{metric}_left"
            col_r = f"{metric}_right"
            valid = merged[[col_l, col_r]].dropna()
            n = int(len(valid))

            if n < 2:
                left_mean = None
                right_mean = None
                diff = None
                t_stat = None
                p_value = None
                decision = "insufficient_n"
            else:
                left_mean = float(valid[col_l].mean())
                right_mean = float(valid[col_r].mean())
                diff = right_mean - left_mean
                test = ttest_rel(valid[col_r], valid[col_l], nan_policy="omit")
                t_stat = float(test.statistic)
                p_value = float(test.pvalue)
                decision = "reject_H0" if p_value < 0.05 else "fail_to_reject_H0"

            rows.append(
                {
                    "comparison": label,
                    "left_run_id": left_id,
                    "right_run_id": right_id,
                    "metric": metric,
                    "pair_key": pair_key,
                    "n_pairs": n,
                    "left_mean": left_mean,
                    "right_mean": right_mean,
                    "mean_diff_right_minus_left": diff,
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "decision_alpha_0_05": decision,
                    "null_hypothesis": "mean_difference_equals_zero",
                    "reference_url": TTEST_REF,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "comparison",
                "left_run_id",
                "right_run_id",
                "metric",
                "pair_key",
                "n_pairs",
                "left_mean",
                "right_mean",
                "mean_diff_right_minus_left",
                "t_stat",
                "p_value",
                "decision_alpha_0_05",
                "null_hypothesis",
                "reference_url",
            ]
        )
    return pd.DataFrame(rows).sort_values(["comparison", "metric"]).reset_index(drop=True)


def generate(
    *,
    per_item_csv: Path = Path("results/analysis/per_item_metrics.csv"),
    out_dir: Path = Path("results/analysis"),
    project_root: Path | None = None,
) -> dict[str, Any]:
    """
    Programmatic entrypoint used by notebooks.

    Returns a compact summary with input/output paths and row counts.
    """
    if project_root is not None:
        per_item_csv = per_item_csv if per_item_csv.is_absolute() else (project_root / per_item_csv)
        out_dir = out_dir if out_dir.is_absolute() else (project_root / out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    if not per_item_csv.exists():
        if project_root is None:
            project_root = Path.cwd()
        run_jsons = _discover_run_jsons(project_root)
        built = _build_per_item_from_run_jsons(run_jsons)
        if built.empty:
            raise FileNotFoundError(
                "Missing per-item input and no run JSON files were discovered. "
                "Expected either results/analysis/per_item_metrics.csv or run JSON artifacts under "
                "results/baseline and results/qlora."
            )
        per_item_csv.parent.mkdir(parents=True, exist_ok=True)
        built.to_csv(per_item_csv, index=False)

    per_item = _prepare_per_item(per_item_csv)

    stats_shapiro = compute_mean_median_shapiro(per_item)
    stats_ttests = compute_paired_ttests(per_item)

    shapiro_path = out_dir / "stats_mean_median_shapiro.csv"
    ttests_path = out_dir / "stats_paired_ttests.csv"

    stats_shapiro.to_csv(shapiro_path, index=False)
    stats_ttests.to_csv(ttests_path, index=False)

    return {
        "input": str(per_item_csv),
        "rows": int(len(per_item)),
        "outputs": [str(shapiro_path), str(ttests_path)],
    }


def main() -> None:
    args = parse_args()
    summary = generate(per_item_csv=args.per_item_csv, out_dir=args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
