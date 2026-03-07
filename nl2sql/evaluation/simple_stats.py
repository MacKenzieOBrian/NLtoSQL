"""Minimal summary and EX-only test helpers for the manual final-pack workflow."""

from __future__ import annotations

import warnings

import pandas as pd
from scipy.stats import wilcoxon


_SUMMARY_METRICS = ("va", "ex", "ts")
_PAIRWISE_TEST_COLUMNS = [
    "comparison",
    "left_condition_id",
    "right_condition_id",
    "n_pairs",
    "left_mean",
    "right_mean",
    "mean_diff_right_minus_left",
    "wilcoxon_stat",
    "wilcoxon_p",
    "wilcoxon_p_bh_fdr",
    "decision_alpha_0_05",
    "decision_bh_fdr_alpha_0_05",
]


def _coerce_per_item(per_item_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the flat per-item table into numeric columns used downstream."""
    required = {
        "condition_id",
        "model_tag",
        "method",
        "k",
        "seed",
        "example_id",
        "va",
        "em",
        "ex",
        "ts",
    }
    missing = sorted(required - set(per_item_df.columns))
    if missing:
        raise ValueError(f"Per-item table is missing columns: {missing}")

    out = per_item_df.copy()
    for col in ["k", "seed", "example_id", "va", "em", "ex", "ts"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _wilcoxon_result(diffs: list[float]) -> tuple[float | None, float | None]:
    """Run the paired Wilcoxon test and return statistic plus p-value."""
    if len(diffs) < 2:
        return None, None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        return float(test.statistic), float(test.pvalue)
    except ValueError:
        return 0.0, 1.0


def _bh_fdr_adjust(pvalues: list[float | None]) -> list[float | None]:
    """Apply Benjamini-Hochberg FDR correction to the available p-values."""
    adjusted: list[float | None] = list(pvalues)
    ranked = sorted((float(p), idx) for idx, p in enumerate(pvalues) if p is not None)
    if not ranked:
        return adjusted

    m = len(ranked)
    running_min = 1.0
    ranked_adjusted = [0.0] * m
    for rank in range(m - 1, -1, -1):
        p_value, _ = ranked[rank]
        candidate = min(1.0, p_value * m / (rank + 1))
        running_min = min(running_min, candidate)
        ranked_adjusted[rank] = running_min

    for rank, (_, original_index) in enumerate(ranked):
        adjusted[original_index] = ranked_adjusted[rank]
    return adjusted


def build_summary_by_condition(per_item_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-item table into one descriptive summary row per condition."""
    per_item = _coerce_per_item(per_item_df)
    by_run = (
        per_item.groupby(["condition_id", "model_tag", "method", "k", "seed"], sort=True)[list(_SUMMARY_METRICS)]
        .mean()
        .reset_index()
    )

    rows: list[dict[str, object]] = []
    for keys, group in by_run.groupby(["condition_id", "model_tag", "method", "k"], sort=True):
        condition_id, model_tag, method, k = keys
        row: dict[str, object] = {
            "condition_id": condition_id,
            "model_tag": model_tag,
            "method": method,
            "k": int(k) if pd.notna(k) else None,
            "n_runs": int(len(group)),
        }
        for metric in _SUMMARY_METRICS:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else None
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
        rows.append(row)
    return pd.DataFrame(rows)


def build_pairwise_tests(per_item_df: pd.DataFrame) -> pd.DataFrame:
    """Run the fixed EX-only comparison set on the matched per-item rows."""
    per_item = _coerce_per_item(per_item_df)
    by_condition = {condition_id: group.copy() for condition_id, group in per_item.groupby("condition_id")}

    comparisons = [
        ("llama_base_k0", "llama_base_k3", "Llama Base k0->k3"),
        ("llama_qlora_k0", "llama_qlora_k3", "Llama QLoRA k0->k3"),
        ("qwen_base_k0", "qwen_base_k3", "Qwen Base k0->k3"),
        ("qwen_qlora_k0", "qwen_qlora_k3", "Qwen QLoRA k0->k3"),
        ("llama_base_k0", "llama_qlora_k0", "Llama Base->QLoRA @k0"),
        ("llama_base_k3", "llama_qlora_k3", "Llama Base->QLoRA @k3"),
        ("qwen_base_k0", "qwen_qlora_k0", "Qwen Base->QLoRA @k0"),
        ("qwen_base_k3", "qwen_qlora_k3", "Qwen Base->QLoRA @k3"),
    ]

    rows: list[dict[str, object]] = []
    for left_id, right_id, label in comparisons:
        if left_id not in by_condition or right_id not in by_condition:
            continue

        merged = by_condition[left_id][["seed", "example_id", "ex"]].merge(
            by_condition[right_id][["seed", "example_id", "ex"]],
            on=["seed", "example_id"],
            how="inner",
            suffixes=("_left", "_right"),
        )
        valid = merged[["ex_left", "ex_right"]].dropna().astype(float)
        n_pairs = int(len(valid))
        left_mean = float(valid["ex_left"].mean()) if n_pairs else None
        right_mean = float(valid["ex_right"].mean()) if n_pairs else None
        mean_diff = (right_mean - left_mean) if n_pairs else None
        stat, p_value = _wilcoxon_result((valid["ex_right"] - valid["ex_left"]).tolist() if n_pairs else [])

        rows.append(
            {
                "comparison": label,
                "left_condition_id": left_id,
                "right_condition_id": right_id,
                "n_pairs": n_pairs,
                "left_mean": left_mean,
                "right_mean": right_mean,
                "mean_diff_right_minus_left": mean_diff,
                "wilcoxon_stat": stat,
                "wilcoxon_p": p_value,
                "wilcoxon_p_bh_fdr": None,
                "decision_alpha_0_05": (
                    "insufficient_n"
                    if p_value is None
                    else ("reject_H0" if p_value < 0.05 else "fail_to_reject_H0")
                ),
                "decision_bh_fdr_alpha_0_05": None,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=_PAIRWISE_TEST_COLUMNS)

    adjusted = _bh_fdr_adjust(out["wilcoxon_p"].tolist())
    out["wilcoxon_p_bh_fdr"] = adjusted
    out["decision_bh_fdr_alpha_0_05"] = [
        "insufficient_n" if p is None else ("reject_H0" if p < 0.05 else "fail_to_reject_H0")
        for p in adjusted
    ]
    return out[_PAIRWISE_TEST_COLUMNS]


__all__ = [
    "build_pairwise_tests",
    "build_summary_by_condition",
]
