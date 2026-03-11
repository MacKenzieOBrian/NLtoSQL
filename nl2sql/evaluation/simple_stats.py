"""Minimal summaries and simple EX-only significance helpers for final-pack runs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


_SUMMARY_METRICS = ("va", "ex", "ts")
_PAIRWISE_TEST_COLUMNS = [
    "comparison",
    "left_condition_id",
    "right_condition_id",
    "n_left",
    "n_right",
    "left_mean",
    "right_mean",
    "mean_diff_right_minus_left",
    "test_name",
    "test_stat",
    "p_value",
    "decision_alpha_0_05",
]
_ALPHA = 0.05
_DET_ATOL = 1e-12


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


def _to_float_or_none(value: float | np.floating | None) -> float | None:
    """Convert finite numeric values to ``float``; return ``None`` otherwise."""
    if value is None:
        return None
    out = float(value)
    return out if np.isfinite(out) else None


def _decision_from_p(p_value: float | None) -> str:
    """Map a p-value to the alpha=0.05 decision label."""
    if p_value is None:
        return "insufficient_n"
    return "reject_H0" if p_value < _ALPHA else "fail_to_reject_H0"


def _is_deterministic(values: list[float]) -> bool:
    """Return ``True`` when all rates collapse to one constant value."""
    if not values:
        return False
    arr = np.asarray(values, dtype=float)
    return bool(np.allclose(arr, arr[0], rtol=0.0, atol=_DET_ATOL))


def compare_runs(
    left_rates: list[float],
    right_rates: list[float],
    left_k: int | None,
    right_k: int | None,
) -> dict[str, object]:
    """Compare two k=3 run-level EX vectors with one simple Mann-Whitney rule."""
    # Keep k in the signature to make this helper explicit about condition metadata.
    _ = (left_k, right_k)

    left = np.asarray(left_rates, dtype=float)
    right = np.asarray(right_rates, dtype=float)
    n_left = int(left.size)
    n_right = int(right.size)
    left_det = _is_deterministic(left_rates)
    right_det = _is_deterministic(right_rates)

    result: dict[str, object] = {
        "normality_left_p": None,
        "normality_right_p": None,
        "test_name": "none",
        "test_stat": None,
        "p_value": None,
        "decision_alpha_0_05": "insufficient_n",
    }

    if n_left == 0 or n_right == 0:
        return result

    if n_left < 2 or n_right < 2:
        return result

    result["test_name"] = "mann_whitney_u"
    stat, p_val = stats.mannwhitneyu(left, right, alternative="two-sided", method="auto")

    p_out = _to_float_or_none(p_val)
    result["test_stat"] = _to_float_or_none(stat)
    result["p_value"] = p_out
    result["decision_alpha_0_05"] = _decision_from_p(p_out)
    return result


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
    """Run the simple EX-only k=3 baseline-vs-QLoRA comparison set."""
    per_item = _coerce_per_item(per_item_df)
    run_level = (
        per_item.dropna(subset=["ex"])
        .groupby(["condition_id", "k", "seed"], sort=True)["ex"]
        .mean()
        .reset_index()
    )
    by_condition: dict[str, dict[str, object]] = {}
    for condition_id, group in run_level.groupby("condition_id", sort=True):
        k_values = [int(k) for k in group["k"].dropna().unique()]
        if len(k_values) > 1:
            raise ValueError(f"{condition_id}: expected a single k value, found {k_values}")
        by_condition[condition_id] = {
            "k": (k_values[0] if k_values else None),
            "rates": group["ex"].dropna().astype(float).tolist(),
        }

    comparisons = [
        ("llama_base_k3", "llama_qlora_k3", "Llama Base->QLoRA @k3"),
        ("qwen_base_k3", "qwen_qlora_k3", "Qwen Base->QLoRA @k3"),
    ]

    rows: list[dict[str, object]] = []
    for left_id, right_id, label in comparisons:
        if left_id not in by_condition or right_id not in by_condition:
            continue

        left_rates = list(by_condition[left_id]["rates"])  # type: ignore[index]
        right_rates = list(by_condition[right_id]["rates"])  # type: ignore[index]
        left_k = by_condition[left_id]["k"]  # type: ignore[index]
        right_k = by_condition[right_id]["k"]  # type: ignore[index]

        n_left = int(len(left_rates))
        n_right = int(len(right_rates))
        left_mean = float(np.mean(left_rates)) if n_left else None
        right_mean = float(np.mean(right_rates)) if n_right else None
        mean_diff = (right_mean - left_mean) if (left_mean is not None and right_mean is not None) else None
        test = compare_runs(
            left_rates=left_rates,
            right_rates=right_rates,
            left_k=(int(left_k) if left_k is not None else None),
            right_k=(int(right_k) if right_k is not None else None),
        )

        rows.append(
            {
                "comparison": label,
                "left_condition_id": left_id,
                "right_condition_id": right_id,
                "n_left": n_left,
                "n_right": n_right,
                "left_mean": left_mean,
                "right_mean": right_mean,
                "mean_diff_right_minus_left": mean_diff,
                "test_name": test["test_name"],
                "test_stat": test["test_stat"],
                "p_value": test["p_value"],
                "decision_alpha_0_05": test["decision_alpha_0_05"],
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=_PAIRWISE_TEST_COLUMNS)
    return out[_PAIRWISE_TEST_COLUMNS]


__all__ = [
    "build_pairwise_tests",
    "build_summary_by_condition",
]
