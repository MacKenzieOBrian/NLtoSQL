"""Statistics helpers for the research comparison workflow."""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
from scipy.stats import shapiro, t as t_dist, ttest_rel, wilcoxon


METRICS = ("va", "em", "ex", "ts")
PLANNED_COMPARISONS = [
    ("llama_base_k0", "llama_base_k3", "Llama Base k0->k3"),
    ("llama_qlora_k0", "llama_qlora_k3", "Llama QLoRA k0->k3"),
    ("qwen_base_k0", "qwen_base_k3", "Qwen Base k0->k3"),
    ("qwen_qlora_k0", "qwen_qlora_k3", "Qwen QLoRA k0->k3"),
    ("llama_base_k0", "llama_qlora_k0", "Llama Base->QLoRA @k0"),
    ("llama_base_k3", "llama_qlora_k3", "Llama Base->QLoRA @k3"),
    ("qwen_base_k0", "qwen_qlora_k0", "Qwen Base->QLoRA @k0"),
    ("qwen_base_k3", "qwen_qlora_k3", "Qwen Base->QLoRA @k3"),
    ("llama_qlora_k3", "llama_react_k3", "Llama QLoRA->ReAct @k3"),
    ("llama_base_k3", "llama_react_k3", "Llama Base->ReAct @k3"),
    ("qwen_qlora_k3", "qwen_react_k3", "Qwen QLoRA->ReAct @k3"),
    ("qwen_base_k3", "qwen_react_k3", "Qwen Base->ReAct @k3"),
]


def _coerce_metric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    try:
        return float(value)
    except Exception:
        return None


def _shapiro_result(values: Any, n: int) -> tuple[float | None, float | None, str]:
    # Inspired by the standard Shapiro-Wilk normality check before reading paired-difference tests.
    if n < 3:
        return None, None, "insufficient_n"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            statistic, p_value = shapiro(values)
    except Exception:
        return None, None, "error"
    decision = "reject_normality" if p_value < 0.05 else "fail_to_reject_normality"
    return float(statistic), float(p_value), decision


def prepare_per_item_table(df: pd.DataFrame) -> pd.DataFrame:
    required = {"run_id", "condition_id", "run_label", "method", "model_tag", "k", "seed", "nlq", "va", "em", "ex", "ts"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Per-item rows missing columns: {missing}")

    out = df.copy()
    for metric in METRICS:
        out[metric] = pd.to_numeric(out[metric], errors="coerce")
    out["k"] = pd.to_numeric(out["k"], errors="coerce")
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce")
    if "example_id" in out.columns:
        out["example_id"] = pd.to_numeric(out["example_id"], errors="coerce")
    return out


def compute_mean_median_std(per_item: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_id, group in per_item.groupby("run_id", sort=True):
        first = group.iloc[0]
        for metric in METRICS:
            values = group[metric].dropna()
            n = int(len(values))
            shapiro_w, shapiro_p, shapiro_decision = _shapiro_result(values.to_numpy(), n)
            rows.append(
                {
                    "run_id": run_id,
                    "condition_id": first["condition_id"],
                    "run_label": first["run_label"],
                    "model_tag": first["model_tag"],
                    "method": first["method"],
                    "k": int(first["k"]) if pd.notna(first["k"]) else None,
                    "seed": int(first["seed"]) if pd.notna(first["seed"]) else None,
                    "metric": metric,
                    "n": n,
                    "mean": float(values.mean()) if n else None,
                    "median": float(values.median()) if n else None,
                    "std": float(values.std(ddof=1)) if n > 1 else None,
                    "shapiro_w": shapiro_w,
                    "shapiro_p": shapiro_p,
                    "shapiro_decision_alpha_0_05": shapiro_decision,
                }
            )
    return pd.DataFrame(rows)


def planned_comparisons(condition_ids: set[str]) -> list[tuple[str, str, str]]:
    return [(left, right, label) for left, right, label in PLANNED_COMPARISONS if left in condition_ids and right in condition_ids]


def _join_pair(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, str, list[int]]:
    if left["seed"].notna().all() and right["seed"].notna().all() and left["example_id"].notna().all() and right["example_id"].notna().all():
        keys = ["seed", "example_id"]
        pair_key = "seed+example_id"
    else:
        keys = ["seed", "nlq"]
        pair_key = "seed+nlq"

    left_keyed = left.sort_values(["run_id", *keys]).drop_duplicates(subset=keys, keep="first")
    right_keyed = right.sort_values(["run_id", *keys]).drop_duplicates(subset=keys, keep="first")
    merged = left_keyed[keys + list(METRICS)].merge(
        right_keyed[keys + list(METRICS)],
        on=keys,
        how="inner",
        suffixes=("_left", "_right"),
    )
    matched_seeds = sorted({int(seed) for seed in merged["seed"].dropna().tolist()}) if "seed" in merged.columns else []
    return merged, pair_key, matched_seeds


def _paired_summary(valid: pd.DataFrame, left_col: str, right_col: str) -> dict[str, Any]:
    n_pairs = int(len(valid))
    if n_pairs < 2:
        return {
            "n_pairs": n_pairs,
            "left_mean": None,
            "right_mean": None,
            "mean_diff_right_minus_left": None,
            "ci_95_lower": None,
            "ci_95_upper": None,
            "cohens_d": None,
            "diffs": None,
        }

    left_mean = float(valid[left_col].mean())
    right_mean = float(valid[right_col].mean())
    diffs = (valid[right_col] - valid[left_col]).to_numpy()
    diff_mean = right_mean - left_mean
    std_diff = float(diffs.std(ddof=1))
    se = std_diff / (n_pairs ** 0.5)
    t_crit = float(t_dist.ppf(0.975, df=n_pairs - 1))
    ci_low = diff_mean - (t_crit * se)
    ci_high = diff_mean + (t_crit * se)
    cohens_d = (diff_mean / std_diff) if std_diff > 0 else 0.0
    return {
        "n_pairs": n_pairs,
        "left_mean": left_mean,
        "right_mean": right_mean,
        "mean_diff_right_minus_left": diff_mean,
        "ci_95_lower": ci_low,
        "ci_95_upper": ci_high,
        "cohens_d": cohens_d,
        "diffs": diffs,
    }


def _wilcoxon_result(diffs: Any, n_pairs: int) -> tuple[float | None, float | None, str]:
    # Inspired by the usual Wilcoxon signed-rank test for paired, nonparametric comparisons.
    if n_pairs < 2:
        return None, None, "insufficient_n"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        p_value = float(test.pvalue)
        decision = "reject_H0" if p_value < 0.05 else "fail_to_reject_H0"
        return float(test.statistic), p_value, decision
    except ValueError:
        return 0.0, 1.0, "fail_to_reject_H0"


def _paired_ttest_result(valid: pd.DataFrame, left_col: str, right_col: str, n_pairs: int) -> tuple[float | None, float | None, str]:
    # Keep a paired t-test as a simple secondary check on the same matched items.
    if n_pairs < 2:
        return None, None, "insufficient_n"
    test = ttest_rel(valid[right_col], valid[left_col], nan_policy="omit")
    p_value = float(test.pvalue)
    decision = "reject_H0" if p_value < 0.05 else "fail_to_reject_H0"
    return float(test.statistic), p_value, decision


def _bh_fdr_adjust(pvalues: list[float | None]) -> list[float | None]:
    """Benjamini-Hochberg FDR correction for one metric family.

    Inspired by the common BH adjustment used when many related p-values are tested together.
    """
    adjusted: list[float | None] = list(pvalues)
    ranked = sorted((float(p), idx) for idx, p in enumerate(pvalues) if p is not None)
    if not ranked:
        return adjusted

    m = len(ranked)
    ranked_adjusted = [0.0] * m
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        p_value, _ = ranked[rank]
        candidate = min(1.0, p_value * m / (rank + 1))
        running_min = min(running_min, candidate)
        ranked_adjusted[rank] = running_min

    for rank, (_, original_index) in enumerate(ranked):
        adjusted[original_index] = ranked_adjusted[rank]
    return adjusted


def _bh_decision(p_value: float | None) -> str:
    if p_value is None:
        return "insufficient_n"
    return "reject_H0" if p_value < 0.05 else "fail_to_reject_H0"


def compute_paired_tests(per_item: pd.DataFrame, comparisons: list[tuple[str, str, str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_condition = {condition_id: group.copy() for condition_id, group in per_item.groupby("condition_id")}

    for left_id, right_id, label in comparisons:
        left_df = by_condition[left_id]
        right_df = by_condition[right_id]
        merged, pair_key, matched_seeds = _join_pair(left_df, right_df)

        for metric in METRICS:
            left_col = f"{metric}_left"
            right_col = f"{metric}_right"
            valid = merged[[left_col, right_col]].dropna().astype(float)
            summary = _paired_summary(valid, left_col, right_col)
            n_pairs = summary["n_pairs"]
            diffs = summary["diffs"]
            wilcoxon_stat, wilcoxon_p, wilcoxon_decision = _wilcoxon_result(diffs, n_pairs)
            t_stat, t_p, t_decision = _paired_ttest_result(valid, left_col, right_col, n_pairs)
            diff_shapiro_w, diff_shapiro_p, diff_shapiro_decision = _shapiro_result(diffs if diffs is not None else [], n_pairs)

            rows.append(
                {
                    "comparison": label,
                    "left_condition_id": left_id,
                    "right_condition_id": right_id,
                    "metric": metric,
                    "pair_key": pair_key,
                    "matched_seeds": ",".join(str(seed) for seed in matched_seeds),
                    "n_pairs": n_pairs,
                    "left_mean": summary["left_mean"],
                    "right_mean": summary["right_mean"],
                    "mean_diff_right_minus_left": summary["mean_diff_right_minus_left"],
                    "ci_95_lower": summary["ci_95_lower"],
                    "ci_95_upper": summary["ci_95_upper"],
                    "cohens_d": summary["cohens_d"],
                    "wilcoxon_stat": wilcoxon_stat,
                    "wilcoxon_p": wilcoxon_p,
                    "wilcoxon_decision_alpha_0_05": wilcoxon_decision,
                    "wilcoxon_p_bh_fdr": None,
                    "wilcoxon_decision_bh_fdr_alpha_0_05": None,
                    "t_stat": t_stat,
                    "p_value": t_p,
                    "decision_alpha_0_05": t_decision,
                    "diff_shapiro_w": diff_shapiro_w,
                    "diff_shapiro_p": diff_shapiro_p,
                    "diff_shapiro_decision_alpha_0_05": diff_shapiro_decision,
                }
            )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    for metric in METRICS:
        mask = out["metric"] == metric
        adjusted = _bh_fdr_adjust(out.loc[mask, "wilcoxon_p"].tolist())
        out.loc[mask, "wilcoxon_p_bh_fdr"] = adjusted
        out.loc[mask, "wilcoxon_decision_bh_fdr_alpha_0_05"] = [_bh_decision(p_value) for p_value in adjusted]
    return out


__all__ = [
    "METRICS",
    "PLANNED_COMPARISONS",
    "compute_mean_median_std",
    "compute_paired_tests",
    "planned_comparisons",
    "prepare_per_item_table",
]
