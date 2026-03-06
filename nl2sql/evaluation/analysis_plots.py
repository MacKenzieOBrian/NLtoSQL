"""
Reporting helpers used by notebook 06.

These functions format already-computed comparison outputs; they are not part
of the core NL-to-SQL pipeline or statistical generation step.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

CONDITION_ORDER = [
    "llama_base_k0",
    "llama_base_k3",
    "llama_qlora_k0",
    "llama_qlora_k3",
    "llama_react_k3",
    "qwen_base_k0",
    "qwen_base_k3",
    "qwen_qlora_k0",
    "qwen_qlora_k3",
    "qwen_react_k3",
]

CONDITION_LABELS = [
    "Llama\nBase k=0",
    "Llama\nBase k=3",
    "Llama\nQLoRA k=0",
    "Llama\nQLoRA k=3",
    "Llama\nReAct k=3",
    "Qwen\nBase k=0",
    "Qwen\nBase k=3",
    "Qwen\nQLoRA k=0",
    "Qwen\nQLoRA k=3",
    "Qwen\nReAct k=3",
]

CONDITION_COLORS = [
    "#9ecae1",
    "#3182bd",
    "#fdae6b",
    "#e6550d",
    "#31a354",
    "#9ecae1",
    "#3182bd",
    "#fdae6b",
    "#e6550d",
    "#31a354",
]


def print_ex_hypothesis_results(ttest_df: pd.DataFrame) -> None:
    ex_rows = ttest_df[ttest_df["metric"] == "ex"].copy()
    if ex_rows.empty:
        print("No EX comparisons available yet. Add matching run pairs first.")
        return

    for _, row in ex_rows.sort_values("comparison").iterrows():
        delta = row["mean_diff_right_minus_left"]
        p_value = row["wilcoxon_p_bh_fdr"]
        decision = row["wilcoxon_decision_bh_fdr_alpha_0_05"]
        direction = "improved" if pd.notna(delta) and delta > 0 else "decreased"
        significance = "significant" if decision == "reject_H0" else "not significant"

        print(f"Comparison: {row['comparison']}")
        if pd.isna(delta):
            print("  EX delta: unavailable (insufficient matched pairs)")
        else:
            print(f"  EX {direction} by {abs(delta):.3f}")
        print(f"  Wilcoxon p (BH-FDR): {p_value:.4g} ({significance})")
        print(f"  Decision: {decision}")
        print()


def plot_condition_metric_bar(
    stats_df: pd.DataFrame,
    *,
    metric: str = "ex",
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt
    import numpy as np

    metric_df = stats_df[(stats_df["metric"] == metric) & (stats_df["n"] > 0)]
    cond_stats = metric_df.groupby("condition_id")["mean"].agg(mean="mean", std="std").reindex(CONDITION_ORDER)

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(CONDITION_ORDER))
    bars = ax.bar(
        x,
        cond_stats["mean"].fillna(0),
        color=CONDITION_COLORS,
        yerr=cond_stats["std"].fillna(0),
        capsize=3,
        edgecolor="white",
        linewidth=0.5,
    )

    for bar, value in zip(bars, cond_stats["mean"].fillna(0)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    metric_label = metric.upper()
    ax.axvline(4.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_LABELS, fontsize=9)
    ax.set_ylabel("Execution Accuracy (EX)" if metric == "ex" else f"{metric_label} Rate")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Mean {metric_label} Rate by Condition (error bars = std across seeds)")
    ax.text(2.0, 1.03, "Llama-3-8B", ha="center", fontsize=10, fontweight="bold")
    ax.text(7.0, 1.03, "Qwen-2.5-7B", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_ex_ci_forest(ttest_df: pd.DataFrame) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ex_pairs = ttest_df[
        (ttest_df["metric"] == "ex") &
        (pd.to_numeric(ttest_df["n_pairs"], errors="coerce") > 0)
    ].copy()
    ex_pairs = ex_pairs.sort_values("mean_diff_right_minus_left", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 7))
    for idx, row in ex_pairs.iterrows():
        mid = row["mean_diff_right_minus_left"]
        low = row["ci_95_lower"]
        high = row["ci_95_upper"]
        significant = row["wilcoxon_decision_bh_fdr_alpha_0_05"] == "reject_H0"
        if significant and mid > 0:
            color = "#2ca02c"
        elif significant and mid < 0:
            color = "#d62728"
        else:
            color = "#aaaaaa"

        ax.plot([low, high], [idx, idx], color=color, linewidth=2.5)
        ax.plot(mid, idx, "o", color=color, markersize=7)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(ex_pairs)))
    ax.set_yticklabels(ex_pairs["comparison"].tolist(), fontsize=9)
    ax.set_xlabel("Mean EX Difference (right - left condition)")
    ax.set_title(
        "95% CI on Mean EX Difference per Comparison\n"
        "Green = significant positive  |  Red = significant decrease  |  Grey = not significant"
    )
    ax.legend(
        handles=[
            mpatches.Patch(facecolor="#2ca02c", label="Significant improvement"),
            mpatches.Patch(facecolor="#d62728", label="Significant decrease"),
            mpatches.Patch(facecolor="#aaaaaa", label="Not significant (BH-FDR alpha=0.05)"),
        ],
        loc="lower right",
        fontsize=9,
    )
    plt.tight_layout()
    plt.show()
    return fig, ax


def compute_failure_breakdown(per_item_df: pd.DataFrame) -> pd.DataFrame:
    out = per_item_df.copy()
    out["syntax_fail"] = (out["va"] == 0).astype(int)
    out["semantic_fail"] = ((out["va"] == 1) & (out["ex"] == 0)).astype(int)
    out["near_miss"] = ((out["ex"] == 1) & (out["em"] == 0)).astype(int)
    out["exact_match"] = (out["em"] == 1).astype(int)

    breakdown = (
        out.groupby("condition_id")[["syntax_fail", "semantic_fail", "near_miss", "exact_match"]]
        .mean()
        .round(3)
        .reindex([cond for cond in CONDITION_ORDER if cond in out["condition_id"].unique()])
    )
    breakdown.columns = [
        "Syntax Fail (VA=0)",
        "Semantic Fail (VA=1, EX=0)",
        "Near Miss (EX=1, EM=0)",
        "Exact Match (EM=1)",
    ]
    return breakdown


__all__ = [
    "compute_failure_breakdown",
    "CONDITION_COLORS",
    "CONDITION_LABELS",
    "CONDITION_ORDER",
    "plot_condition_metric_bar",
    "plot_ex_ci_forest",
    "print_ex_hypothesis_results",
]
