#!/usr/bin/env python3
"""
Research stats generator for dissertation reporting.

Single source of truth:
- reads run JSON files from `results/**/results_k*_seed*.json`
- reads ReAct extension runs from `results/agent/runs/**/results_react_200.json`

Supported matrix:
- model tags: llama, qwen
- methods: baseline (base), qlora, react (extension path)
- k values used in comparisons: 0 and 3

Outputs:
- results/analysis/per_item_metrics_primary_raw.csv
- results/analysis/run_manifest.csv
- results/analysis/stats_mean_median_shapiro.csv
- results/analysis/stats_paired_ttests.csv

Statistical approach:
- Primary test: Wilcoxon signed-rank (non-parametric, appropriate for binary 0/1 metrics)
- Effect size: Cohen's d (mean_diff / std of paired differences)
- Confidence interval: 95% CI on mean difference (t-distribution, df=n-1)
- Multiple comparisons: Benjamini-Hochberg FDR correction within each metric family
- Corroborating test: paired t-test (CLT justification at n>=600 pairs)
- Normality check: Shapiro-Wilk on paired differences (expected to reject for binary data)

Documentation:
- Wilcoxon signed-rank: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
- Paired t-test: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
- Shapiro-Wilk: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import shapiro, t as t_dist, ttest_rel, wilcoxon

METRICS = ("va", "em", "ex", "ts")
SUPPORTED_K = {0, 3}
SUPPORTED_MODEL_TAGS = {"llama", "qwen"}
SUPPORTED_METHODS = {"base", "qlora", "react"}

TTEST_COLUMNS = [
    "comparison",
    "left_condition_id",
    "right_condition_id",
    "metric",
    "pair_key",
    "matched_seeds",
    "n_pairs",
    "left_mean",
    "right_mean",
    "mean_diff_right_minus_left",
    "ci_95_lower",
    "ci_95_upper",
    "cohens_d",
    "diff_shapiro_w",
    "diff_shapiro_p",
    "diff_shapiro_decision_alpha_0_05",
    # Wilcoxon signed-rank: primary test (non-parametric, suited for binary data)
    "wilcoxon_stat",
    "wilcoxon_p",
    "wilcoxon_decision_alpha_0_05",
    "wilcoxon_p_bh_fdr",
    "wilcoxon_decision_bh_fdr_alpha_0_05",
    # Paired t-test: corroborating test (CLT justification at n>=600)
    "t_stat",
    "p_value",
    "decision_alpha_0_05",
]


@dataclass(frozen=True)
class RunSpec:
    path: Path
    condition_id: str
    model_tag: str
    method_tag: str
    k: int
    seed: int | None
    run_label: str
    eval_profile: str | None
    ts_enabled: bool | None
    run_timestamp: float

    @property
    def run_id(self) -> str:
        seed_label = "na" if self.seed is None else str(self.seed)
        return f"{self.condition_id}_s{seed_label}"


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
        "--allow-non-raw",
        action="store_true",
        help="Include runs where eval_profile is not model_only_raw.",
    )
    return parser.parse_args()


def _coerce_metric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    try:
        return float(value)
    except Exception:
        return None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        iv = int(value)
    except Exception:
        return None
    return iv


def _parse_timestamp(payload: dict[str, Any], path: Path) -> float:
    ts = payload.get("timestamp")
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            pass
    return path.stat().st_mtime


def _infer_model_tag(path: Path, payload: dict[str, Any]) -> str | None:
    md = payload.get("run_metadata") if isinstance(payload.get("run_metadata"), dict) else {}
    rm = payload.get("run_meta") if isinstance(payload.get("run_meta"), dict) else {}

    text = " ".join(
        str(x)
        for x in [
            md.get("model_alias"),
            md.get("model_id"),
            rm.get("model_alias"),
            rm.get("model_id"),
            str(path),
        ]
        if x
    ).lower()

    if "llama" in text:
        return "llama"
    if "qwen" in text:
        return "qwen"
    return None


def _infer_method_tag(path: Path, payload: dict[str, Any]) -> str | None:
    md = payload.get("run_metadata") if isinstance(payload.get("run_metadata"), dict) else {}
    rm = payload.get("run_meta") if isinstance(payload.get("run_meta"), dict) else {}
    raw_method = str(md.get("method") or rm.get("method") or "").strip().lower()
    path_text = str(path).lower()

    if raw_method in {"baseline", "base"}:
        return "base"
    if raw_method == "qlora":
        return "qlora"

    if "qlora" in path_text:
        return "qlora"
    if "baseline" in path_text:
        return "base"
    return None


def _infer_k(payload: dict[str, Any]) -> int | None:
    md = payload.get("run_metadata") if isinstance(payload.get("run_metadata"), dict) else {}
    rm = payload.get("run_meta") if isinstance(payload.get("run_meta"), dict) else {}
    return _int_or_none(payload.get("k") if payload.get("k") is not None else (md.get("k") if md.get("k") is not None else rm.get("k")))


def _infer_seed(payload: dict[str, Any]) -> int | None:
    md = payload.get("run_metadata") if isinstance(payload.get("run_metadata"), dict) else {}
    rm = payload.get("run_meta") if isinstance(payload.get("run_meta"), dict) else {}
    return _int_or_none(
        payload.get("seed") if payload.get("seed") is not None else (md.get("seed") if md.get("seed") is not None else rm.get("seed"))
    )


def _eval_profile(payload: dict[str, Any]) -> str | None:
    md = payload.get("run_metadata") if isinstance(payload.get("run_metadata"), dict) else {}
    rm = payload.get("run_meta") if isinstance(payload.get("run_meta"), dict) else {}
    val = payload.get("eval_profile") or md.get("eval_profile") or rm.get("eval_profile")
    return str(val) if val is not None else None


def _ts_enabled(payload: dict[str, Any]) -> bool | None:
    md = payload.get("run_metadata") if isinstance(payload.get("run_metadata"), dict) else {}
    rm = payload.get("run_meta") if isinstance(payload.get("run_meta"), dict) else {}
    val = md.get("ts_enabled")
    if val is None:
        val = rm.get("ts_enabled")
    if val is None:
        return None
    return bool(val)


def _make_run_label(model_tag: str, method_tag: str, k: int, seed: int | None) -> str:
    model_part = "Llama" if model_tag == "llama" else "Qwen"
    if method_tag == "react":
        method_part = "ReAct"
    elif method_tag == "base":
        method_part = "Base"
    else:
        method_part = "QLoRA"
    seed_part = "na" if seed is None else str(seed)
    return f"{model_part} {method_part} | k={k} | seed={seed_part}"


def discover_runs(
    *,
    project_root: Path,
    runs_root: Path,
    allow_non_raw: bool = False,
) -> tuple[list[RunSpec], list[dict[str, Any]]]:
    root = runs_root if runs_root.is_absolute() else (project_root / runs_root)
    files = sorted(root.rglob("results_k*_seed*.json"))

    discovered: dict[tuple[str, int | None], RunSpec] = {}
    drops: list[dict[str, Any]] = []

    for path in files:
        payload = _load_json(path)
        model_tag = _infer_model_tag(path, payload)
        method_tag = _infer_method_tag(path, payload)
        k = _infer_k(payload)
        seed = _infer_seed(payload)
        eval_profile = _eval_profile(payload)

        if model_tag not in SUPPORTED_MODEL_TAGS:
            drops.append({"path": str(path), "reason": "unsupported_or_missing_model_tag"})
            continue
        if method_tag not in SUPPORTED_METHODS:
            drops.append({"path": str(path), "reason": "unsupported_or_missing_method_tag"})
            continue
        if k not in SUPPORTED_K:
            drops.append({"path": str(path), "reason": "unsupported_k"})
            continue
        if not allow_non_raw and eval_profile not in {None, "model_only_raw"}:
            drops.append({"path": str(path), "reason": f"eval_profile={eval_profile}"})
            continue

        condition_id = f"{model_tag}_{method_tag}_k{k}"
        spec = RunSpec(
            path=path,
            condition_id=condition_id,
            model_tag=model_tag,
            method_tag=method_tag,
            k=k,
            seed=seed,
            run_label=_make_run_label(model_tag, method_tag, k, seed),
            eval_profile=eval_profile,
            ts_enabled=_ts_enabled(payload),
            run_timestamp=_parse_timestamp(payload, path),
        )

        dedup_key = (condition_id, seed)
        existing = discovered.get(dedup_key)
        if existing is None or spec.run_timestamp >= existing.run_timestamp:
            if existing is not None:
                drops.append(
                    {
                        "path": str(existing.path),
                        "reason": "superseded_by_newer_duplicate_seed",
                        "replacement": str(spec.path),
                    }
                )
            discovered[dedup_key] = spec
        else:
            drops.append({"path": str(spec.path), "reason": "older_duplicate_seed", "kept": str(existing.path)})

    return sorted(discovered.values(), key=lambda x: (x.condition_id, x.seed if x.seed is not None else -1, x.run_id)), drops


def discover_react_runs(*, project_root: Path) -> tuple[list[RunSpec], list[dict[str, Any]]]:
    agent_root = project_root / "results" / "agent" / "runs"
    files = sorted(agent_root.rglob("results_react_200.json"))

    discovered: dict[tuple[str, int | None], RunSpec] = {}
    drops: list[dict[str, Any]] = []

    for path in files:
        payload = _load_json(path)
        model_tag = _infer_model_tag(path, payload)
        config = payload.get("config", {})
        seed = _int_or_none(config.get("few_shot_seed"))
        k = _int_or_none(config.get("few_shot_k"))

        if model_tag not in SUPPORTED_MODEL_TAGS:
            drops.append({"path": str(path), "reason": "unsupported_or_missing_model_tag"})
            continue
        if k not in SUPPORTED_K:
            drops.append({"path": str(path), "reason": "unsupported_k"})
            continue

        items = payload.get("items", [])
        ts_values = [i.get("ts") for i in items if i.get("ts") is not None]
        ts_enabled = len(ts_values) > 0

        condition_id = f"{model_tag}_react_k{k}"
        spec = RunSpec(
            path=path,
            condition_id=condition_id,
            model_tag=model_tag,
            method_tag="react",
            k=k,
            seed=seed,
            run_label=_make_run_label(model_tag, "react", k, seed),
            eval_profile="react",
            ts_enabled=ts_enabled,
            run_timestamp=_parse_timestamp(payload, path),
        )

        dedup_key = (condition_id, seed)
        existing = discovered.get(dedup_key)
        if existing is None or spec.run_timestamp >= existing.run_timestamp:
            if existing is not None:
                drops.append(
                    {
                        "path": str(existing.path),
                        "reason": "superseded_by_newer_duplicate_seed",
                        "replacement": str(spec.path),
                    }
                )
            discovered[dedup_key] = spec
        else:
            drops.append({"path": str(spec.path), "reason": "older_duplicate_seed", "kept": str(existing.path)})

    return sorted(discovered.values(), key=lambda x: (x.condition_id, x.seed if x.seed is not None else -1, x.run_id)), drops


def _rows_from_run(spec: RunSpec, payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_raw = payload.get("results") if isinstance(payload.get("results"), list) else payload.get("items")
    if not isinstance(rows_raw, list):
        raise ValueError(f"Missing results list in: {spec.path}")

    rows: list[dict[str, Any]] = []
    for item in rows_raw:
        example_id = item.get("i")
        if example_id is None:
            example_id = item.get("example_id")
        rows.append(
            {
                "run_id": spec.run_id,
                "condition_id": spec.condition_id,
                "run_label": spec.run_label,
                "method": spec.method_tag,
                "model_tag": spec.model_tag,
                "k": spec.k,
                "seed": spec.seed,
                "example_id": example_id,
                "nlq": item.get("nlq", ""),
                "va": _coerce_metric(item.get("va")),
                "em": _coerce_metric(item.get("em")),
                "ex": _coerce_metric(item.get("ex")),
                "ts": _coerce_metric(item.get("ts")),
                "source_json": str(spec.path),
            }
        )

    manifest = {
        "run_id": spec.run_id,
        "condition_id": spec.condition_id,
        "run_label": spec.run_label,
        "method": spec.method_tag,
        "model_tag": spec.model_tag,
        "k": spec.k,
        "seed": spec.seed,
        "n_items": len(rows_raw),
        "eval_profile": spec.eval_profile,
        "ts_enabled": spec.ts_enabled,
        "source_json": str(spec.path),
    }
    return rows, manifest


def build_tables_from_runs(specs: list[RunSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for spec in specs:
        payload = _load_json(spec.path)
        rows, manifest = _rows_from_run(spec, payload)
        all_rows.extend(rows)
        manifest_rows.append(manifest)
    return pd.DataFrame(all_rows), pd.DataFrame(manifest_rows)


def _prepare_per_item(df: pd.DataFrame) -> pd.DataFrame:
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


def compute_mean_median_shapiro(per_item: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_id, group in per_item.groupby("run_id", sort=True):
        first = group.iloc[0]
        for metric in METRICS:
            values = group[metric].dropna()
            n = int(len(values))
            mean = float(values.mean()) if n else None
            median = float(values.median()) if n else None
            std = float(values.std(ddof=1)) if n > 1 else None

            if n < 3:
                w = None
                p = None
                decision = "insufficient_n"
            else:
                w_val, p_val = shapiro(values.to_numpy())
                w = float(w_val)
                p = float(p_val)
                decision = "reject_normality" if p < 0.05 else "fail_to_reject_normality"

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
                    "shapiro_scope": "per_run_metric_values",
                    "n": n,
                    "mean": mean,
                    "median": median,
                    "std": std,
                    "shapiro_w": w,
                    "shapiro_p": p,
                    "decision_alpha_0_05": decision,
                }
            )
    return pd.DataFrame(rows)


def _bh_fdr_adjust(pvalues: list[float | None]) -> list[float | None]:
    """Benjamini-Hochberg FDR-adjusted p-values. Returns None where input is None."""
    result: list[float | None] = list(pvalues)
    valid_idx = [i for i, p in enumerate(pvalues) if p is not None]
    if not valid_idx:
        return result
    m = len(valid_idx)
    sorted_idx = sorted(valid_idx, key=lambda i: pvalues[i])  # ascending by raw p
    adj = [pvalues[i] * m / (rank + 1) for rank, i in enumerate(sorted_idx)]
    for j in range(m - 2, -1, -1):
        adj[j] = min(adj[j], adj[j + 1])
    for rank, orig_i in enumerate(sorted_idx):
        result[orig_i] = min(adj[rank], 1.0)
    return result


def build_planned_comparisons(condition_ids: set[str]) -> list[tuple[str, str, str]]:
    plan = [
        ("llama_base_k0", "llama_base_k3", "Llama Base k0->k3"),
        ("llama_qlora_k0", "llama_qlora_k3", "Llama QLoRA k0->k3"),
        ("qwen_base_k0", "qwen_base_k3", "Qwen Base k0->k3"),
        ("qwen_qlora_k0", "qwen_qlora_k3", "Qwen QLoRA k0->k3"),
        ("llama_base_k0", "llama_qlora_k0", "Llama Base->QLoRA @k0"),
        ("llama_base_k3", "llama_qlora_k3", "Llama Base->QLoRA @k3"),
        ("qwen_base_k0", "qwen_qlora_k0", "Qwen Base->QLoRA @k0"),
        ("qwen_base_k3", "qwen_qlora_k3", "Qwen Base->QLoRA @k3"),
        # ReAct extension comparisons (extension path, descriptive + inferential)
        ("llama_qlora_k3", "llama_react_k3", "Llama QLoRA->ReAct @k3"),
        ("llama_base_k3", "llama_react_k3", "Llama Base->ReAct @k3"),
        ("qwen_qlora_k3", "qwen_react_k3", "Qwen QLoRA->ReAct @k3"),
        ("qwen_base_k3", "qwen_react_k3", "Qwen Base->ReAct @k3"),
    ]
    return [(l, r, label) for (l, r, label) in plan if l in condition_ids and r in condition_ids]


def _drop_duplicate_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return df.sort_values(["run_id", *keys]).drop_duplicates(subset=keys, keep="first")


def _join_for_pair(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, str, list[int]]:
    left = left.copy()
    right = right.copy()

    if left["seed"].notna().all() and right["seed"].notna().all() and left["example_id"].notna().all() and right["example_id"].notna().all():
        keys = ["seed", "example_id"]
        pair_key = "seed+example_id"
    else:
        keys = ["seed", "nlq"]
        pair_key = "seed+nlq"

    left_keyed = _drop_duplicate_keys(left, keys)
    right_keyed = _drop_duplicate_keys(right, keys)
    merged = left_keyed[keys + list(METRICS)].merge(
        right_keyed[keys + list(METRICS)],
        on=keys,
        how="inner",
        suffixes=("_left", "_right"),
    )

    seed_vals = sorted({int(s) for s in merged["seed"].dropna().tolist()}) if "seed" in merged.columns else []
    return merged, pair_key, seed_vals


def compute_paired_ttests(per_item: pd.DataFrame, comparisons: list[tuple[str, str, str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_condition = {cid: grp.copy() for cid, grp in per_item.groupby("condition_id")}

    for left_id, right_id, label in comparisons:
        left_df = by_condition[left_id]
        right_df = by_condition[right_id]
        merged, pair_key, matched_seeds = _join_for_pair(left_df, right_df)

        for metric in METRICS:
            left_col = f"{metric}_left"
            right_col = f"{metric}_right"
            valid = merged[[left_col, right_col]].dropna().astype(float)
            n = int(len(valid))

            diffs_arr = (valid[right_col] - valid[left_col]).to_numpy() if n >= 2 else None

            # Shapiro-Wilk on paired differences (assumption check; expected to reject for binary data)
            if n < 3:
                diff_shapiro_w = None
                diff_shapiro_p = None
                diff_shapiro_decision = "insufficient_n"
            else:
                w_val, p_val = shapiro(diffs_arr)
                diff_shapiro_w = float(w_val)
                diff_shapiro_p = float(p_val)
                diff_shapiro_decision = "reject_normality" if diff_shapiro_p < 0.05 else "fail_to_reject_normality"

            if n < 2:
                left_mean = right_mean = diff = None
                ci_lower = ci_upper = cohens_d = None
                wilcoxon_stat = wilcoxon_p = None
                wilcoxon_decision = "insufficient_n"
                t_stat = p_value = None
                t_decision = "insufficient_n"
            else:
                left_mean = float(valid[left_col].mean())
                right_mean = float(valid[right_col].mean())
                diff = right_mean - left_mean
                std_diff = float(diffs_arr.std(ddof=1))

                # 95% CI on mean difference using t-distribution
                se = std_diff / (n ** 0.5)
                t_crit = float(t_dist.ppf(0.975, df=n - 1))
                ci_lower = diff - t_crit * se
                ci_upper = diff + t_crit * se

                # Cohen's d for paired differences (dz)
                cohens_d = (diff / std_diff) if std_diff > 0 else 0.0

                # Wilcoxon signed-rank: primary test (non-parametric, suited for binary 0/1 data)
                # zero_method='wilcox' excludes tied-zero differences from the rank sum
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        wx = wilcoxon(diffs_arr, zero_method="wilcox", alternative="two-sided")
                    wilcoxon_stat = float(wx.statistic)
                    wilcoxon_p = float(wx.pvalue)
                    wilcoxon_decision = "reject_H0" if wilcoxon_p < 0.05 else "fail_to_reject_H0"
                except ValueError:
                    # all differences are zero — conditions are identical on this metric
                    wilcoxon_stat = 0.0
                    wilcoxon_p = 1.0
                    wilcoxon_decision = "fail_to_reject_H0"

                # Paired t-test: corroborating test (CLT justification at n>=600)
                test = ttest_rel(valid[right_col], valid[left_col], nan_policy="omit")
                t_stat = float(test.statistic)
                p_value = float(test.pvalue)
                t_decision = "reject_H0" if p_value < 0.05 else "fail_to_reject_H0"

            rows.append(
                {
                    "comparison": label,
                    "left_condition_id": left_id,
                    "right_condition_id": right_id,
                    "metric": metric,
                    "pair_key": pair_key,
                    "matched_seeds": ",".join(str(s) for s in matched_seeds),
                    "n_pairs": n,
                    "left_mean": left_mean,
                    "right_mean": right_mean,
                    "mean_diff_right_minus_left": diff,
                    "ci_95_lower": ci_lower,
                    "ci_95_upper": ci_upper,
                    "cohens_d": cohens_d,
                    "diff_shapiro_w": diff_shapiro_w,
                    "diff_shapiro_p": diff_shapiro_p,
                    "diff_shapiro_decision_alpha_0_05": diff_shapiro_decision,
                    "wilcoxon_stat": wilcoxon_stat,
                    "wilcoxon_p": wilcoxon_p,
                    "wilcoxon_decision_alpha_0_05": wilcoxon_decision,
                    "wilcoxon_p_bh_fdr": None,  # filled in BH pass below
                    "wilcoxon_decision_bh_fdr_alpha_0_05": None,  # filled in BH pass below
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "decision_alpha_0_05": t_decision,
                }
            )

    if not rows:
        return pd.DataFrame(columns=TTEST_COLUMNS)

    df = pd.DataFrame(rows, columns=TTEST_COLUMNS)

    # BH FDR correction applied within each metric family (12 comparisons per metric)
    for metric in METRICS:
        mask = df["metric"] == metric
        adj_ps = _bh_fdr_adjust(df.loc[mask, "wilcoxon_p"].tolist())
        df.loc[mask, "wilcoxon_p_bh_fdr"] = adj_ps
        df.loc[mask, "wilcoxon_decision_bh_fdr_alpha_0_05"] = [
            "reject_H0" if p is not None and p < 0.05 else ("fail_to_reject_H0" if p is not None else "insufficient_n")
            for p in adj_ps
        ]

    return df


def generate(
    *,
    runs_root: Path = Path("results"),
    per_item_csv: Path = Path("results/analysis/per_item_metrics_primary_raw.csv"),
    out_dir: Path = Path("results/analysis"),
    project_root: Path | None = None,
    allow_non_raw: bool = False,
) -> dict[str, Any]:
    if project_root is None:
        project_root = Path.cwd()

    per_item_csv = per_item_csv if per_item_csv.is_absolute() else (project_root / per_item_csv)
    out_dir = out_dir if out_dir.is_absolute() else (project_root / out_dir)
    runs_root = runs_root if runs_root.is_absolute() else (project_root / runs_root)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_item_csv.parent.mkdir(parents=True, exist_ok=True)

    specs, drops = discover_runs(project_root=project_root, runs_root=runs_root, allow_non_raw=allow_non_raw)
    react_specs, react_drops = discover_react_runs(project_root=project_root)
    drops.extend(react_drops)
    all_specs = sorted(
        specs + react_specs,
        key=lambda x: (x.condition_id, x.seed if x.seed is not None else -1, x.run_id),
    )

    if not all_specs:
        raise FileNotFoundError(
            f"No valid run files found under: {runs_root}\n"
            "Expected files like: results_k0_seed7.json, results_k3_seed7.json"
        )

    items_df, manifest_df = build_tables_from_runs(all_specs)
    per_item = _prepare_per_item(items_df)

    condition_ids = set(manifest_df["condition_id"].dropna().astype(str).tolist())
    comparisons = build_planned_comparisons(condition_ids)

    manifest_out = out_dir / "run_manifest.csv"
    shapiro_out = out_dir / "stats_mean_median_shapiro.csv"
    ttests_out = out_dir / "stats_paired_ttests.csv"

    paired_tests = compute_paired_ttests(per_item, comparisons).sort_values(["comparison", "metric"])

    manifest_df.sort_values(["model_tag", "method", "k", "seed", "run_id"]).to_csv(manifest_out, index=False)
    per_item.sort_values(["condition_id", "seed", "example_id"]).to_csv(per_item_csv, index=False)
    compute_mean_median_shapiro(per_item).sort_values(["condition_id", "seed", "metric"]).to_csv(shapiro_out, index=False)
    paired_tests.to_csv(ttests_out, index=False)

    return {
        "design": "multi_model_multi_method_from_results_tree",
        "runs_root": str(runs_root),
        "run_files_found": int(len(list(runs_root.rglob('results_k*_seed*.json')))),
        "react_files_found": int(len(react_specs)),
        "runs_included": int(len(all_specs)),
        "conditions_included": sorted(condition_ids),
        "comparisons_planned": int(len(comparisons)),
        "dropped_files": drops,
        "per_item_csv": str(per_item_csv),
        "outputs": [str(manifest_out), str(shapiro_out), str(ttests_out)],
    }


def main() -> None:
    args = parse_args()
    summary = generate(
        runs_root=args.runs_root,
        per_item_csv=args.per_item_csv,
        out_dir=args.out_dir,
        allow_non_raw=args.allow_non_raw,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
