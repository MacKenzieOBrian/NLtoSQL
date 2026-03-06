"""
Build comparison tables from saved baseline, QLoRA, and ReAct run JSON files.
"""

from __future__ import annotations

import argparse
import json
import re
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
PRIMARY_METHODS = {"base", "qlora"}
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
        seed_text = "na" if self.seed is None else str(self.seed)
        return f"{self.condition_id}_s{seed_text}"


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
    return parser.parse_args()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_metric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    try:
        return float(value)
    except Exception:
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _parse_timestamp(payload: dict[str, Any], path: Path) -> float:
    timestamp = payload.get("timestamp")
    if isinstance(timestamp, str):
        try:
            return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
        except Exception:
            pass
    return path.stat().st_mtime


def _model_tag_from_payload(payload: dict[str, Any]) -> str | None:
    metadata = _as_dict(payload.get("run_metadata"))
    text = " ".join(str(x) for x in [metadata.get("model_alias"), metadata.get("model_id")] if x).lower()
    if "llama" in text:
        return "llama"
    if "qwen" in text:
        return "qwen"
    return None


def _make_run_label(model_tag: str, method_tag: str, k: int, seed: int | None) -> str:
    model_part = "Llama" if model_tag == "llama" else "Qwen"
    method_part = {"base": "Base", "qlora": "QLoRA", "react": "ReAct"}[method_tag]
    seed_part = "na" if seed is None else str(seed)
    return f"{model_part} {method_part} | k={k} | seed={seed_part}"


def _add_drop(drops: list[dict[str, Any]], path: Path, reason: str, **extra: Any) -> None:
    row = {"path": str(path), "reason": reason}
    row.update(extra)
    drops.append(row)


def _keep_newest(
    discovered: dict[tuple[str, int | None], RunSpec],
    drops: list[dict[str, Any]],
    dedup_key: tuple[str, int | None],
    spec: RunSpec,
) -> None:
    existing = discovered.get(dedup_key)
    if existing is None or spec.run_timestamp >= existing.run_timestamp:
        if existing is not None:
            _add_drop(
                drops,
                existing.path,
                "superseded_by_newer_duplicate_seed",
                replacement=str(spec.path),
            )
        discovered[dedup_key] = spec
        return
    _add_drop(drops, spec.path, "older_duplicate_seed", kept=str(existing.path))


def _primary_method_from_path(path: Path) -> str | None:
    path_text = str(path).lower()
    if "/baseline/runs/" in path_text:
        return "base"
    if "/qlora/runs/" in path_text:
        return "qlora"
    return None


def _parse_primary_filename(path: Path) -> tuple[int | None, int | None]:
    match = re.fullmatch(r"results_k(\d+)_seed(\d+)", path.stem)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _eval_profile(payload: dict[str, Any]) -> str | None:
    value = payload.get("eval_profile")
    return str(value) if value is not None else None


def _ts_enabled_from_payload(payload: dict[str, Any]) -> bool | None:
    metadata = _as_dict(payload.get("run_metadata"))
    value = metadata.get("ts_enabled")
    return bool(value) if value is not None else None


def _primary_spec_from_file(path: Path, payload: dict[str, Any]) -> tuple[RunSpec | None, str | None]:
    model_tag = _model_tag_from_payload(payload)
    method_tag = _primary_method_from_path(path)
    k, seed = _parse_primary_filename(path)
    eval_profile = _eval_profile(payload)

    if model_tag not in SUPPORTED_MODEL_TAGS:
        return None, "unsupported_or_missing_model_tag"
    if method_tag not in PRIMARY_METHODS:
        return None, "unsupported_or_missing_method_tag"
    if k not in SUPPORTED_K:
        return None, "unsupported_or_missing_k"
    if seed is None:
        return None, "missing_seed_in_filename"
    if eval_profile not in {None, "model_only_raw"}:
        return None, f"eval_profile={eval_profile}"

    condition_id = f"{model_tag}_{method_tag}_k{k}"
    return RunSpec(
        path=path,
        condition_id=condition_id,
        model_tag=model_tag,
        method_tag=method_tag,
        k=k,
        seed=seed,
        run_label=_make_run_label(model_tag, method_tag, k, seed),
        eval_profile=eval_profile,
        ts_enabled=_ts_enabled_from_payload(payload),
        run_timestamp=_parse_timestamp(payload, path),
    ), None


def _react_spec_from_file(path: Path, payload: dict[str, Any]) -> tuple[RunSpec | None, str | None]:
    model_tag = _model_tag_from_payload(payload)
    config = _as_dict(payload.get("config"))
    seed = _int_or_none(config.get("few_shot_seed"))
    k = _int_or_none(config.get("few_shot_k"))

    if model_tag not in SUPPORTED_MODEL_TAGS:
        return None, "unsupported_or_missing_model_tag"
    if k not in SUPPORTED_K:
        return None, "unsupported_k"

    items = payload.get("items", [])
    ts_enabled = any(item.get("ts") is not None for item in items if isinstance(item, dict))
    condition_id = f"{model_tag}_react_k{k}"
    return RunSpec(
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
    ), None


def _sort_specs(specs: list[RunSpec]) -> list[RunSpec]:
    return sorted(specs, key=lambda spec: (spec.condition_id, spec.seed if spec.seed is not None else -1, spec.run_id))


def discover_primary_runs(*, project_root: Path, runs_root: Path) -> tuple[list[RunSpec], list[dict[str, Any]]]:
    root = runs_root if runs_root.is_absolute() else (project_root / runs_root)
    discovered: dict[tuple[str, int | None], RunSpec] = {}
    drops: list[dict[str, Any]] = []

    for subdir in ("baseline", "qlora"):
        run_dir = root / subdir / "runs"
        if not run_dir.exists():
            continue
        for path in sorted(run_dir.rglob("results_k*_seed*.json")):
            payload = _load_json(path)
            spec, reason = _primary_spec_from_file(path, payload)
            if spec is None:
                _add_drop(drops, path, reason or "invalid_primary_run")
                continue
            _keep_newest(discovered, drops, (spec.condition_id, spec.seed), spec)

    return _sort_specs(list(discovered.values())), drops


def discover_react_runs(*, runs_root: Path) -> tuple[list[RunSpec], list[dict[str, Any]]]:
    discovered: dict[tuple[str, int | None], RunSpec] = {}
    drops: list[dict[str, Any]] = []
    agent_root = runs_root / "agent" / "runs"

    if not agent_root.exists():
        return [], drops

    for path in sorted(agent_root.rglob("results_react_200.json")):
        payload = _load_json(path)
        spec, reason = _react_spec_from_file(path, payload)
        if spec is None:
            _add_drop(drops, path, reason or "invalid_react_run")
            continue
        _keep_newest(discovered, drops, (spec.condition_id, spec.seed), spec)

    return _sort_specs(list(discovered.values())), drops


def discover_all_runs(*, project_root: Path, runs_root: Path) -> tuple[list[RunSpec], list[dict[str, Any]]]:
    primary_specs, drops = discover_primary_runs(project_root=project_root, runs_root=runs_root)
    react_specs, react_drops = discover_react_runs(runs_root=runs_root)
    return _sort_specs(primary_specs + react_specs), drops + react_drops


def _rows_from_run(spec: RunSpec, payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = payload.get("results") if isinstance(payload.get("results"), list) else payload.get("items")
    if not isinstance(raw_rows, list):
        raise ValueError(f"Missing results list in: {spec.path}")

    rows: list[dict[str, Any]] = []
    for item in raw_rows:
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

    manifest_row = {
        "run_id": spec.run_id,
        "condition_id": spec.condition_id,
        "run_label": spec.run_label,
        "method": spec.method_tag,
        "model_tag": spec.model_tag,
        "k": spec.k,
        "seed": spec.seed,
        "n_items": len(raw_rows),
        "eval_profile": spec.eval_profile,
        "ts_enabled": spec.ts_enabled,
        "source_json": str(spec.path),
    }
    return rows, manifest_row


def build_tables_from_runs(specs: list[RunSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    item_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for spec in specs:
        rows, manifest = _rows_from_run(spec, _load_json(spec.path))
        item_rows.extend(rows)
        manifest_rows.append(manifest)
    return pd.DataFrame(item_rows), pd.DataFrame(manifest_rows)


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


def _shapiro_result(values: Any, n: int) -> tuple[float | None, float | None, str]:
    if n < 3:
        return None, None, "insufficient_n"
    try:
        statistic, p_value = shapiro(values)
    except Exception:
        return None, None, "error"
    decision = "reject_normality" if p_value < 0.05 else "fail_to_reject_normality"
    return float(statistic), float(p_value), decision


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
    if n_pairs < 2:
        return None, None, "insufficient_n"
    test = ttest_rel(valid[right_col], valid[left_col], nan_policy="omit")
    p_value = float(test.pvalue)
    decision = "reject_H0" if p_value < 0.05 else "fail_to_reject_H0"
    return float(test.statistic), p_value, decision


def _bh_fdr_adjust(pvalues: list[float | None]) -> list[float | None]:
    """Benjamini-Hochberg FDR correction for one metric family."""
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


def generate(
    *,
    runs_root: Path = Path("results"),
    per_item_csv: Path = Path("results/analysis/per_item_metrics_primary_raw.csv"),
    out_dir: Path = Path("results/analysis"),
    project_root: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root or Path.cwd()
    runs_root, per_item_csv, out_dir = _resolve_paths(
        runs_root=runs_root,
        per_item_csv=per_item_csv,
        out_dir=out_dir,
        project_root=project_root,
    )

    all_specs, drops = discover_all_runs(project_root=project_root, runs_root=runs_root)
    if not all_specs:
        raise FileNotFoundError(
            f"No valid run files found under: {runs_root}\n"
            "Expected files like: results_k0_seed7.json, results_k3_seed7.json"
        )

    items_df, manifest_df = build_tables_from_runs(all_specs)
    per_item = prepare_per_item_table(items_df)
    condition_ids = set(manifest_df["condition_id"].dropna().astype(str).tolist())
    comparisons = planned_comparisons(condition_ids)

    manifest_out = out_dir / "run_manifest.csv"
    means_out = out_dir / "stats_mean_median_std.csv"
    paired_out = out_dir / "stats_paired_ttests.csv"

    manifest_df.sort_values(["model_tag", "method", "k", "seed", "run_id"]).to_csv(manifest_out, index=False)
    per_item.sort_values(["condition_id", "seed", "example_id"]).to_csv(per_item_csv, index=False)
    compute_mean_median_std(per_item).sort_values(["condition_id", "seed", "metric"]).to_csv(means_out, index=False)
    compute_paired_tests(per_item, comparisons).sort_values(["comparison", "metric"]).to_csv(paired_out, index=False)

    react_specs = [spec for spec in all_specs if spec.method_tag == "react"]
    primary_count = len(list(runs_root.rglob("results_k*_seed*.json")))
    return {
        "design": "multi_model_multi_method_from_results_tree",
        "runs_root": str(runs_root),
        "run_files_found": int(primary_count),
        "react_files_found": int(len(react_specs)),
        "runs_included": int(len(all_specs)),
        "conditions_included": sorted(condition_ids),
        "comparisons_planned": int(len(comparisons)),
        "dropped_files": drops,
        "per_item_csv": str(per_item_csv),
        "outputs": [str(manifest_out), str(means_out), str(paired_out)],
    }


def main() -> None:
    args = parse_args()
    summary = generate(
        runs_root=args.runs_root,
        per_item_csv=args.per_item_csv,
        out_dir=args.out_dir,
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
