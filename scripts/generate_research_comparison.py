#!/usr/bin/env python3
"""
Research stats generator for dissertation reporting.

Single source of truth:
- reads run JSON files from `results/baseline/runs/**/results_k*_seed*.json`

Supported matrix:
- model tags: llama, qwen
- methods: baseline (base), qlora
- k values used in comparisons: 0 and 3

Outputs:
- results/analysis/per_item_metrics_primary_raw.csv
- results/analysis/run_manifest.csv
- results/analysis/stats_mean_median_shapiro.csv
- results/analysis/stats_paired_shapiro.csv
- results/analysis/stats_paired_ttests.csv

Documentation:
- Shapiro-Wilk: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
- Paired t-test: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
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
from scipy.stats import shapiro, ttest_rel

METRICS = ("va", "em", "ex", "ts")
SUPPORTED_K = {0, 3}
SUPPORTED_MODEL_TAGS = {"llama", "qwen"}
SUPPORTED_METHODS = {"base", "qlora"}

SHAPIRO_REF = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html"
TTEST_REF = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html"

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
    "diff_shapiro_w",
    "diff_shapiro_p",
    "diff_shapiro_decision_alpha_0_05",
    "diff_shapiro_note",
    "diff_shapiro_reference_url",
    "t_stat",
    "p_value",
    "decision_alpha_0_05",
    "reference_url",
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
        default=Path("results/baseline/runs"),
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
    method_part = "Base" if method_tag == "base" else "QLoRA"
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
                    "reference_url": SHAPIRO_REF,
                }
            )
    return pd.DataFrame(rows)


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

            # best-practice assumption check for paired t-test:
            # test normality of paired differences, not raw per-run values.
            if n < 3:
                diff_shapiro_w = None
                diff_shapiro_p = None
                diff_shapiro_decision = "insufficient_n"
                diff_shapiro_note = None
            else:
                diffs = (valid[right_col] - valid[left_col]).to_numpy()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    w_val, p_val = shapiro(diffs)
                diff_shapiro_w = float(w_val)
                diff_shapiro_p = float(p_val)
                diff_shapiro_decision = "reject_normality" if diff_shapiro_p < 0.05 else "fail_to_reject_normality"
                diff_shapiro_note = "; ".join(str(w.message) for w in caught) if caught else None

            if n < 2:
                left_mean = None
                right_mean = None
                diff = None
                t_stat = None
                p_value = None
                decision = "insufficient_n"
            else:
                left_mean = float(valid[left_col].mean())
                right_mean = float(valid[right_col].mean())
                diff = right_mean - left_mean
                test = ttest_rel(valid[right_col], valid[left_col], nan_policy="omit")
                t_stat = float(test.statistic)
                p_value = float(test.pvalue)
                decision = "reject_H0" if p_value < 0.05 else "fail_to_reject_H0"

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
                    "diff_shapiro_w": diff_shapiro_w,
                    "diff_shapiro_p": diff_shapiro_p,
                    "diff_shapiro_decision_alpha_0_05": diff_shapiro_decision,
                    "diff_shapiro_note": diff_shapiro_note,
                    "diff_shapiro_reference_url": SHAPIRO_REF,
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "decision_alpha_0_05": decision,
                    "reference_url": TTEST_REF,
                }
            )

    if not rows:
        return pd.DataFrame(columns=TTEST_COLUMNS)
    return pd.DataFrame(rows, columns=TTEST_COLUMNS)


def generate(
    *,
    runs_root: Path = Path("results/baseline/runs"),
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
    if not specs:
        raise FileNotFoundError(
            f"No valid run files found under: {runs_root}\n"
            "Expected files like: results_k0_seed7.json, results_k3_seed7.json"
        )

    items_df, manifest_df = build_tables_from_runs(specs)
    per_item = _prepare_per_item(items_df)

    condition_ids = set(manifest_df["condition_id"].dropna().astype(str).tolist())
    comparisons = build_planned_comparisons(condition_ids)

    manifest_out = out_dir / "run_manifest.csv"
    shapiro_out = out_dir / "stats_mean_median_shapiro.csv"
    paired_shapiro_out = out_dir / "stats_paired_shapiro.csv"
    ttests_out = out_dir / "stats_paired_ttests.csv"

    paired_tests = compute_paired_ttests(per_item, comparisons).sort_values(["comparison", "metric"])
    paired_shapiro_cols = [
        "comparison",
        "left_condition_id",
        "right_condition_id",
        "metric",
        "pair_key",
        "matched_seeds",
        "n_pairs",
        "diff_shapiro_w",
        "diff_shapiro_p",
        "diff_shapiro_decision_alpha_0_05",
        "diff_shapiro_note",
        "diff_shapiro_reference_url",
    ]

    manifest_df.sort_values(["model_tag", "method", "k", "seed", "run_id"]).to_csv(manifest_out, index=False)
    per_item.sort_values(["condition_id", "seed", "example_id"]).to_csv(per_item_csv, index=False)
    compute_mean_median_shapiro(per_item).sort_values(["condition_id", "seed", "metric"]).to_csv(shapiro_out, index=False)
    paired_tests[paired_shapiro_cols].to_csv(paired_shapiro_out, index=False)
    paired_tests.to_csv(ttests_out, index=False)

    return {
        "design": "multi_model_multi_method_from_baseline_runs",
        "runs_root": str(runs_root),
        "run_files_found": int(len(list(runs_root.rglob('results_k*_seed*.json')))),
        "runs_included": int(len(specs)),
        "conditions_included": sorted(condition_ids),
        "comparisons_planned": int(len(comparisons)),
        "dropped_files": drops,
        "per_item_csv": str(per_item_csv),
        "outputs": [str(manifest_out), str(shapiro_out), str(paired_shapiro_out), str(ttests_out)],
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
