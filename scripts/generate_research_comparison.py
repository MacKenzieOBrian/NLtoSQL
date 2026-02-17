#!/usr/bin/env python3
"""
Statistical reporting:
- Per-run uncertainty: Wilson 95% interval per metric rate.
- Paired comparisons: right_rate - left_rate on identical NLQs.
- Paired significance: exact McNemar from discordant pairs.

Pipeline overview:
1) discover and load run JSON files into one per-item table,
2) compute per-run VA/EM/EX/TS with Wilson intervals,
3) compute paired deltas and McNemar significance for fixed comparisons,
4) build failure taxonomy and summary plots,
5) write analysis artifacts to `results/analysis/`.

References (project anchors):
- Evaluation focus (execution semantics): `REFERENCES.md#ref-yu2018-spider`,
  `REFERENCES.md#ref-zhong2020-ts`
- Significance testing practice: `REFERENCES.md#ref-dror2018-significance`
- Interval/test formulas: `REFERENCES.md#ref-wilson1927`,
  `REFERENCES.md#ref-mcnemar1947`
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nl2sql.research_stats import mcnemar_exact_p, wilson_interval


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    label: str
    method_family: str
    model_variant: str
    prompting: str
    k: int | None
    run_type: str
    path_candidates: tuple[str, ...]


def _to_flag(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value != 0)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes"}:
            return 1
        if low in {"0", "false", "no"}:
            return 0
    return None


def _find_latest_agent_json(project_root: Path) -> str | None:
    agent_dir = project_root / "results" / "agent"
    preferred = agent_dir / "react_infra_n20_v9.json"
    if preferred.exists():
        return str(preferred.relative_to(project_root))

    candidates = sorted(
        list(agent_dir.glob("react_infra*.json"))
        + list(agent_dir.glob("results_react_200*.json"))
    )
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest.relative_to(project_root))


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "model"


def _discover_model_family_specs(project_root: Path) -> list[RunSpec]:
    model_family_dir = project_root / "results" / "baseline" / "model_family"
    if not model_family_dir.exists():
        return []

    specs: list[RunSpec] = []
    seen_run_ids: set[str] = set()
    for path in sorted(model_family_dir.glob("*_k*.json")):
        match = re.match(r"(?P<alias>.+)_k(?P<k>\d+)\.json$", path.name)
        if not match:
            continue

        alias = match.group("alias")
        alias_slug = _slugify(alias)
        k_value = int(match.group("k"))
        run_id = f"modelfam_{alias_slug}_k{k_value}"
        if run_id in seen_run_ids:
            continue
        seen_run_ids.add(run_id)

        rel = str(path.relative_to(project_root))
        specs.append(
            RunSpec(
                run_id=run_id,
                label=f"{alias.replace('_', '-')} | k={k_value}",
                method_family="model_family",
                model_variant=alias_slug,
                prompting=f"k={k_value}",
                k=k_value,
                run_type="direct",
                path_candidates=(rel,),
            )
        )
    return specs


def _discover_qlora_model_family_specs(project_root: Path) -> list[RunSpec]:
    model_family_dir = project_root / "results" / "qlora" / "model_family"
    if not model_family_dir.exists():
        return []

    specs: list[RunSpec] = []
    seen_run_ids: set[str] = set()
    for path in sorted(model_family_dir.glob("*_qlora_k*.json")):
        match = re.match(r"(?P<alias>.+)_qlora_k(?P<k>\d+)\.json$", path.name)
        if not match:
            continue

        alias = match.group("alias")
        alias_slug = _slugify(alias)
        k_value = int(match.group("k"))
        run_id = f"qlorafam_{alias_slug}_k{k_value}"
        if run_id in seen_run_ids:
            continue
        seen_run_ids.add(run_id)

        rel = str(path.relative_to(project_root))
        specs.append(
            RunSpec(
                run_id=run_id,
                label=f"{alias.replace('_', '-')} | QLoRA | k={k_value}",
                method_family="qlora_model_family",
                model_variant=alias_slug,
                prompting=f"k={k_value}",
                k=k_value,
                run_type="direct",
                path_candidates=(rel,),
            )
        )
    return specs


def default_specs(project_root: Path) -> list[RunSpec]:
    latest_agent = _find_latest_agent_json(project_root)
    agent_candidates: tuple[str, ...] = tuple(
        p for p in (latest_agent, "results/agent/results_react_200.json") if p is not None
    )
    specs = [
        RunSpec(
            run_id="baseline_k0",
            label="Base | k=0",
            method_family="baseline",
            model_variant="base",
            prompting="k=0",
            k=0,
            run_type="direct",
            path_candidates=(
                "results/baseline/baseline_k0.json",
                "results/baseline/results_zero_shot_200.json",
                "results/results_zero_shot_200.json",
            ),
        ),
        RunSpec(
            run_id="baseline_k3",
            label="Base | k=3",
            method_family="baseline",
            model_variant="base",
            prompting="k=3",
            k=3,
            run_type="direct",
            path_candidates=(
                "results/baseline/baseline_k3.json",
                "results/baseline/results_few_shot_k3_200.json",
                "results/results_few_shot_k3_200.json",
            ),
        ),
        RunSpec(
            run_id="qlora_k0",
            label="QLoRA | k=0",
            method_family="qlora",
            model_variant="qlora",
            prompting="k=0",
            k=0,
            run_type="direct",
            path_candidates=(
                "results/qlora/llama3_8b_instruct_qlora_k0.json",
                "results/qlora/qlora_k0.json",
                "results/qlora/results_zero_shot_200.json",
            ),
        ),
        RunSpec(
            run_id="qlora_k3",
            label="QLoRA | k=3",
            method_family="qlora",
            model_variant="qlora",
            prompting="k=3",
            k=3,
            run_type="direct",
            path_candidates=(
                "results/qlora/llama3_8b_instruct_qlora_k3.json",
                "results/qlora/qlora_k3.json",
                "results/qlora/results_few_shot_k3_200.json",
            ),
        ),
        RunSpec(
            run_id="react_exec",
            label="ReAct infra",
            method_family="agent_infra",
            model_variant="unspecified",
            prompting="n/a",
            k=None,
            run_type="agent",
            path_candidates=agent_candidates,
        ),
    ]
    specs.extend(_discover_model_family_specs(project_root))
    specs.extend(_discover_qlora_model_family_specs(project_root))
    return specs


def _resolve_existing_path(project_root: Path, candidates: tuple[str, ...]) -> Path | None:
    for rel in candidates:
        p = project_root / rel
        if p.exists():
            return p
    return None


def _normalize_sql(sql: str | None) -> str:
    if not sql:
        return ""
    return re.sub(r"\s+", " ", sql.strip().lower()).rstrip(";")


def _extract_tables(sql: str) -> set[str]:
    return {m.group(1) for m in re.finditer(r"(?:from|join)\s+([a-zA-Z_][\w]*)", sql)}


def _extract_agg_funcs(sql: str) -> set[str]:
    return {m.group(1) for m in re.finditer(r"\b(count|sum|avg|min|max)\s*\(", sql)}


def _extract_where_literals(sql: str) -> set[str]:
    match = re.search(r"\bwhere\b(.*?)(\bgroup\b|\border\b|\blimit\b|$)", sql, re.DOTALL)
    if not match:
        return set()
    where_clause = match.group(1)
    quoted = [m[0] or m[1] for m in re.findall(r"'([^']*)'|\"([^\"]*)\"", where_clause)]
    numerics = re.findall(r"\b\d+(?:\.\d+)?\b", where_clause)
    return {s.strip().lower() for s in quoted + numerics if s.strip()}


def _split_select_items(select_clause: str) -> list[str]:
    out: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(select_clause):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        elif ch == "," and depth == 0:
            out.append(select_clause[start:i])
            start = i + 1
    out.append(select_clause[start:])
    return out


def _extract_projection_signature(sql: str) -> set[str]:
    match = re.search(r"\bselect\b(.*?)\bfrom\b", sql, re.DOTALL)
    if not match:
        return set()
    clause = match.group(1)
    cleaned = []
    for item in _split_select_items(clause):
        token = item.strip()
        token = re.sub(r"\s+as\s+[a-zA-Z_][\w]*$", "", token)
        token = re.sub(r"\b[a-zA-Z_][\w]*\.", "", token)
        token = re.sub(r"\s+", " ", token).strip()
        if token:
            cleaned.append(token)
    return set(cleaned)


def classify_failure(row: pd.Series) -> str:
    va_value = _to_flag(row.get("va"))
    if va_value != 1:
        return "invalid_sql"

    pred = _normalize_sql(row.get("pred_sql"))
    gold = _normalize_sql(row.get("gold_sql"))
    if not pred:
        return "invalid_sql"

    err = str(row.get("error") or "").lower()
    pred_tables = _extract_tables(pred)
    gold_tables = _extract_tables(gold)
    pred_aggs = _extract_agg_funcs(pred)
    gold_aggs = _extract_agg_funcs(gold)
    pred_has_group = " group by " in f" {pred} "
    gold_has_group = " group by " in f" {gold} "
    pred_literals = _extract_where_literals(pred)
    gold_literals = _extract_where_literals(gold)
    pred_proj = _extract_projection_signature(pred)
    gold_proj = _extract_projection_signature(gold)

    uses_join_context = (
        " join " in f" {pred} "
        or " join " in f" {gold} "
        or len(pred_tables | gold_tables) > 1
    )
    if pred_tables != gold_tables and uses_join_context:
        return "join_path"
    if pred_aggs != gold_aggs or pred_has_group != gold_has_group:
        return "aggregation"
    if pred_literals != gold_literals and (pred_literals or gold_literals):
        return "value_linking"
    if pred_proj != gold_proj or "column mismatch" in err:
        return "projection"

    pred_has_order = " order by " in f" {pred} "
    gold_has_order = " order by " in f" {gold} "
    pred_has_limit = " limit " in f" {pred} "
    gold_has_limit = " limit " in f" {gold} "
    if pred_has_order != gold_has_order or pred_has_limit != gold_has_limit:
        return "ordering_limit"

    return "other_semantic"


def _load_one_run(project_root: Path, spec: RunSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidate_str = " | ".join(spec.path_candidates)
    path = _resolve_existing_path(project_root, spec.path_candidates)
    if path is None:
        manifest = {
            "run_id": spec.run_id,
            "label": spec.label,
            "status": "missing",
            "resolved_path": None,
            "path_candidates": candidate_str,
            "method_family": spec.method_family,
            "model_variant": spec.model_variant,
            "prompting": spec.prompting,
            "k": spec.k,
            "run_type": spec.run_type,
        }
        return pd.DataFrame(), manifest

    payload = json.loads(path.read_text(encoding="utf-8"))
    item_key = "results" if "results" in payload else "items"
    raw_items = payload[item_key]
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        rows.append(
            {
                "run_id": spec.run_id,
                "run_label": spec.label,
                "method_family": spec.method_family,
                "model_variant": spec.model_variant,
                "prompting": spec.prompting,
                "k": spec.k,
                "run_type": spec.run_type,
                "source_path": str(path.relative_to(project_root)),
                "timestamp": payload.get("timestamp"),
                "model_id": (payload.get("run_metadata") or {}).get("model_id"),
                "notebook": (payload.get("run_metadata") or {}).get("notebook"),
                "example_id": item.get("i", idx),
                "nlq": item.get("nlq"),
                "gold_sql": item.get("gold_sql"),
                "pred_sql": item.get("pred_sql"),
                "va": _to_flag(item.get("va")),
                "em": _to_flag(item.get("em")),
                "ex": _to_flag(item.get("ex")),
                "ts": _to_flag(item.get("ts")),
                "error": item.get("error") or item.get("pred_err"),
            }
        )

    df = pd.DataFrame(rows)
    for col in ("va", "em", "ex", "ts"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")

    manifest = {
        "run_id": spec.run_id,
        "label": spec.label,
        "status": "loaded",
        "resolved_path": str(path.relative_to(project_root)),
        "path_candidates": candidate_str,
        "method_family": spec.method_family,
        "model_variant": spec.model_variant,
        "prompting": spec.prompting,
        "k": spec.k,
        "run_type": spec.run_type,
        "n_items": len(rows),
    }
    return df, manifest


def load_runs(project_root: Path, specs: list[RunSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    manifest_rows = []
    for spec in specs:
        df, manifest = _load_one_run(project_root, spec)
        if not df.empty:
            all_rows.append(df)
        manifest_rows.append(manifest)
    per_item = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    manifest_df = pd.DataFrame(manifest_rows)
    return per_item, manifest_df


def build_overall_metrics(per_item: pd.DataFrame) -> pd.DataFrame:
    if per_item.empty:
        return pd.DataFrame()

    metric_cols = ["va", "em", "ex", "ts"]
    rows = []
    grouped = per_item.groupby(
        ["run_id", "run_label", "method_family", "model_variant", "prompting", "k", "run_type"],
        dropna=False,
        sort=False,
    )
    for keys, df in grouped:
        run_id, run_label, method_family, model_variant, prompting, k, run_type = keys
        base = {
            "run_id": run_id,
            "run_label": run_label,
            "method_family": method_family,
            "model_variant": model_variant,
            "prompting": prompting,
            "k": k,
            "run_type": run_type,
            "n": len(df),
        }
        for metric in metric_cols:
            valid = df[metric].dropna()
            if valid.empty:
                continue
            successes = int(valid.sum())
            n = int(valid.count())
            rate = successes / n
            # Wilson interval gives stable CI coverage for binary rates.
            # Ref: REFERENCES.md#ref-wilson1927
            lo, hi = wilson_interval(successes, n)
            row = dict(base)
            row.update(
                {
                    "metric": metric,
                    "successes": successes,
                    "n_metric": n,
                    "rate": rate,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
            rows.append(row)

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics
    metrics["rate_pct"] = metrics["rate"] * 100.0
    metrics["ci_low_pct"] = metrics["ci_low"] * 100.0
    metrics["ci_high_pct"] = metrics["ci_high"] * 100.0
    return metrics


def build_wide_metric_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    pivot = metrics.pivot(
        index=["run_id", "run_label", "method_family", "model_variant", "prompting", "k", "run_type", "n"],
        columns="metric",
        values="rate",
    ).reset_index()
    pivot.columns.name = None
    for metric in ("va", "em", "ex", "ts"):
        if metric in pivot.columns:
            pivot[f"{metric}_pct"] = pivot[metric] * 100.0
    return pivot


def build_failure_taxonomy(per_item: pd.DataFrame) -> pd.DataFrame:
    if per_item.empty:
        return pd.DataFrame()

    failed = per_item[(per_item["ex"] == 0) | (per_item["va"] == 0)].copy()
    if failed.empty:
        return pd.DataFrame()

    failed["failure_type"] = failed.apply(classify_failure, axis=1)
    counts = (
        failed.groupby(
            ["run_id", "run_label", "method_family", "model_variant", "prompting", "failure_type"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    totals_fail = failed.groupby("run_id").size().to_dict()
    totals_all = per_item.groupby("run_id").size().to_dict()
    counts["share_of_failures"] = counts.apply(
        lambda r: r["count"] / max(totals_fail.get(r["run_id"], 1), 1), axis=1
    )
    counts["share_of_all"] = counts.apply(
        lambda r: r["count"] / max(totals_all.get(r["run_id"], 1), 1), axis=1
    )
    return counts.sort_values(["run_label", "count"], ascending=[True, False]).reset_index(drop=True)


def _paired_metric_rows(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_id: str,
    right_id: str,
    comparison_id: str,
    comparison_label: str,
) -> list[dict[str, Any]]:
    merged = left.merge(
        right,
        on="nlq",
        how="inner",
        suffixes=("_left", "_right"),
    )
    if merged.empty:
        return []

    rows: list[dict[str, Any]] = []
    for metric in ("va", "em", "ex", "ts"):
        col_l = f"{metric}_left"
        col_r = f"{metric}_right"
        if col_l not in merged or col_r not in merged:
            continue
        valid = merged[[col_l, col_r]].dropna()
        if valid.empty:
            continue
        # Paired delta on identical NLQs (controls per-example difficulty).
        # Guidance ref: REFERENCES.md#ref-dror2018-significance
        left_rate = float(valid[col_l].mean())
        right_rate = float(valid[col_r].mean())
        delta = right_rate - left_rate
        # Exact McNemar uses only discordant outcomes.
        # Ref: REFERENCES.md#ref-mcnemar1947
        improve = int((valid[col_r] > valid[col_l]).sum())
        degrade = int((valid[col_r] < valid[col_l]).sum())
        ties = int((valid[col_r] == valid[col_l]).sum())
        p_value = mcnemar_exact_p(improve, degrade)
        rows.append(
            {
                "comparison_id": comparison_id,
                "comparison_label": comparison_label,
                "left_run_id": left_id,
                "right_run_id": right_id,
                "metric": metric,
                "n_overlap": int(valid.shape[0]),
                "left_rate": left_rate,
                "right_rate": right_rate,
                "delta": delta,
                "delta_pct": delta * 100.0,
                "improved_count": improve,
                "degraded_count": degrade,
                "tied_count": ties,
                "discordant_count": improve + degrade,
                "mcnemar_p": p_value,
                "significant_0_05": int(p_value < 0.05),
            }
        )
    return rows


def build_paired_deltas(per_item: pd.DataFrame) -> pd.DataFrame:
    if per_item.empty:
        return pd.DataFrame()
    rows = []
    by_run = {run_id: df.copy() for run_id, df in per_item.groupby("run_id")}
    # Controlled, pre-registered style comparisons used for dissertation claims.
    pair_defs = [
        ("baseline_k0", "baseline_k3", "few_shot_gain_base", "Few-shot gain (Base: k=0 -> k=3)"),
        ("qlora_k0", "qlora_k3", "few_shot_gain_qlora", "Few-shot gain (QLoRA: k=0 -> k=3)"),
        ("baseline_k0", "qlora_k0", "qlora_gain_k0", "Fine-tune gain (k=0: Base -> QLoRA)"),
        ("baseline_k3", "qlora_k3", "qlora_gain_k3", "Fine-tune gain (k=3: Base -> QLoRA)"),
        ("baseline_k3", "react_exec", "react_vs_base_k3", "Execution infra effect (Base k=3 -> ReAct)"),
    ]
    for left_id, right_id, comp_id, comp_label in pair_defs:
        if left_id not in by_run or right_id not in by_run:
            continue
        rows.extend(
            _paired_metric_rows(
                by_run[left_id],
                by_run[right_id],
                left_id=left_id,
                right_id=right_id,
                comparison_id=comp_id,
                comparison_label=comp_label,
            )
        )

    model_family = per_item[per_item["method_family"] == "model_family"][
        ["run_id", "model_variant", "k"]
    ].drop_duplicates()
    if not model_family.empty:
        model_family["k"] = pd.to_numeric(model_family["k"], errors="coerce")
        for model_variant, group in model_family.groupby("model_variant", dropna=False):
            group = group.dropna(subset=["k"])
            if group.empty:
                continue

            k0 = group[group["k"] == 0]["run_id"].drop_duplicates().head(1).tolist()
            k3 = group[group["k"] == 3]["run_id"].drop_duplicates().head(1).tolist()
            if not k0 or not k3:
                continue

            left_id = k0[0]
            right_id = k3[0]
            if left_id not in by_run or right_id not in by_run:
                continue

            model_slug = _slugify(str(model_variant))
            model_label = str(model_variant).replace("_", "-")
            rows.extend(
                _paired_metric_rows(
                    by_run[left_id],
                    by_run[right_id],
                    left_id=left_id,
                    right_id=right_id,
                    comparison_id=f"few_shot_gain_{model_slug}",
                    comparison_label=f"Few-shot gain ({model_label}: k=0 -> k=3)",
                )
            )

            if "baseline_k3" in by_run:
                rows.extend(
                    _paired_metric_rows(
                        by_run["baseline_k3"],
                        by_run[right_id],
                        left_id="baseline_k3",
                        right_id=right_id,
                        comparison_id=f"k3_model_family_{model_slug}",
                        comparison_label=f"Model-family contrast (Base k=3 -> {model_label} k=3)",
                    )
                )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _write_markdown_summary(
    out_path: Path,
    manifest: pd.DataFrame,
    wide_metrics: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    def _pipe_table(df: pd.DataFrame, floatfmt: str = ".2f") -> list[str]:
        if df.empty:
            return ["(empty)"]
        headers = list(df.columns)
        rows = [headers]
        for values in df.itertuples(index=False):
            row = []
            for v in values:
                if isinstance(v, float):
                    if math.isnan(v):
                        cell = ""
                    else:
                        cell = format(v, floatfmt)
                else:
                    cell = "" if v is None else str(v)
                row.append(cell.replace("|", "\\|"))
            rows.append(row)
        sep = ["---"] * len(headers)
        lines = [
            "| " + " | ".join(rows[0]) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for row in rows[1:]:
            lines.append("| " + " | ".join(row) + " |")
        return lines

    lines: list[str] = []
    lines.append("# Research Comparison Summary")
    lines.append("")
    lines.append("## Loaded Runs")
    for row in manifest.itertuples(index=False):
        if row.status == "loaded":
            lines.append(f"- {row.label}: loaded from `{row.resolved_path}` (n={row.n_items})")
        else:
            lines.append(f"- {row.label}: missing (checked `{row.path_candidates}`)")
    lines.append("")

    if wide_metrics.empty:
        lines.append("No metrics available.")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("## Overall Metrics")
    ordered = wide_metrics.copy()
    cols = ["run_label", "n"] + [c for c in ["va_pct", "em_pct", "ex_pct", "ts_pct"] if c in ordered.columns]
    lines.extend(_pipe_table(ordered[cols], floatfmt=".2f"))
    lines.append("")

    if not paired.empty:
        lines.append("## Paired Delta Highlights")
        keep = paired[paired["metric"].isin(["va", "em", "ex", "ts"])].copy()
        keep["delta_pct"] = keep["delta_pct"].map(lambda x: f"{x:+.2f}")
        keep["mcnemar_p"] = keep["mcnemar_p"].map(lambda x: f"{x:.4f}")
        pair_table = keep[["comparison_label", "metric", "n_overlap", "left_rate", "right_rate", "delta_pct"]].assign(
            left_rate=lambda d: (d["left_rate"] * 100.0),
            right_rate=lambda d: (d["right_rate"] * 100.0),
        )
        pair_table["mcnemar_p"] = keep["mcnemar_p"]
        pair_table["sig_0_05"] = keep["significant_0_05"].astype(int)
        lines.extend(_pipe_table(pair_table, floatfmt=".2f"))
        lines.append("")

    lines.append("## Notes")
    lines.append("- Agent run is treated as execution infrastructure, not the primary contribution.")
    lines.append("- Primary claims should use base/QLoRA and k=0/k=3 controlled comparisons.")
    loaded_model_family = manifest[
        (manifest["status"] == "loaded") & (manifest["method_family"] == "model_family")
    ]
    if not loaded_model_family.empty:
        lines.append("- Model-family runs were auto-loaded from `results/baseline/model_family/`.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _ensure_plots_available() -> Any | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def save_figures(
    out_dir: Path,
    metrics: pd.DataFrame,
    failure_taxonomy: pd.DataFrame,
    paired: pd.DataFrame,
) -> list[Path]:
    plt = _ensure_plots_available()
    if plt is None:
        return []
    figure_paths: list[Path] = []

    if not metrics.empty:
        plot_df = metrics[metrics["metric"].isin(["va", "em", "ex", "ts"])].copy()
        if not plot_df.empty:
            pivot = plot_df.pivot(index="run_label", columns="metric", values="rate_pct").fillna(0.0)
            ax = pivot.plot(kind="bar", figsize=(10, 5), rot=20)
            ax.set_ylabel("Rate (%)")
            ax.set_xlabel("Run")
            ax.set_title("Overall Metrics by Run")
            ax.legend(title="Metric", loc="upper left", bbox_to_anchor=(1.01, 1.0))
            ax.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            out = out_dir / "figure_overall_metrics.png"
            plt.savefig(out, dpi=160)
            plt.close()
            figure_paths.append(out)

    if not paired.empty:
        delta_df = paired[
            paired["comparison_id"].astype(str).str.startswith("few_shot_gain_")
            & paired["metric"].isin(["va", "em", "ex"])
        ].copy()
        if not delta_df.empty:
            pivot = delta_df.pivot(index="comparison_label", columns="metric", values="delta_pct").fillna(0.0)
            ax = pivot.plot(kind="bar", figsize=(10, 4), rot=15)
            ax.set_ylabel("Delta (percentage points)")
            ax.set_xlabel("Comparison")
            ax.set_title("Few-shot Gain (k=0 -> k=3)")
            ax.axhline(0.0, color="black", linewidth=1)
            ax.legend(title="Metric", loc="upper left", bbox_to_anchor=(1.01, 1.0))
            ax.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            out = out_dir / "figure_few_shot_deltas.png"
            plt.savefig(out, dpi=160)
            plt.close()
            figure_paths.append(out)

    if not failure_taxonomy.empty:
        pivot = (
            failure_taxonomy.pivot(index="run_label", columns="failure_type", values="share_of_failures")
            .fillna(0.0)
            .sort_index()
        )
        ax = pivot.plot(kind="bar", stacked=True, figsize=(10, 5), rot=20)
        ax.set_ylabel("Share of failed examples")
        ax.set_xlabel("Run")
        ax.set_title("Failure Taxonomy by Run")
        ax.legend(title="Failure type", loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        out = out_dir / "figure_failure_taxonomy.png"
        plt.savefig(out, dpi=160)
        plt.close()
        figure_paths.append(out)

    return figure_paths


def generate(out_dir: Path, project_root: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = default_specs(project_root)
    per_item, manifest = load_runs(project_root, specs)
    metrics = build_overall_metrics(per_item)
    wide_metrics = build_wide_metric_table(metrics)
    failure_taxonomy = build_failure_taxonomy(per_item)
    paired = build_paired_deltas(per_item)

    manifest.to_csv(out_dir / "run_manifest.csv", index=False)
    if not per_item.empty:
        per_item.to_csv(out_dir / "per_item_metrics.csv", index=False)
    if not metrics.empty:
        metrics.to_csv(out_dir / "overall_metrics_long.csv", index=False)
    if not wide_metrics.empty:
        wide_metrics.to_csv(out_dir / "overall_metrics_wide.csv", index=False)
    if not failure_taxonomy.empty:
        failure_taxonomy.to_csv(out_dir / "failure_taxonomy.csv", index=False)
    if not paired.empty:
        paired.to_csv(out_dir / "paired_deltas.csv", index=False)

    _write_markdown_summary(out_dir / "summary.md", manifest, wide_metrics, paired)
    figure_paths = save_figures(out_dir, metrics, failure_taxonomy, paired)

    return {
        "n_runs_loaded": int((manifest["status"] == "loaded").sum()),
        "n_rows": int(len(per_item)),
        "out_dir": str(out_dir),
        "figures": [str(p) for p in figure_paths],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Repository root that contains results/ and notebooks/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/analysis"),
        help="Output folder for CSV/PNG/summary artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate(out_dir=args.out_dir, project_root=args.project_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
