"""Run discovery and table building for the research comparison workflow."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


FULL_BENCHMARK_SIZE = 200
PRIMARY_SUPPORTED_K = {0, 3}
SUPPORTED_MODEL_TAGS = {"llama", "qwen"}
PRIMARY_METHODS = {"base", "qlora"}
DEFAULT_PRIMARY_EVAL_PROFILE = "model_only_raw"
SUPPORTED_PRIMARY_EVAL_PROFILES = {"model_only_raw", "optional_reliability_layer"}
REACT_RUN_FILENAMES = ("results_react_200.json", "results_react_eval.json")


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


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _allowed_primary_eval_profiles(primary_eval_profile: str) -> set[str | None]:
    if primary_eval_profile == "model_only_raw":
        return {None, "model_only_raw"}
    if primary_eval_profile == "optional_reliability_layer":
        return {"optional_reliability_layer"}
    raise ValueError(f"Unsupported primary_eval_profile: {primary_eval_profile}")


def _ts_enabled_from_payload(payload: dict[str, Any]) -> bool | None:
    metadata = _as_dict(payload.get("run_metadata"))
    value = metadata.get("ts_enabled")
    return bool(value) if value is not None else None


def _full_run_reason(payload: dict[str, Any]) -> str | None:
    n_items = _int_or_none(payload.get("n"))
    if n_items != FULL_BENCHMARK_SIZE:
        return f"n={n_items}"
    return None


def _primary_spec_from_file(
    path: Path,
    payload: dict[str, Any],
    *,
    allowed_eval_profiles: set[str | None],
) -> tuple[RunSpec | None, str | None]:
    model_tag = _model_tag_from_payload(payload)
    method_tag = _primary_method_from_path(path)
    k, seed = _parse_primary_filename(path)
    eval_profile = _eval_profile(payload)

    if model_tag not in SUPPORTED_MODEL_TAGS:
        return None, "unsupported_or_missing_model_tag"
    if method_tag not in PRIMARY_METHODS:
        return None, "unsupported_or_missing_method_tag"
    if k not in PRIMARY_SUPPORTED_K:
        return None, "unsupported_or_missing_k"
    if seed is None:
        return None, "missing_seed_in_filename"
    if eval_profile not in allowed_eval_profiles:
        return None, f"eval_profile={eval_profile}"

    full_run_reason = _full_run_reason(payload)
    if full_run_reason is not None:
        return None, full_run_reason

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
    metadata = _as_dict(payload.get("run_metadata"))
    seed = _int_or_none(config.get("few_shot_seed"))
    k = _int_or_none(config.get("few_shot_k"))
    quick_limit = _int_or_none(metadata.get("quick_limit"))

    if model_tag not in SUPPORTED_MODEL_TAGS:
        return None, "unsupported_or_missing_model_tag"
    if k not in PRIMARY_SUPPORTED_K:
        return None, "unsupported_k"
    if quick_limit not in {None, FULL_BENCHMARK_SIZE}:
        return None, f"quick_limit={quick_limit}"

    full_run_reason = _full_run_reason(payload)
    if full_run_reason is not None:
        return None, full_run_reason

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


def discover_primary_runs(
    *,
    project_root: Path,
    runs_root: Path,
    primary_eval_profile: str = DEFAULT_PRIMARY_EVAL_PROFILE,
) -> tuple[list[RunSpec], list[dict[str, Any]]]:
    root = runs_root if runs_root.is_absolute() else (project_root / runs_root)
    discovered: dict[tuple[str, int | None], RunSpec] = {}
    drops: list[dict[str, Any]] = []
    allowed_eval_profiles = _allowed_primary_eval_profiles(primary_eval_profile)

    for subdir in ("baseline", "qlora"):
        run_dir = root / subdir / "runs"
        if not run_dir.exists():
            continue
        for path in sorted(run_dir.rglob("results_k*_seed*.json")):
            payload = _load_json(path)
            spec, reason = _primary_spec_from_file(
                path,
                payload,
                allowed_eval_profiles=allowed_eval_profiles,
            )
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

    seen_paths: set[Path] = set()
    for pattern in REACT_RUN_FILENAMES:
        for path in sorted(agent_root.rglob(pattern)):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            payload = _load_json(path)
            spec, reason = _react_spec_from_file(path, payload)
            if spec is None:
                _add_drop(drops, path, reason or "invalid_react_run")
                continue
            _keep_newest(discovered, drops, (spec.condition_id, spec.seed), spec)

    return _sort_specs(list(discovered.values())), drops


def discover_all_runs(
    *,
    project_root: Path,
    runs_root: Path,
    primary_eval_profile: str = DEFAULT_PRIMARY_EVAL_PROFILE,
) -> tuple[list[RunSpec], list[dict[str, Any]]]:
    primary_specs, drops = discover_primary_runs(
        project_root=project_root,
        runs_root=runs_root,
        primary_eval_profile=primary_eval_profile,
    )
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
                "va": item.get("va"),
                "em": item.get("em"),
                "ex": item.get("ex"),
                "ts": item.get("ts"),
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


__all__ = [
    "DEFAULT_PRIMARY_EVAL_PROFILE",
    "FULL_BENCHMARK_SIZE",
    "RunSpec",
    "SUPPORTED_PRIMARY_EVAL_PROFILES",
    "build_tables_from_runs",
    "discover_all_runs",
    "discover_primary_runs",
    "discover_react_runs",
]
