"""Manual final-pack loader for the official dissertation analysis workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


FULL_BENCHMARK_SIZE = 200
PACK_DIR = Path("results/final_pack")
_PRIMARY_PATTERN = re.compile(r"^(llama|qwen)_(base|qlora)_k(0|3)_seed(\d+)\.json$")
_REACT_PATTERN = re.compile(r"^(llama|qwen)_react_k(0|3)_seed(\d+)\.json$")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name}: expected a JSON object at the top level")
    return payload


def _item_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("results")
    if isinstance(rows, list):
        return rows
    rows = payload.get("items")
    if isinstance(rows, list):
        return rows
    raise ValueError("missing `results` or `items` list")


def _parse_filename(path: Path) -> tuple[str, str, int, int]:
    primary = _PRIMARY_PATTERN.fullmatch(path.name)
    if primary:
        model_tag, method, k_text, seed_text = primary.groups()
        return model_tag, method, int(k_text), int(seed_text)

    react = _REACT_PATTERN.fullmatch(path.name)
    if react:
        model_tag, k_text, seed_text = react.groups()
        return model_tag, "react", int(k_text), int(seed_text)

    raise ValueError(
        f"{path.name}: unsupported filename. Expected names like "
        "`llama_base_k0_seed7.json`, `llama_base_k3_seed37.json`, or `qwen_react_k3_seed7.json`."
    )


def _validate_payload(path: Path, payload: dict[str, Any], method: str, k: int, seed: int) -> list[dict[str, Any]]:
    rows = _item_rows(payload)
    n_items = int(payload.get("n", len(rows)))
    if n_items != FULL_BENCHMARK_SIZE:
        raise ValueError(f"{path.name}: expected n=200, found n={n_items}")
    if len(rows) != FULL_BENCHMARK_SIZE:
        raise ValueError(f"{path.name}: expected 200 result rows, found {len(rows)}")

    if method != "react":
        eval_profile = payload.get("eval_profile")
        if eval_profile not in {None, "model_only_raw"}:
            raise ValueError(
                f"{path.name}: unsupported eval_profile={eval_profile!r}. "
                "Only raw-model runs belong in the final pack."
            )

    payload_k = payload.get("k")
    if payload_k is not None and int(payload_k) != k:
        raise ValueError(f"{path.name}: filename says k={k}, payload says k={payload_k}")
    payload_seed = payload.get("seed")
    if payload_seed is not None and int(payload_seed) != seed:
        raise ValueError(f"{path.name}: filename says seed={seed}, payload says seed={payload_seed}")
    return rows


def build_tables_from_pack(pack_dir: Path = PACK_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the manual final pack and build the official manifest and per-item tables."""
    resolved = Path(pack_dir)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing final pack folder: {resolved}")

    manifest_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    json_paths = sorted(resolved.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(
            f"No JSON files found in {resolved}. Copy the official run files into this folder first."
        )

    for path in json_paths:
        model_tag, method, k, seed = _parse_filename(path)
        payload = _load_json(path)
        rows = _validate_payload(path, payload, method, k, seed)
        condition_id = f"{model_tag}_{method}_k{k}"
        ts_enabled = any(isinstance(item, dict) and item.get("ts") is not None for item in rows)

        manifest_rows.append(
            {
                "condition_id": condition_id,
                "model_tag": model_tag,
                "method": method,
                "k": k,
                "seed": seed,
                "n_items": FULL_BENCHMARK_SIZE,
                "ts_enabled": ts_enabled,
                "source_json": path.name,
            }
        )

        for item in rows:
            if not isinstance(item, dict):
                raise ValueError(f"{path.name}: every result row must be a JSON object")
            item_rows.append(
                {
                    "condition_id": condition_id,
                    "model_tag": model_tag,
                    "method": method,
                    "k": k,
                    "seed": seed,
                    "example_id": item.get("i", item.get("example_id")),
                    "nlq": item.get("nlq", ""),
                    "va": item.get("va"),
                    "em": item.get("em"),
                    "ex": item.get("ex"),
                    "ts": item.get("ts"),
                    "source_json": path.name,
                }
            )

    return pd.DataFrame(item_rows), pd.DataFrame(manifest_rows)


__all__ = ["build_tables_from_pack"]
