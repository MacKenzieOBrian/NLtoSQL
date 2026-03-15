"""Small helpers for notebook auth, paths, and dataset loading."""

from __future__ import annotations

import json
import os
import sys
from getpass import getpass
from pathlib import Path
from typing import Any


HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")


def _read_json(path: str | Path) -> Any:
    # json read from stdlib docs
    # https://docs.python.org/3/library/json.html#json.loads
    # https://docs.python.org/3/library/pathlib.html#pathlib.Path.read_text
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    # jsonl read pattern from docs
    # https://docs.python.org/3/library/pathlib.html#pathlib.Path.read_text
    # https://docs.python.org/3/library/json.html
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def add_project_root_to_path(start: str | Path | None = None) -> Path:
    """Add the nearest project root to sys.path and return it."""
    # project root walk and sys.path add
    # https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.parents
    # https://docs.python.org/3/library/sys.html#sys.path
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "nl2sql").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    if str(current) not in sys.path:
        sys.path.insert(0, str(current))
    return current


def authenticate_colab_gcp() -> bool:
    """Authenticate with GCP in Colab. Return False outside Colab."""
    # colab auth from docs
    # https://cloud.google.com/colab/docs/authentication
    try:
        from google.colab import auth
    except ModuleNotFoundError:
        print("Not running in Colab; ensure ADC or service-account auth is configured.")
        return False

    auth.authenticate_user()
    return True


def ensure_hf_token(
    *,
    prompt_if_missing: bool = False,
    login_if_missing: bool = True,
) -> str | None:
    """Return a Hugging Face token from env, login flow, or prompt."""
    # hf notebook login from docs
    # https://huggingface.co/docs/huggingface_hub/main/en/package_reference/authentication#huggingface_hub.notebook_login
    token = next((os.getenv(name) for name in HF_TOKEN_ENV_VARS if os.getenv(name)), None)
    if token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        print("Using HF token from env")
        return token

    if login_if_missing:
        try:
            from huggingface_hub import notebook_login

            notebook_login()
        except Exception as exc:
            print("HF auth not configured:", exc)
        else:
            token = next((os.getenv(name) for name in HF_TOKEN_ENV_VARS if os.getenv(name)), None)
            if token:
                os.environ["HUGGINGFACE_HUB_TOKEN"] = token
                return token
            return None

    if prompt_if_missing:
        token = getpass("Enter HF_TOKEN: ").strip()
        if token:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = token
            return token
    return None


def authenticate_notebook_services(*, gcp: bool = False, prompt_for_hf: bool = False) -> str | None:
    """Run the usual notebook auth steps and return the HF token if one is available."""
    # wrapper for colab and hf auth helpers
    # https://cloud.google.com/colab/docs/authentication
    # https://huggingface.co/docs/huggingface_hub/main/en/package_reference/authentication#huggingface_hub.notebook_login
    if gcp:
        authenticate_colab_gcp()
    return ensure_hf_token(prompt_if_missing=prompt_for_hf, login_if_missing=True)


def load_test_set(path: str | Path = "data/classicmodels_test_200.json") -> list[dict[str, Any]]:
    """Load the benchmark test set."""
    # quick json list shape check
    # https://docs.python.org/3/library/json.html
    rows = _read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return rows


def load_train_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL training rows and require nlq/sql keys."""
    # file exists and key checks
    # https://docs.python.org/3/library/pathlib.html#pathlib.Path.exists
    # https://docs.python.org/3/library/stdtypes.html#mapping-types-dict
    train_path = Path(path)
    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing training set at {train_path}. Expected JSONL rows with keys: nlq, sql."
        )

    rows = _read_jsonl(train_path)
    for idx, row in enumerate(rows[:5]):
        if "nlq" not in row or "sql" not in row:
            raise ValueError(f"Missing keys at row {idx}: {row}")
    return rows


def load_test_and_train_sets(
    *,
    test_path: str | Path,
    train_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the benchmark test split and the JSONL training set."""
    # simple tuple return pattern
    # https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences
    return load_test_set(test_path), load_train_records(train_path)


__all__ = [
    "add_project_root_to_path",
    "authenticate_colab_gcp",
    "authenticate_notebook_services",
    "ensure_hf_token",
    "load_test_and_train_sets",
    "load_test_set",
    "load_train_records",
]
