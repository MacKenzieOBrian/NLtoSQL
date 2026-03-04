"""
Shared helpers for notebook experiment orchestration.

Currently kept minimal: only model ID normalization used by notebooks.
"""

from __future__ import annotations

import re


def model_alias_from_id(model_id: str) -> str:
    """Convert a model id into a filesystem-safe alias."""
    tail = (model_id or "model").split("/")[-1]
    alias = re.sub(r"[^a-z0-9]+", "_", tail.lower()).strip("_")
    return alias or "model"


__all__ = [
    "model_alias_from_id",
]
