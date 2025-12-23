#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def strip_notebook(nb: dict) -> dict:
    nb_out = {
        "cells": [],
        "metadata": {},
        "nbformat": nb.get("nbformat", 4),
        "nbformat_minor": nb.get("nbformat_minor", 5),
    }

    # Keep minimal, widely-used metadata for nicer local rendering.
    for key in ("kernelspec", "language_info"):
        if isinstance(nb.get("metadata"), dict) and key in nb["metadata"]:
            nb_out["metadata"][key] = nb["metadata"][key]

    cells = nb.get("cells", [])
    if not isinstance(cells, list):
        cells = []

    for cell in cells:
        if not isinstance(cell, dict):
            continue

        cell_type = cell.get("cell_type", "markdown")
        out_cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": cell.get("source", []),
        }

        if cell_type == "code":
            out_cell["execution_count"] = None
            out_cell["outputs"] = []

        # Preserve attachments for markdown cells if present.
        if cell_type == "markdown" and "attachments" in cell:
            out_cell["attachments"] = cell["attachments"]

        nb_out["cells"].append(out_cell)

    return nb_out


def main() -> int:
    parser = argparse.ArgumentParser(description="Strip outputs and metadata from a Jupyter notebook.")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    cleaned = strip_notebook(raw)

    out_path = args.output or args.input
    out_path.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

