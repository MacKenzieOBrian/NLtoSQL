#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def _drop_widgets(obj: dict) -> bool:
    changed = False
    metadata = obj.get("metadata")
    if isinstance(metadata, dict) and "widgets" in metadata:
        # GitHub notebook renderer expects `metadata.widgets.state` when this key exists.
        # If it's missing or uses a different schema, the safest fix is to drop it.
        widgets = metadata.get("widgets")
        if not isinstance(widgets, dict) or "state" not in widgets:
            metadata.pop("widgets", None)
            changed = True
    return changed


def fix_notebook(nb: dict) -> tuple[dict, bool]:
    changed = _drop_widgets(nb)

    cells = nb.get("cells")
    if isinstance(cells, list):
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            changed = _drop_widgets(cell) or changed

    return nb, changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fix notebooks that fail to render on GitHub due to invalid `metadata.widgets`."
    )
    parser.add_argument("input", type=Path, help="Input .ipynb path")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Write to a new path")
    parser.add_argument("--dry-run", action="store_true", help="Report whether changes are needed")
    args = parser.parse_args()

    nb = json.loads(args.input.read_text(encoding="utf-8"))
    nb, changed = fix_notebook(nb)

    if args.dry_run:
        print("changed" if changed else "no-change")
        return 0

    out_path = args.output or args.input
    out_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
