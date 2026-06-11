"""Pre-commit check: catch corrupted Jupyter notebooks.

Flags two failure modes:
1. A raw or markdown cell whose source looks like a serialized notebook
   (e.g. a JSON-dump accidentally pasted as a single cell — the exact bug
   that prompted this hook).
2. Generic nbformat structural corruption.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat


NESTED_MARKERS = ('"cell_type"', '"nbformat"')


def check_notebook(path: Path) -> tuple[list[str], list[str]]:
    """Return (fatal_errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    try:
        nb = nbformat.read(path, as_version=4)
    except Exception as exc:
        return [f"{path}: failed to parse as notebook ({exc})"], warnings

    try:
        nbformat.validate(nb)
    except nbformat.ValidationError as exc:
        warnings.append(f"{path}: nbformat validation warning ({exc.message})")

    for i, cell in enumerate(nb.cells):
        if cell.cell_type not in ("raw", "markdown"):
            continue
        src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
        if all(marker in src for marker in NESTED_MARKERS):
            errors.append(
                f"{path}: cell {i} ({cell.cell_type}) appears to contain a "
                "serialized notebook — likely a bad paste or merge artifact"
            )
    return errors, warnings


def main(argv: list[str]) -> int:
    all_errors: list[str] = []
    all_warnings: list[str] = []
    for arg in argv:
        errs, warns = check_notebook(Path(arg))
        all_errors.extend(errs)
        all_warnings.extend(warns)
    for warn in all_warnings:
        print(f"warning: {warn}", file=sys.stderr)
    for err in all_errors:
        print(f"error: {err}", file=sys.stderr)
    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
