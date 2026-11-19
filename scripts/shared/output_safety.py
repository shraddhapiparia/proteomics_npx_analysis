"""
Guard against silently overwriting existing analysis outputs.

Usage pattern in an analysis script:

    planned = [outdir / "table_a.tsv", outdir / "table_b.tsv"]
    check_outputs_writable(planned, overwrite=args.overwrite)   # before ANY write
    ... compute and write ...

The check runs before anything is written, so a refused run leaves every
existing file untouched (no partial overwrite).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable


def check_outputs_writable(paths: Iterable[Path], overwrite: bool) -> None:
    """Exit with an error if any planned output already exists and overwrite is False."""
    paths = [Path(p) for p in paths]
    existing = [p for p in paths if p.exists()]

    print(f"\nPlanned outputs ({len(paths)}):")
    for p in paths:
        tag = "  [exists]" if p.exists() else ""
        print(f"  {p}{tag}")

    if existing and not overwrite:
        sys.exit(
            f"\nERROR: {len(existing)} planned output file(s) already exist; nothing "
            "was written.\n"
            "  Re-run with --outdir <new directory> to write elsewhere, or pass\n"
            "  --overwrite to replace them deliberately (e.g. regenerating\n"
            "  manuscript tables)."
        )
    if existing and overwrite:
        print(f"  [--overwrite] {len(existing)} existing file(s) will be replaced.")
