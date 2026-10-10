"""Shared path-display helper (round-4 audit follow-up).

Several scripts printed paths via `path.relative_to(REPO_ROOT)`, which crashes
when the path is relative (e.g. `PROCESSED_DIR=data/vid`) or lives outside the
repo (env-override dirs). Use `disp()` everywhere a path is shown to a user.
"""
from __future__ import annotations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def disp(path) -> str:
    """Human-readable path: repo-relative when possible, else as-is."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except (ValueError, OSError):
        return str(p)