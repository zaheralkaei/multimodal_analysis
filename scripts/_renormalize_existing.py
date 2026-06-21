"""
Helper: re-normalize emotion in an EXISTING shot_vision.csv without re-running
the whole pipeline. Useful when you've added new synonyms to _normalize_emotion
and want to apply them to the current data.

Usage:
    python scripts/_renormalize_existing.py
    # Writes data/processed/shot_vision_normalized.csv (does NOT overwrite original)
"""
from __future__ import annotations
import csv, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"
sys.path.insert(0, str(Path(__file__).parent))
from _normalize_emotion import normalize_emotion


def main() -> int:
    src = PROCESSED / "shot_vision.csv"
    if not src.exists():
        print(f"[error] {src} not found")
        return 1
    dst = PROCESSED / "shot_vision_normalized.csv"
    n = 0
    n_changed = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            old = row.get("emotion", "")
            new = normalize_emotion(old) if old else ""
            if old != new:
                n_changed += 1
            row["emotion"] = new
            writer.writerow(row)
            n += 1
    print(f"[ok] re-normalized {n} rows ({n_changed} changed)")
    print(f"[ok] wrote {dst.relative_to(REPO_ROOT)}")
    print()
    print("To use the normalized data, either:")
    print(f"  cp {dst} {src}")
    print("or update phase7_sync.py to read shot_vision_normalized.csv instead")
    return 0


if __name__ == "__main__":
    sys.exit(main())