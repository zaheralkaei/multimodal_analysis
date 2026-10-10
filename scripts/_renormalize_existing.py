"""
Helper: re-normalize emotion in an EXISTING shot_vision.csv without re-running
the whole pipeline. Useful when you've added new synonyms to _normalize_emotion
and want to apply them to the current data.

Usage:
    python scripts/_renormalize_existing.py
    # Writes <PROCESSED_DIR>/shot_vision_normalized.csv (does NOT overwrite original)

Honors the PROCESSED_DIR env var like every phase script (round-4 audit F7:
the old code always read data/processed/, which broke in the v3 per-video
restructure).
"""
from __future__ import annotations
import csv, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
from _paths import disp
PROCESSED = REPO_ROOT / "data" / "processed"
if "PROCESSED_DIR" in os.environ:
    PROCESSED = Path(os.environ["PROCESSED_DIR"])
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
    print(f"[ok] wrote {disp(dst)}")
    print()
    print("To use the normalized data, either:")
    print(f"  cp {dst} {src}")
    print("or update phase7_sync.py to read shot_vision_normalized.csv instead")
    return 0


if __name__ == "__main__":
    sys.exit(main())