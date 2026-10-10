"""
One-off repair utility for AUDIT_R4 F1a: recompute every shot's mid_frame_path
in shots.json (and the mid_frame column in shot_vision.csv / sync_per_shot.csv)
with the fps-aware frame math, WITHOUT re-running shot detection. Shot timing
(start_sec/end_sec) was never affected — only the frame-file choice was.

Usage:
    python scripts/_refit_midframes.py            # all videos under data/
    python scripts/_refit_midframes.py <video_id> # one video

Note: the vision CSV's captions still describe the OLD (wrong) frames until
phase 2 is re-run. The resume logic in phase 2 re-analyzes rows whose
mid_frame changed, so `python scripts/phase2_vision.py` (after fixing
mid-frames) will redo exactly the changed shots.
"""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
from _paths import disp
sys.path.insert(0, str(Path(__file__).parent))

from phase1_shots import mid_frame_index, nearest_existing_frame


def refit_video(video_dir: Path) -> dict:
    """Recompute mid_frame_path (+n_frames) in shots.json for one video."""
    meta_path = video_dir / "metadata.json"
    shots_path = video_dir / "shots.json"
    if not meta_path.exists() or not shots_path.exists():
        return {"video": video_dir.name, "status": "skipped (missing metadata/shots.json)"}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fps = int(meta.get("frame_fps") or 1)
    n_extracted = int(meta.get("frames_extracted") or 0)
    frames_dir = video_dir / "frames"
    shots = json.loads(shots_path.read_text(encoding="utf-8"))

    if n_extracted <= 0:
        n_extracted = len(list(frames_dir.glob("frame_*.jpg")))
    old_map = {}
    changed = 0
    for s in shots:
        old_map[s["shot_idx"]] = s.get("mid_frame_path", "")
        mid = float(s["mid_sec"])
        idx = mid_frame_index(mid, fps, n_extracted)
        f = nearest_existing_frame(idx, fps, frames_dir)
        try:
            new_rel = disp(f) if f else ""
        except ValueError:  # PROCESSED_DIR can point outside the repo
            new_rel = str(f) if f else ""
        if new_rel != old_map[s["shot_idx"]]:
            changed += 1
        s["mid_frame_path"] = new_rel
        duration = float(s["end_sec"]) - float(s["start_sec"])
        s["n_frames"] = int(round(duration * fps))

    shots_path.write_text(json.dumps(shots, indent=2) + "\n", encoding="utf-8")

    # Propagate to downstream CSVs so phase 7/8 stay consistent (and so phase 2's
    # resume logic sees mid_frame changed → re-analyzes those shots)
    n_rows_prop = 0
    for csv_name in ("shot_vision.csv", "sync_per_shot.csv"):
        p = video_dir / csv_name
        if not p.exists():
            continue
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        if not rows:
            continue
        cols = list(rows[0].keys())
        for r in rows:
            idx = int(r["shot_idx"])
            r["mid_frame"] = shots[idx].get("mid_frame_path", "")
        with p.open("w", encoding="utf-8", newline="") as f_out:
            w = csv.DictWriter(f_out, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        n_rows_prop += 1

    return {"video": video_dir.name, "shots": len(shots), "changed_mid_frames": changed,
            "csvs_propagated": n_rows_prop}


def main() -> int:
    data_dir = REPO_ROOT / "data"
    targets = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    if targets:
        video_dirs = [data_dir / t for t in targets]
    else:
        video_dirs = sorted(p for p in data_dir.iterdir()
                            if p.is_dir() and (p / "shots.json").exists()) if data_dir.exists() else []

    if not video_dirs:
        print("[error] no videos found under data/ (looked for data/<id>/shots.json)")
        return 1

    for vd in video_dirs:
        result = refit_video(vd)
        print(f"[refit] {json.dumps(result)}")
    print("\n[done] shots.json + CSVs now use fps-correct mid-frames.")
    print("[note] captions/emotions still describe the OLD frames until phase 2 re-runs;")
    print("       phase 2's resume logic will re-analyze exactly the changed shots.")
    return 0


if __name__ == "__main__":
    sys.exit(main())