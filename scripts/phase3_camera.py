"""
Phase 3 — Camera movement classification via optical flow.

Reads:  data/processed/frames/frame_*.jpg (from Phase 0)
        data/processed/shots.json (from Phase 1)
Writes: data/processed/shot_camera.csv — per-shot camera movement classification

Uses OpenCV's calcOpticalFlowFarneback between consecutive frames within each
shot, then classifies the motion field as: static / pan / tilt / zoom / handheld.

This is faster and more reliable than asking a vision-language model "is this
a pan?" for every shot.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"
FRAMES_DIR = PROCESSED / "frames"


def optical_flow_between(img1, img2):
    """Compute dense optical flow between two grayscale frames."""
    import cv2
    import numpy as np
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    return flow


def classify_motion(flow) -> dict:
    """Classify the dominant motion in an optical flow field.

    Returns dict with: dominant_direction, dominant_motion, median_magnitude,
    pan_score, tilt_score, zoom_score, static_score.
    """
    import numpy as np
    fx = flow[..., 0]
    fy = flow[..., 1]
    mag = np.sqrt(fx ** 2 + fy ** 2)
    med_mag = float(np.median(mag))

    # If flow is tiny, it's static
    if med_mag < 0.3:
        return {
            "dominant_direction": "static",
            "median_magnitude": med_mag,
            "pan_score": 0.0,
            "tilt_score": 0.0,
            "zoom_score": 0.0,
            "static_score": 1.0,
            "n_frames": 0,
        }

    # Mean flow direction (excluding tiny vectors)
    mask = mag > 1.0
    if mask.sum() < 100:
        mask = mag > 0.5
    if mask.sum() == 0:
        return {
            "dominant_direction": "static",
            "median_magnitude": med_mag,
            "pan_score": 0.0, "tilt_score": 0.0, "zoom_score": 0.0, "static_score": 1.0,
            "n_frames": 0,
        }

    mean_fx = float(np.mean(fx[mask]))
    mean_fy = float(np.mean(fy[mask]))

    # Divergence tells us about zoom
    # fx grows with x position (zoom out = divergent from center, zoom in = convergent)
    h, w = flow.shape[:2]
    cy, cx = h / 2, w / 2
    yy, xx = np.mgrid[0:h, 0:w]
    # Vector from center
    rx = xx - cx
    ry = yy - cy
    # Divergence = sum over (d fx/dx + d fy/dy)
    # Simpler: compute correlation between flow direction and outward radial vector
    radial_x = rx / (np.sqrt(rx ** 2 + ry ** 2) + 1e-6)
    radial_y = ry / (np.sqrt(rx ** 2 + ry ** 2) + 1e-6)
    # Dot product: positive = outward (zoom out), negative = inward (zoom in)
    dot = (fx * radial_x + fy * radial_y)[mask]
    zoom_score = float(np.clip(np.mean(dot) / 5.0, -1.0, 1.0))

    # Pan: horizontal motion (fx dominant, fy small)
    pan_score = float(np.clip(mean_fx / 5.0, -1.0, 1.0))
    tilt_score = float(np.clip(mean_fy / 5.0, -1.0, 1.0))

    # Decide dominant
    abs_pan, abs_tilt, abs_zoom = abs(pan_score), abs(tilt_score), abs(zoom_score)
    if abs_zoom > max(abs_pan, abs_tilt) and abs_zoom > 0.2:
        direction = "zoom-out" if zoom_score > 0 else "zoom-in"
    elif abs_pan > abs_tilt:
        direction = "pan-right" if pan_score > 0 else "pan-left"
    elif abs_tilt > 0.2:
        direction = "tilt-down" if tilt_score > 0 else "tilt-up"
    else:
        direction = "handheld"

    return {
        "dominant_direction": direction,
        "median_magnitude": round(med_mag, 3),
        "pan_score": round(pan_score, 3),
        "tilt_score": round(tilt_score, 3),
        "zoom_score": round(zoom_score, 3),
        "static_score": 0.0,
        "n_frames": int(mask.sum()),
    }


def analyze_shots(shots: list[dict], frames_dir: Path) -> list[dict]:
    """For each shot, compute optical flow between consecutive frames."""
    import cv2
    import numpy as np

    rows = []
    for i, shot in enumerate(shots):
        # Find frame files for this shot
        # Shot's start_sec and end_sec correspond to frame indices at 1 fps
        start_frame = int(shot["start_sec"]) + 1  # 1-indexed filenames
        end_frame = int(shot["end_sec"]) + 1
        frame_paths = []
        for fi in range(start_frame, end_frame + 1):
            p = frames_dir / f"frame_{fi:05d}.jpg"
            if p.exists():
                frame_paths.append(p)

        if len(frame_paths) < 2:
            print(f"[warn] shot {i}: only {len(frame_paths)} frames, skipping")
            continue

        # Accumulate motion across all frame pairs in this shot
        all_directions = []
        all_pan, all_tilt, all_zoom = [], [], []
        all_mags = []
        for j in range(len(frame_paths) - 1):
            img1 = cv2.imread(str(frame_paths[j]))
            img2 = cv2.imread(str(frame_paths[j + 1]))
            if img1 is None or img2 is None:
                continue
            flow = optical_flow_between(img1, img2)
            m = classify_motion(flow)
            all_directions.append(m["dominant_direction"])
            all_pan.append(m["pan_score"])
            all_tilt.append(m["tilt_score"])
            all_zoom.append(m["zoom_score"])
            all_mags.append(m["median_magnitude"])

        if not all_directions:
            continue

        # Majority vote on direction
        from collections import Counter
        direction_counts = Counter(all_directions)
        direction = direction_counts.most_common(1)[0][0]

        rows.append({
            "shot_idx": i,
            "start_sec": shot["start_sec"],
            "end_sec": shot["end_sec"],
            "duration_sec": shot["duration_sec"],
            "n_frames": len(frame_paths),
            "camera_motion": direction,
            "pan_score_mean": round(float(np.mean(all_pan)), 3),
            "tilt_score_mean": round(float(np.mean(all_tilt)), 3),
            "zoom_score_mean": round(float(np.mean(all_zoom)), 3),
            "median_motion_magnitude": round(float(np.median(all_mags)), 3),
        })

        if (i + 1) % 10 == 0 or i == len(shots) - 1:
            print(f"  [{i+1}/{len(shots)}] shots classified")

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args = parser.parse_args()

    shots_path = PROCESSED / "shots.json"
    if not shots_path.exists():
        print(f"[error] shots.json not found at {shots_path}")
        print("  run phase 1 first")
        return 1

    shots = json.loads(shots_path.read_text(encoding="utf-8"))
    print(f"[info] loaded {len(shots)} shots")

    rows = analyze_shots(shots, FRAMES_DIR)

    out_csv = PROCESSED / "shot_camera.csv"
    cols = ["shot_idx", "start_sec", "end_sec", "duration_sec", "n_frames",
            "camera_motion", "pan_score_mean", "tilt_score_mean", "zoom_score_mean",
            "median_motion_magnitude"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    from collections import Counter
    counts = Counter(r["camera_motion"] for r in rows)
    print(f"\n[ok] wrote {out_csv.relative_to(REPO_ROOT)} ({len(rows)} rows)")
    print(f"[stats] camera motion distribution:")
    for direction, count in counts.most_common():
        print(f"   {direction:<14} {count:>4} shots ({count/len(rows)*100:>5.1f}%)")

    print(f"\n[next] Phase 4: python scripts/phase4_transcribe.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
