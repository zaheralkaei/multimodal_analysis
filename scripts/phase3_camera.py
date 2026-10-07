"""
Phase 3 — Camera movement classification from frame-to-frame motion.

Reads:  <PROCESSED>/frames/frame_*.jpg + metadata.json (from Phase 0)
        <PROCESSED>/shots.json (from Phase 1)
Writes: <PROCESSED>/shot_camera.csv — per-shot camera movement classification

Method (round 3):
  1. For each consecutive pair of extracted frames inside a shot, track corner
     features (Shi-Tomasi + pyramidal Lucas-Kanade).
  2. Fit a similarity transform (translation + scale + rotation) with RANSAC.
     The largest consistent motion is the camera (background); people moving
     independently are rejected as outliers instead of being averaged in, which
     was the main failure of the previous mean-dense-flow approach.
  3. Express motion per second and relative to frame size, so thresholds do not
     depend on resolution or on the extraction fps:
       pan_speed   = −tx / width  per second  (+ = camera pans right)
       tilt_speed  = −ty / height per second  (+ = camera tilts down)
       zoom_rate   = ln(scale)    per second  (+ = zoom in)
     Signs are inverted for pan/tilt: when the camera pans right, the image
     content moves left.
  4. Per shot, take the median of each over all frame pairs and classify:
       static    all medians below their thresholds and little jitter
       handheld  medians below thresholds but frame-to-frame jitter is high
       zoom-in/out, pan-left/right, tilt-up/down
                 the largest component (relative to its threshold) wins
       handheld  also when jitter exceeds the net pan/tilt (shake, not a pan)
       unknown   motion could not be estimated (too few features AND frames differ)

Thresholds are CLI flags; defaults are deliberately conservative.

Note: the previous version labelled directions backwards (content moving
right was called pan-right; outward flow was called zoom-out).
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path

from common import (FRAMES_DIR, PROCESSED, display_path, frame_fps as load_frame_fps,
                    frame_path, frames_in_range, record_run)

DEFAULTS = {
    "pan_thresh": 0.04,     # fraction of frame width per second
    "tilt_thresh": 0.04,    # fraction of frame height per second
    "zoom_thresh": 0.03,    # ln(scale) per second (≈3% size change per second)
    "jitter_thresh": 0.06,  # std of per-pair pan/tilt speed for "handheld"
    "min_inliers": 10,
    "static_diff": 3.0,     # mean abs grey-level difference treated as "no change"
}


STILL = {"pan_speed": 0.0, "tilt_speed": 0.0, "zoom_rate": 0.0, "inlier_ratio": 1.0}


def pair_motion(img1, img2, fps: float, min_inliers: int = 10, static_diff: float = 3.0) -> dict | None:
    """Camera motion between two BGR frames, or None if it cannot be estimated.

    Low-texture frames (flat colour, darkness) have too few trackable corners;
    if such a pair is also nearly identical pixel-wise, it is a still pair.
    """
    import cv2
    import numpy as np
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h, w = g1.shape

    def fallback():
        return dict(STILL) if float(np.mean(cv2.absdiff(g1, g2))) < static_diff else None

    p1 = cv2.goodFeaturesToTrack(g1, maxCorners=400, qualityLevel=0.001, minDistance=max(4, w // 80))
    if p1 is None or len(p1) < min_inliers:
        return fallback()
    p2, status, _ = cv2.calcOpticalFlowPyrLK(g1, g2, p1, None, winSize=(21, 21), maxLevel=3)
    ok = status.reshape(-1) == 1
    if ok.sum() < min_inliers:
        return fallback()
    a, b = p1[ok].reshape(-1, 2), p2[ok].reshape(-1, 2)
    M, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None or inliers is None or inliers.sum() < min_inliers:
        return fallback()
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    # Translation of the image centre (not of the origin), so a pure zoom about
    # the centre does not show up as pan/tilt.
    cx, cy = w / 2, h / 2
    tx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2] - cx
    ty = M[1, 0] * cx + M[1, 1] * cy + M[1, 2] - cy
    return {
        "pan_speed": float(-tx / w * fps),
        "tilt_speed": float(-ty / h * fps),
        "zoom_rate": float(np.log(max(scale, 1e-6)) * fps),
        "inlier_ratio": float(inliers.sum() / len(a)),
    }


def classify_shot(pairs: list[dict], th: dict) -> dict:
    """Aggregate per-pair motion into one label + summary numbers for the shot."""
    import numpy as np
    if not pairs:
        return {"camera_motion": "unknown", "pan": 0.0, "tilt": 0.0, "zoom": 0.0, "jitter": 0.0, "inlier": 0.0}
    pan = float(np.median([p["pan_speed"] for p in pairs]))
    tilt = float(np.median([p["tilt_speed"] for p in pairs]))
    zoom = float(np.median([p["zoom_rate"] for p in pairs]))
    inlier = float(np.median([p["inlier_ratio"] for p in pairs]))
    jitter = 0.0
    if len(pairs) >= 2:
        jitter = float(max(np.std([p["pan_speed"] for p in pairs]), np.std([p["tilt_speed"] for p in pairs])))

    ratios = {"pan": abs(pan) / th["pan_thresh"], "tilt": abs(tilt) / th["tilt_thresh"],
              "zoom": abs(zoom) / th["zoom_thresh"]}
    kind, ratio = max(ratios.items(), key=lambda kv: kv[1])
    # Shake: frame-to-frame motion that is large but keeps changing direction,
    # so it outweighs the net (median) pan/tilt.
    shaky = jitter > th["jitter_thresh"] and jitter > max(abs(pan), abs(tilt))
    if ratio < 1.0 or shaky:
        label = "handheld" if jitter > th["jitter_thresh"] else "static"
    elif kind == "zoom":
        label = "zoom-in" if zoom > 0 else "zoom-out"
    elif kind == "pan":
        label = "pan-right" if pan > 0 else "pan-left"
    else:
        label = "tilt-down" if tilt > 0 else "tilt-up"
    return {"camera_motion": label, "pan": pan, "tilt": tilt, "zoom": zoom, "jitter": jitter, "inlier": inlier}


def analyze_shots(shots: list[dict], frames_dir: Path, fps: float, th: dict | None = None) -> list[dict]:
    """Classify camera motion for every shot."""
    import cv2
    th = {**DEFAULTS, **(th or {})}
    rows = []
    for i, shot in enumerate(shots):
        paths = [p for p in (frame_path(k, frames_dir) for k in
                             frames_in_range(float(shot["start_sec"]), float(shot["end_sec"]), fps))
                 if p.exists()]
        pairs = []
        prev = cv2.imread(str(paths[0])) if paths else None
        for p in paths[1:]:
            cur = cv2.imread(str(p))
            if prev is not None and cur is not None:
                m = pair_motion(prev, cur, fps, th["min_inliers"], th["static_diff"])
                if m is not None:
                    pairs.append(m)
            prev = cur
        c = classify_shot(pairs, th)
        if len(paths) < 2:
            c["camera_motion"] = "unknown"  # a single frame cannot show motion
        rows.append({
            "shot_idx": i,
            "start_sec": shot["start_sec"],
            "end_sec": shot["end_sec"],
            "duration_sec": shot["duration_sec"],
            "n_frames": len(paths),
            "n_pairs_used": len(pairs),
            "camera_motion": c["camera_motion"],
            # Kept the round-2 column names; values are now per-second speeds
            "pan_score_mean": round(c["pan"], 4),
            "tilt_score_mean": round(c["tilt"], 4),
            "zoom_score_mean": round(c["zoom"], 4),
            "jitter": round(c["jitter"], 4),
            "inlier_ratio": round(c["inlier"], 3),
        })
        if (i + 1) % 10 == 0 or i == len(shots) - 1:
            print(f"  [{i + 1}/{len(shots)}] shots classified")
    return rows


CSV_COLS = ["shot_idx", "start_sec", "end_sec", "duration_sec", "n_frames", "n_pairs_used",
            "camera_motion", "pan_score_mean", "tilt_score_mean", "zoom_score_mean",
            "jitter", "inlier_ratio"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in DEFAULTS.items():  # every threshold is tunable from the CLI
        parser.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    args = parser.parse_args()
    th = {k: getattr(args, k) for k in DEFAULTS}

    shots_path = PROCESSED / "shots.json"
    if not shots_path.exists():
        print(f"[error] shots.json not found at {shots_path}")
        print("  run phase 1 first")
        return 1
    shots = json.loads(shots_path.read_text(encoding="utf-8"))
    fps = load_frame_fps()
    print(f"[info] loaded {len(shots)} shots; frames at {fps:g} fps; thresholds {th}")

    rows = analyze_shots(shots, FRAMES_DIR, fps, th)

    out_csv = PROCESSED / "shot_camera.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(rows)

    counts = Counter(r["camera_motion"] for r in rows)
    print(f"\n[ok] wrote {display_path(out_csv)} ({len(rows)} rows)")
    print("[stats] camera motion distribution:")
    for direction, count in counts.most_common():
        print(f"   {direction:<14} {count:>4} shots ({count / len(rows) * 100:>5.1f}%)")

    record_run(3, inputs=[shots_path, FRAMES_DIR], outputs=[out_csv], params={**th, "frame_fps": fps})
    print("\n[next] Phase 4: python scripts/phase4_transcribe.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
