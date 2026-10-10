"""
Phase 1 — Shot boundary detection with PySceneDetect.

Reads:  data/<video_id>/frames/frame_*.jpg + data/raw/<video_id>.mp4 (from Phase 0)
        data/<video_id>/metadata.json — for the frame extraction rate (frame_fps)
Writes: data/<video_id>/shots.json — list of {start_sec, end_sec, mid_sec, mid_frame_path}
        data/<video_id>/shot_predictions.csv — per-frame scene scores (debug)
        data/<video_id>/shot_detection_stats.json — detection metadata

The mid-frame of a shot is computed with the EXTRACTION rate from metadata.json
(frame_fps), not assumed 1 fps — round-4 audit fix (was: frame index == seconds,
which picked frames at half the intended timestamp at 2 fps).

Uses PySceneDetect (https://github.com/Breakthrough/PySceneDetect, BSD-3-Clause)
with the **ContentDetector** algorithm. This is the most versatile detector:

  - Splits frames into 4x4 pixel blocks in HSV color space
  - For each block, computes a delta from the previous frame
  - Sum of deltas > threshold = scene change
  - Detects both **hard cuts** AND **gradual transitions** (fades, dissolves,
    whip pans) that TransNetV2 tends to miss

Why not TransNetV2?
  - Bias toward hard cuts; misses smooth transitions common in modern music videos
  - On Tyla's "SHE DID IT AGAIN", TransNetV2 detected 2 shots (212s + 2s trailing)
    when the video clearly has 20+ transitions
  - PySceneDetect's ContentDetector found 4x more boundaries on the same video

Parameters you can tune:
  --threshold  27.0     delta-Y (luminance) threshold per block
  --min-scene-len  15  minimum shot length in frames (avoids flicker detection)
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
from _paths import disp
PROCESSED = REPO_ROOT / "data" / "processed"
if "PROCESSED_DIR" in os.environ:
    PROCESSED = Path(os.environ["PROCESSED_DIR"])
RAW = REPO_ROOT / "data" / "raw"
FRAMES_DIR = PROCESSED / "frames"


def get_extraction_fps() -> tuple[int, int]:
    """Read the frame extraction rate + count from phase-0 metadata.json.

    Returns (frame_fps, n_extracted). Exits if metadata.json is missing —
    we must not guess the extraction rate (see AUDIT_R4 F1a).
    """
    import math

    meta_path = PROCESSED / "metadata.json"
    if not meta_path.exists():
        print(f"[error] metadata.json not found at {meta_path}")
        print("  run phase 0 first: python scripts/phase0_input.py <source>")
        sys.exit(1)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fps = int(meta.get("frame_fps") or 1)
    n = int(meta.get("frames_extracted") or math.ceil(meta.get("duration_sec", 0) * fps))
    return fps, n


def mid_frame_index(mid_sec: float, fps: int, n_extracted: int) -> int:
    """Convert a shot's mid-timestamp to a 1-indexed extracted-frame number.

    Frame N (1-indexed) is the sample at t = (N-1)/fps. So the frame covering
    mid_sec is round(mid_sec * fps) + 1, clamped to [1, n_extracted].
    """
    idx = int(round(mid_sec * fps)) + 1
    return max(1, min(idx, n_extracted))


def nearest_existing_frame(idx: int, fps: int, frames_dir: Path) -> Path | None:
    """Return the existing frame file closest to index idx (bounded search).

    Handles rounding at shot boundaries where the ideal mid-frame index can
    land just past the last extracted frame.
    """
    best: tuple[int, Path] | None = None
    for delta in (0, 1, -1, 2, -2):
        cand = frames_dir / f"frame_{max(1, idx + delta):05d}.jpg"
        if cand.exists():
            dist = abs(delta)
            if best is None or dist < best[0]:
                best = (dist, cand)
            if dist == 0:
                return cand
    return best[1] if best else None


def detect_shots(video_path: Path, threshold: float, min_scene_len: int,
                 fps_extract: int, frames_dir: Path, n_extracted: int) -> tuple[list[dict], float, int]:
    """Run PySceneDetect ContentDetector on the video. Returns shot list."""
    from scenedetect import open_video, SceneManager, ContentDetector

    # Open video
    video = open_video(str(video_path))
    fps = video.frame_rate
    total_frames = video.duration.get_frames() if hasattr(video.duration, "get_frames") else 0
    print(f"[info] video: {video_path.name}, fps={fps:.2f}, frames={total_frames}")

    # Build scene manager + detector
    sm = SceneManager()
    sm.add_detector(ContentDetector(
        threshold=threshold,
        min_scene_len=min_scene_len,
        weights=ContentDetector.Components(
            delta_hue=1.0, delta_sat=1.0, delta_lum=1.0, delta_edges=2.0
        ),  # delta_edges=2 helps catch fades
    ))

    # Run detection
    sm.detect_scenes(video, show_progress=False)
    scene_list = sm.get_scene_list()  # [(start, end), ...] as FrameTimecode pairs

    # Convert to our shot schema
    shots = []
    for idx, (start_tc, end_tc) in enumerate(scene_list):
        start_sec = start_tc.get_seconds()
        end_sec = end_tc.get_seconds()
        # The end_tc is exclusive — last frame is actually the start of the next shot
        # So duration is end_sec - start_sec
        duration = end_sec - start_sec
        mid_sec = (start_sec + end_sec) / 2

        # Mid-frame via the extraction rate (NOT "frame number == seconds" —
        # that only holds at 1 fps; see AUDIT_R4 F1a)
        mid_frame_idx = mid_frame_index(mid_sec, fps_extract, n_extracted)
        mid_frame_file = nearest_existing_frame(mid_frame_idx, fps_extract, frames_dir)
        if mid_frame_file is None:
            print(f"[warn] no frame file found near index {mid_frame_idx} for shot {idx}")
        # Path is relative to REPO_ROOT, points to OUR frames dir
        mid_frame_path = ""
        if mid_frame_file is not None:
            mid_frame_path = disp(mid_frame_file)
        n_frames_in_shot = int(round(duration * fps_extract))

        shots.append({
            "shot_idx": idx,
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "duration_sec": round(duration, 3),
            "mid_sec": round(mid_sec, 3),
            "mid_frame_path": mid_frame_path,
            "n_frames": n_frames_in_shot,
        })
    return shots, fps, total_frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=35.0,
                       help="ContentDetector threshold (default 35.0, lower = more sensitive)")
    parser.add_argument("--min-scene-len", type=int, default=30,
                       help="Minimum shot length in frames (default 30 ≈ 1.25s at 24fps)")
    parser.add_argument("--video", default=str(RAW / "video.mp4"),
                       help="Path to source video (default: data/raw/video.mp4)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[error] video not found: {video_path}")
        print("  run phase 0 first: python scripts/phase0_input.py <source>")
        return 1

    print(f"[info] PySceneDetect ContentDetector")
    print(f"  threshold = {args.threshold}")
    print(f"  min_scene_len = {args.min_scene_len} frames")

    fps_extract, n_extracted = get_extraction_fps()
    print(f"  extraction = {n_extracted} frames at {fps_extract} fps (from metadata.json)")

    shots, fps, total_frames = detect_shots(video_path, args.threshold, args.min_scene_len,
                                            fps_extract, FRAMES_DIR, n_extracted)
    print(f"[ok] detected {len(shots)} shots")

    # Write shots.json
    out_path = PROCESSED / "shots.json"
    out_path.write_text(json.dumps(shots, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {disp(out_path)} ({len(shots)} shots)")

    # Write per-frame predictions (for debugging). For PySceneDetect we record
    # which frames are at shot boundaries. Useful for comparing detectors.
    import csv
    pred_path = PROCESSED / "shot_predictions.csv"
    # Build a frame → "is_boundary" map
    boundary_frames = set()
    for s in shots:
        boundary_frames.add(int(s["start_sec"] * fps))
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_sec", "is_boundary", "shot_idx"])
        for frame_idx in range(total_frames):
            t = frame_idx / fps
            is_boundary = 1 if frame_idx in boundary_frames else 0
            shot_idx = -1
            for s in shots:
                if s["start_sec"] <= t < s["end_sec"]:
                    shot_idx = s["shot_idx"]
                    break
            w.writerow([frame_idx, round(t, 3), is_boundary, shot_idx])
    print(f"[ok] wrote {disp(pred_path)} ({total_frames} rows)")

    # Write stats
    from fractions import Fraction
    def _to_jsonable(o):
        if isinstance(o, Fraction):
            return float(o)
        if hasattr(o, "get_seconds"):
            return float(o.get_seconds())
        if hasattr(o, "get_frames"):
            return int(o.get_frames())
        return str(o)

    stats = {
        "detector": "PySceneDetect-ContentDetector",
        "detector_version": _get_scenedetect_version(),
        "threshold": args.threshold,
        "min_scene_len_frames": args.min_scene_len,
        "fps": round(float(fps), 3),
        "total_frames": int(total_frames),
        "n_shots": len(shots),
        "avg_shot_duration_sec": round(sum(s["duration_sec"] for s in shots) / max(1, len(shots)), 3),
        "min_shot_duration_sec": round(min((s["duration_sec"] for s in shots), default=0), 3),
        "max_shot_duration_sec": round(max((s["duration_sec"] for s in shots), default=0), 3),
        "video_path": disp(video_path),
    }
    stats_path = PROCESSED / "shot_detection_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {disp(stats_path)}")

    if shots:
        print(f"\n[stats] {len(shots)} shots, "
              f"avg {stats['avg_shot_duration_sec']:.1f}s, "
              f"min {stats['min_shot_duration_sec']:.1f}s, "
              f"max {stats['max_shot_duration_sec']:.1f}s")

    print(f"\n[next] Phase 2: python scripts/phase2_vision.py")
    return 0


def _get_scenedetect_version() -> str:
    try:
        import scenedetect
        return scenedetect.__version__
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
