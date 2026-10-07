"""
Phase 1 — Shot boundary detection with PySceneDetect.

Reads:  <PROCESSED>/metadata.json + frames/frame_*.jpg + source video (from Phase 0)
Writes: <PROCESSED>/shots.json — list of {start_sec, end_sec, mid_sec, mid_frame_path, key_frame_paths}
        <PROCESSED>/shot_predictions.csv — per-frame boundary flags (debug)
        <PROCESSED>/shot_detection_stats.json — detection metadata

Uses PySceneDetect (https://github.com/Breakthrough/PySceneDetect, BSD-3-Clause).

Detectors (--detector):
  adaptive (default)  AdaptiveDetector: compares each frame's HSV/edge change to
                      a rolling average of its neighbours, so fast camera
                      motion or flashing lights (common in music videos) cause
                      fewer false cuts than a fixed threshold.
  content             ContentDetector with a fixed threshold (the round-1/2
                      behaviour, --threshold 35).

Both are HARD-CUT detectors. --fades (on by default) adds ThresholdDetector
for fades to/from black. Cross-dissolves are still not detected; that needs a
learned model such as TransNetV2 (see docs/METHODOLOGY_REVIEW.md).

Each shot also gets three key frames (at 15%, 50%, 85% of its duration) that
Phase 2 sends to the vision model, so it can see change within the shot.
"""
from __future__ import annotations
import argparse, bisect, csv, json, sys
from pathlib import Path

from common import (FRAMES_DIR, PROCESSED, RAW, count_frames, display_path,
                    frame_for_time, frame_fps as load_frame_fps, frame_path,
                    frames_in_range, load_metadata, record_run)

KEY_FRAME_POSITIONS = (0.15, 0.5, 0.85)


def _secs(tc) -> float:
    """FrameTimecode → seconds (works on PySceneDetect 0.6 and 0.7)."""
    return float(tc.seconds) if hasattr(type(tc), "seconds") else float(tc.get_seconds())


def _frames(tc) -> int:
    return int(tc.frame_num) if hasattr(type(tc), "frame_num") else int(tc.get_frames())


def build_detectors(detector: str, threshold: float, adaptive_threshold: float,
                    min_scene_len: int, fades: bool) -> list:
    from scenedetect import AdaptiveDetector, ContentDetector, ThresholdDetector
    weights = ContentDetector.Components(delta_hue=1.0, delta_sat=1.0, delta_lum=1.0, delta_edges=2.0)
    if detector == "content":
        dets = [ContentDetector(threshold=threshold, min_scene_len=min_scene_len, weights=weights)]
    else:
        dets = [AdaptiveDetector(adaptive_threshold=adaptive_threshold, min_scene_len=min_scene_len,
                                 weights=weights)]
    if fades:
        dets.append(ThresholdDetector(threshold=12, min_scene_len=min_scene_len))
    return dets


def shots_from_boundaries(bounds: list[tuple[float, float]], frame_fps: float,
                          n_extracted: int) -> list[dict]:
    """Turn (start_sec, end_sec) pairs into our shot schema."""
    shots = []
    for idx, (start_sec, end_sec) in enumerate(bounds):
        duration = end_sec - start_sec
        mid_sec = (start_sec + end_sec) / 2
        key_paths = [display_path(frame_path(frame_for_time(start_sec + p * duration, frame_fps, n_extracted)))
                     for p in KEY_FRAME_POSITIONS]
        shots.append({
            "shot_idx": idx,
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "duration_sec": round(duration, 3),
            "mid_sec": round(mid_sec, 3),
            "mid_frame_path": display_path(frame_path(frame_for_time(mid_sec, frame_fps, n_extracted))),
            "key_frame_paths": key_paths,
            # Extracted frames whose timestamp falls in [start, end)
            "n_frames": len(frames_in_range(start_sec, end_sec, frame_fps)),
        })
    return shots


def detect_shots(video_path: Path, detectors: list, frame_fps: float = 1.0) -> tuple[list[dict], float, int]:
    """Run PySceneDetect on the video. Returns (shots, video_fps, total_frames)."""
    from scenedetect import SceneManager, open_video

    video = open_video(str(video_path))
    fps = float(video.frame_rate)
    total_frames = _frames(video.duration)
    print(f"[info] video: {video_path.name}, fps={fps:.2f}, frames={total_frames}")

    sm = SceneManager()
    for d in detectors:
        sm.add_detector(d)
    sm.detect_scenes(video, show_progress=False)
    scene_list = sm.get_scene_list()  # [(start, end), ...]; end is exclusive
    if not scene_list:  # no cut found: the whole video is one shot
        bounds = [(0.0, total_frames / fps)]
    else:
        bounds = [(_secs(s), _secs(e)) for s, e in scene_list]
    return shots_from_boundaries(bounds, frame_fps, count_frames()), fps, total_frames


def write_predictions(path: Path, shots: list[dict], fps: float, total_frames: int) -> None:
    """Per-video-frame boundary flag + shot index (debug / detector comparison)."""
    starts = [s["start_sec"] for s in shots]
    boundary_frames = {int(round(s * fps)) for s in starts}
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_sec", "is_boundary", "shot_idx"])
        for frame_idx in range(total_frames):
            t = frame_idx / fps
            k = bisect.bisect_right(starts, t) - 1
            shot_idx = shots[k]["shot_idx"] if k >= 0 and t < shots[k]["end_sec"] else -1
            w.writerow([frame_idx, round(t, 3), int(frame_idx in boundary_frames), shot_idx])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", choices=["adaptive", "content"], default="adaptive",
                        help="cut detector (default adaptive)")
    parser.add_argument("--threshold", type=float, default=35.0,
                        help="ContentDetector threshold, --detector content only (default 35.0)")
    parser.add_argument("--adaptive-threshold", type=float, default=3.0,
                        help="AdaptiveDetector ratio threshold (default 3.0, lower = more sensitive)")
    parser.add_argument("--min-scene-len", type=int, default=30,
                        help="Minimum shot length in frames (default 30 ≈ 1.25s at 24fps)")
    parser.add_argument("--no-fades", dest="fades", action="store_false",
                        help="disable fade-to/from-black detection")
    parser.add_argument("--video", default=None,
                        help="Path to source video (default: source_file from Phase 0's metadata.json)")
    args = parser.parse_args()

    metadata = load_metadata()
    frame_fps = load_frame_fps()
    video_path = Path(args.video or metadata.get("source_file") or RAW / "video.mp4")
    if not video_path.exists():
        print(f"[error] video not found: {video_path}")
        print("  run phase 0 first: python scripts/phase0_input.py <source>")
        return 1

    params = {"detector": args.detector, "threshold": args.threshold,
              "adaptive_threshold": args.adaptive_threshold,
              "min_scene_len_frames": args.min_scene_len, "fades": args.fades}
    print(f"[info] PySceneDetect: {params}")
    detectors = build_detectors(args.detector, args.threshold, args.adaptive_threshold,
                                args.min_scene_len, args.fades)
    shots, fps, total_frames = detect_shots(video_path, detectors, frame_fps)
    print(f"[ok] detected {len(shots)} shots")

    out_path = PROCESSED / "shots.json"
    out_path.write_text(json.dumps(shots, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {display_path(out_path)} ({len(shots)} shots)")

    pred_path = PROCESSED / "shot_predictions.csv"
    write_predictions(pred_path, shots, fps, total_frames)
    print(f"[ok] wrote {display_path(pred_path)} ({total_frames} rows)")

    durations = [s["duration_sec"] for s in shots]
    stats = {
        "detector": f"PySceneDetect-{'Adaptive' if args.detector == 'adaptive' else 'Content'}Detector"
                    + ("+ThresholdDetector(fades)" if args.fades else ""),
        "detector_version": _get_scenedetect_version(),
        **params,
        "fps": round(fps, 3),
        "total_frames": int(total_frames),
        "n_shots": len(shots),
        "avg_shot_duration_sec": round(sum(durations) / max(1, len(shots)), 3),
        "min_shot_duration_sec": round(min(durations, default=0), 3),
        "max_shot_duration_sec": round(max(durations, default=0), 3),
        "video_path": display_path(video_path),
    }
    stats_path = PROCESSED / "shot_detection_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {display_path(stats_path)}")

    if shots:
        print(f"\n[stats] {len(shots)} shots, "
              f"avg {stats['avg_shot_duration_sec']:.1f}s, "
              f"min {stats['min_shot_duration_sec']:.1f}s, "
              f"max {stats['max_shot_duration_sec']:.1f}s")

    record_run(1, inputs=[video_path, PROCESSED / "metadata.json", FRAMES_DIR],
               outputs=[out_path, pred_path, stats_path], params=params)
    print("\n[next] Phase 2: python scripts/phase2_vision.py")
    return 0


def _get_scenedetect_version() -> str:
    try:
        import scenedetect
        return scenedetect.__version__
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
