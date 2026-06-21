"""
Run the entire 8-phase pipeline on a video. Output goes to per-video folders.

Naming convention (round 3):
  - data/<video_id>/   — all per-video artifacts (audio, frames, CSVs, JSONs)
  - reports/<video_id>/ — dashboard + thumbnails
  - data/raw/<video_id>.mp4 — downloaded video

Where <video_id> is:
  - YouTube ID if source is a YouTube URL (e.g. "rtwpk9rb1Dc" from ?v=rtwpk9rb1Dc)
  - Slug of local filename (without extension) if source is a local path
  - Explicit --id flag overrides

Usage:
  # YouTube video — ID auto-derived from URL
  python scripts/run_pipeline.py "https://www.youtube.com/watch?v=rtwpk9rb1Dc"

  # Local video with explicit ID
  python scripts/run_pipeline.py ~/videos/my_clip.mp4 --id my_clip

  # Skip slow phases
  python scripts/run_pipeline.py URL --skip-phase2 --skip-phase5

  # Resume from phase 4
  python scripts/run_pipeline.py URL --start-from 4
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys, urllib.parse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"


def extract_youtube_id(url: str) -> str | None:
    """Extract the YouTube video ID from a URL. Returns None if not YouTube."""
    # youtu.be/<id>
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    # youtube.com/watch?v=<id>
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    # youtube.com/shorts/<id>
    m = re.search(r"youtube\.com/shorts/([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    return None


def slugify(text: str) -> str:
    """Make a filesystem-safe slug from arbitrary text."""
    text = Path(text).stem
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()[:64]


def derive_video_id(source: str, explicit_id: str | None) -> str:
    """Determine the video ID for naming output folders."""
    if explicit_id:
        return explicit_id
    if source.startswith("http://") or source.startswith("https://"):
        yt_id = extract_youtube_id(source)
        if yt_id:
            return yt_id
        return slugify(source)
    return slugify(source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="YouTube URL or local video path")
    parser.add_argument("--id", default=None,
                       help="Video ID for output folder naming (default: auto-derive from URL/filename)")
    parser.add_argument("--fps", type=int, default=2, help="Frame extraction rate (default 2)")
    parser.add_argument("--model", default=os.environ.get("VISION_MODEL", "gemini-3-flash-preview"),
                       help="Vision model for phase 2 (default: $VISION_MODEL or gemini-3-flash-preview)")
    parser.add_argument("--whisper-model", default="small", help="Whisper model: tiny/base/small/medium (default small multilingual)")
    parser.add_argument("--whisper-language", default=None,
                       help="Force Whisper language (e.g. 'en', 'de'); default = auto-detect")
    parser.add_argument("--skip-phase2", action="store_true",
                       help="Skip the vision model phase (no API calls, faster but no captions/emotions)")
    parser.add_argument("--skip-phase5", action="store_true",
                       help="Skip the CLAP audio phase (no model download, faster)")
    parser.add_argument("--start-from", type=int, default=0,
                       help="Skip phases before this number (0-8). Useful to resume.")
    args = parser.parse_args()

    video_id = derive_video_id(args.source, args.id)
    out_dir = REPO_ROOT / "data" / video_id
    reports_dir = REPO_ROOT / "reports" / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] video_id: {video_id}")
    print(f"[info] output dir: {out_dir.relative_to(REPO_ROOT)}")
    print(f"[info] reports dir: {reports_dir.relative_to(REPO_ROOT)}")
    print(f"[info] source: {args.source}")
    print(f"[info] fps: {args.fps}, model: {args.model}")
    print()

    # Build env so all phase scripts write to the chosen folder
    env = os.environ.copy()
    env["PROCESSED_DIR"] = str(out_dir)
    env["REPORTS_DIR"] = str(reports_dir)

    # Phase 0 needs the video file in a known place. If source is URL, let phase0 download.
    # If local, we could symlink/copy, but phase0 also accepts a local path directly.
    video_arg = args.source
    if not args.source.startswith("http"):
        # Local file — pass absolute path
        video_arg = str(Path(args.source).resolve())

    phases = [
        (0, f'phase0_input.py "{video_arg}" --fps {args.fps}'),
        (1, "phase1_shots.py --threshold 35 --min-scene-len 30"),
        (2, f"phase2_vision.py --model {args.model}"),
        (3, "phase3_camera.py"),
        (4, f"phase4_transcribe.py --model {args.whisper_model}" + (f" --language {args.whisper_language}" if args.whisper_language else "")),
        (5, "phase5_audio.py"),
        (6, "phase6_music.py"),
        (7, "phase7_sync.py"),
        (8, "phase8_dashboard.py"),
    ]

    py = sys.executable
    for n, cmd in phases:
        if n < args.start_from:
            print(f"[skip] phase {n} (start_from={args.start_from})")
            continue
        if n == 2 and args.skip_phase2:
            print(f"[skip] phase 2 (--skip-phase2)")
            continue
        if n == 5 and args.skip_phase5:
            print(f"[skip] phase 5 (--skip-phase5)")
            continue
        print(f"\n========== phase {n} ==========")
        full_cmd = f'cd "{REPO_ROOT}" && "{py}" scripts/{cmd}'
        # Phase 1 needs the video path — pass it from phase 0 metadata if we don't have it
        if n == 1:
            # Find the raw video in data/raw/<video_id>.mp4 (set by phase 0)
            raw_path = REPO_ROOT / "data" / "raw" / f"{video_id}.mp4"
            if raw_path.exists():
                full_cmd = f'cd "{REPO_ROOT}" && "{py}" scripts/phase1_shots.py --threshold 35 --min-scene-len 30 --video "{raw_path}"'
            else:
                print(f"[warn] raw video not found at {raw_path}, phase 1 may fail")
        result = subprocess.run(full_cmd, shell=True, env=env)
        if result.returncode != 0:
            print(f"\n[error] phase {n} failed (returncode {result.returncode})")
            return n

    dashboard = reports_dir / "dashboard.html"
    print(f"\n[ok] all phases done.")
    print(f"[ok] data:   {out_dir.relative_to(REPO_ROOT)}/")
    print(f"[ok] report: {dashboard.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())