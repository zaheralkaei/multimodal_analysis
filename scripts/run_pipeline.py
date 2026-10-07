"""
Run the entire 8-phase pipeline on a video. Output goes to per-video folders.

Naming convention:
  - data/<video_id>/        — all per-video artifacts (audio, frames, CSVs, JSONs)
  - reports/<video_id>/     — dashboard.html (self-contained)
  - data/raw/<video_id>.mp4 — downloaded video

Where <video_id> is:
  - YouTube ID if source is a YouTube URL (e.g. "rtwpk9rb1Dc" from ?v=rtwpk9rb1Dc)
  - Slug of local filename (without extension) if source is a local path
  - Explicit --id flag overrides

Re-running is cheap: every phase records its arguments and input/output
fingerprints in data/<video_id>/run_info.json, and phases whose arguments and
inputs are unchanged are skipped. When an upstream phase re-runs, its outputs
change, so everything downstream re-runs too. Use --force to re-run anyway.

Usage:
  # YouTube video — ID auto-derived from URL
  python scripts/run_pipeline.py "https://www.youtube.com/watch?v=rtwpk9rb1Dc"

  # Local video with explicit ID
  python scripts/run_pipeline.py ~/videos/my_clip.mp4 --id my_clip

  # Skip the model phases (vision, Whisper, CLAP): no API key or downloads needed
  python scripts/run_pipeline.py URL --skip 2,4,5

  # Better lyrics: isolate vocals with Demucs first (pip install demucs)
  python scripts/run_pipeline.py URL --separate-vocals

  # Re-run everything from phase 4 even if up to date
  python scripts/run_pipeline.py URL --start-from 4 --force
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys
from pathlib import Path

from common import is_up_to_date

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def extract_youtube_id(url: str) -> str | None:
    """Extract the YouTube video ID from a URL. Returns None if not YouTube."""
    for pattern in (r"youtu\.be/([A-Za-z0-9_-]{11})",            # youtu.be/<id>
                    r"[?&]v=([A-Za-z0-9_-]{11})",                 # youtube.com/watch?v=<id>
                    r"youtube\.com/shorts/([A-Za-z0-9_-]{11})"):  # youtube.com/shorts/<id>
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def slugify(text: str) -> str:
    """Make a filesystem-safe slug from arbitrary text."""
    text = Path(text).stem
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()[:64] or "video"


def is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def derive_video_id(source: str, explicit_id: str | None) -> str:
    """Determine the video ID for naming output folders."""
    if explicit_id:
        if not ID_RE.match(explicit_id):
            raise SystemExit(f"[error] --id must match {ID_RE.pattern} (got {explicit_id!r})")
        return explicit_id
    if is_url(source):
        return extract_youtube_id(source) or slugify(source)
    return slugify(source)


def build_phases(args, video_arg: str, video_id: str) -> list[tuple[int, str, list[str]]]:
    """(phase number, script, argument list). Lists, not shell strings, so URLs
    and paths with quotes or $(...) are passed verbatim."""
    phase1 = ["--detector", args.detector, "--min-scene-len", "30"]
    phase4 = ["--model", args.whisper_model]
    if args.whisper_language:
        phase4 += ["--language", args.whisper_language]
    if args.separate_vocals:
        phase4.append("--separate-vocals")
    return [
        (0, "phase0_input.py", [video_arg, "--fps", str(args.fps), "--video-id", video_id]),
        (1, "phase1_shots.py", phase1),  # reads the video path from metadata.json
        (2, "phase2_vision.py", ["--model", args.model, "--workers", str(args.workers)]),
        (3, "phase3_camera.py", []),
        (4, "phase4_transcribe.py", phase4),
        (5, "phase5_audio.py", []),
        (6, "phase6_music.py", []),
        (7, "phase7_sync.py", []),
        (8, "phase8_dashboard.py", []),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="YouTube URL or local video path")
    parser.add_argument("--id", default=None,
                        help="Video ID for output folder naming (default: auto-derive from URL/filename)")
    parser.add_argument("--fps", type=int, default=2, help="Frame extraction rate (default 2)")
    parser.add_argument("--detector", choices=["adaptive", "content"], default="adaptive",
                        help="Shot detector for phase 1 (default adaptive)")
    parser.add_argument("--model", default=os.environ.get("VISION_MODEL", "gemini-3-flash-preview"),
                        help="Vision model for phase 2 (default: $VISION_MODEL or gemini-3-flash-preview)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel vision-model requests (default 4)")
    parser.add_argument("--whisper-model", default="small",
                        help="Whisper model: tiny/base/small/medium/large-v3 (default small, multilingual)")
    parser.add_argument("--whisper-language", default=None,
                        help="Force Whisper language (e.g. 'en', 'de'); default = auto-detect")
    parser.add_argument("--separate-vocals", action="store_true",
                        help="Isolate vocals with Demucs before transcription (needs `pip install demucs`)")
    parser.add_argument("--skip-phase2", action="store_true",
                        help="Skip the vision model phase (no API calls, faster but no captions/emotions)")
    parser.add_argument("--skip-phase5", action="store_true",
                        help="Skip the CLAP audio phase (no model download, faster)")
    parser.add_argument("--skip", default="",
                        help="Comma-separated phase numbers to skip, e.g. --skip 2,4,5 (no model downloads/API calls)")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Skip phases before this number (0-8)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run phases even if their inputs and arguments are unchanged")
    args = parser.parse_args()

    if not is_url(args.source) and not Path(args.source).exists():
        print(f"[error] file not found: {args.source}")
        return 1

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

    # Every phase script writes to the folders named in these variables
    env = os.environ.copy()
    env["PROCESSED_DIR"] = str(out_dir)
    env["REPORTS_DIR"] = str(reports_dir)

    video_arg = args.source if is_url(args.source) else str(Path(args.source).resolve())
    try:
        skip = {int(x) for x in args.skip.split(",") if x.strip()}
    except ValueError:
        print(f"[error] --skip expects phase numbers like 2,4,5 (got {args.skip!r})")
        return 1
    skip |= {n for n, flag in ((2, args.skip_phase2), (5, args.skip_phase5)) if flag}

    py = sys.executable
    for n, script, script_args in build_phases(args, video_arg, video_id):
        if n < args.start_from:
            print(f"[skip] phase {n} (start_from={args.start_from})")
            continue
        if n in skip:
            print(f"[skip] phase {n} (--skip)")
            continue
        if not args.force:
            fresh, why = is_up_to_date(n, script_args, out_dir)
            if fresh:
                print(f"[skip] phase {n} (up to date)")
                continue
            print(f"\n========== phase {n} ({why}) ==========")
        else:
            print(f"\n========== phase {n} (--force) ==========")
        result = subprocess.run([py, str(SCRIPTS / script), *script_args], cwd=REPO_ROOT, env=env)
        if result.returncode != 0:
            print(f"\n[error] phase {n} failed (returncode {result.returncode})")
            return 1

    print("\n[ok] all phases done.")
    print(f"[ok] data:   {out_dir.relative_to(REPO_ROOT)}/")
    print(f"[ok] report: {(reports_dir / 'dashboard.html').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
