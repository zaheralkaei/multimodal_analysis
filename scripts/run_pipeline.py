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
import argparse, json, os, re, subprocess, sys, urllib.parse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
from _paths import disp
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


def resolve_staged_video(video_id: str, out_dir: Path) -> str | None:
    """Return the staged video path after phase 0.

    Prefers metadata.json's source_file (written by phase 0); falls back to
    data/raw/<video_id>.<ext> which phase 0 also guarantees.
    """
    meta_path = out_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            src = Path(meta.get("source_file", ""))
            if src.exists():
                return str(src.resolve())
        except (json.JSONDecodeError, KeyError):
            pass
    raw_dir = REPO_ROOT / "data" / "raw"
    for ext in (".mp4", ".webm", ".mkv", ".mov", ".avi"):
        cand = raw_dir / f"{video_id}{ext}"
        if cand.exists():
            return str(cand.resolve())
    return None


def _finish(run: dict, out_dir: Path, video_id: str) -> Path:
    from _provenance import finish_run
    vp = resolve_staged_video(video_id, out_dir)
    return finish_run(run, out_dir, Path(vp) if vp else None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="YouTube URL or local video path")
    parser.add_argument("--id", default=None,
                       help="Video ID for output folder naming (default: auto-derive from URL/filename)")
    parser.add_argument("--fps", type=int, default=2, help="Frame extraction rate (default 2)")
    parser.add_argument("--model", default=os.environ.get("VISION_MODEL", "google/gemma-4-31b-it"),
                       help="Vision model for phase 2 (default: $VISION_MODEL or google/gemma-4-31b-it via OpenRouter)")
    parser.add_argument("--provider", default="auto", choices=["auto", "openrouter", "ollama", "mock"],
                       help="Vision backend for phase 2 (default: auto — OpenRouter if OPENROUTER_API_KEY set, else Ollama)")
    parser.add_argument("--whisper-model", default="small", help="Whisper model: tiny/base/small/medium (default small multilingual)")
    parser.add_argument("--whisper-language", default=None,
                       help="Force Whisper language (e.g. 'en', 'de'); default = auto-detect")
    parser.add_argument("--skip-phase2", action="store_true",
                       help="Skip the vision model phase (no API calls, faster but no captions/emotions)")
    parser.add_argument("--skip-phase5", action="store_true",
                       help="Skip the CLAP audio phase (no model download, faster)")
    # UI audit round 6: content-type drives the dashboard's emphasis + layout
    parser.add_argument("--content-type", default="auto",
                        choices=["auto", "song", "speech", "film", "series", "vlog", "other"],
                        help="Dashboard emphasis (song/speech/film/vlog/...). Default: auto-detect.")
    parser.add_argument("--shot-threshold", type=float, default=35.0,
                        help="PySceneDetect ContentDetector threshold (default 35; lower = more cuts)")
    parser.add_argument("--min-scene-len", type=int, default=15,
                        help="Minimum shot length in native frames (default 15)")
    parser.add_argument("--flow-fps", type=float, default=8.0,
                        help="Phase 3 sampling rate inside each shot (default 8)")
    parser.add_argument("--vision-dry-run", action="store_true",
                        help="Phase 2: estimate cost / check model slug only, no paid calls")
    parser.add_argument("--max-usd", type=float, default=None,
                        help="Phase 2 budget cap in USD (default: $VISION_MAX_USD or 1.0)")
    # Round 11: audio stack (heavy parts on hosted services, light parts local)
    parser.add_argument("--separation", default="auto",
                        choices=["auto", "fal", "local", "passthrough", "off"],
                        help="vocal separation before transcription (auto = fal if FAL_KEY "
                             "and music; passthrough = offline wiring test)")
    parser.add_argument("--asr-backend", default="auto", choices=["auto", "local", "groq"],
                        help="transcription backend (auto = groq if GROQ_API_KEY, else local)")
    parser.add_argument("--beats", default="auto", choices=["auto", "beat_this", "replicate", "librosa"],
                        help="beats/downbeats (auto = Beat This! locally if installed)")
    parser.add_argument("--sections", default="auto", choices=["auto", "replicate", "local", "none"],
                        help="song sections (auto = Replicate All-In-One if REPLICATE_API_TOKEN, else local)")
    parser.add_argument("--offline", action="store_true",
                        help="no paid audio services: separation off, local transcription, "
                             "local sections (vision is controlled by --provider)")
    parser.add_argument("--start-from", type=int, default=0,
                       help="Skip phases before this number (0-8). Useful to resume.")
    args = parser.parse_args()

    if args.offline:
        if args.separation == "auto":
            args.separation = "off"
        if args.asr_backend == "auto":
            args.asr_backend = "local"
        if args.sections == "auto":
            args.sections = "local"
        if args.beats == "replicate":
            args.beats = "auto"
    video_id = derive_video_id(args.source, args.id)
    out_dir = REPO_ROOT / "data" / video_id
    reports_dir = REPO_ROOT / "reports" / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] video_id: {video_id}")
    print(f"[info] output dir: {disp(out_dir)}")
    print(f"[info] reports dir: {disp(reports_dir)}")
    print(f"[info] source: {args.source}")
    print(f"[info] fps: {args.fps}, model: {args.model}")
    print()

    # Build env so all phase scripts write to the chosen folder
    env = os.environ.copy()
    env["PROCESSED_DIR"] = str(out_dir)
    env["REPORTS_DIR"] = str(reports_dir)
    env.setdefault("PYTHONIOENCODING", "utf-8")

    # Phase 0 needs the video file in a known place. If source is URL, let phase0 download.
    # If local, we could symlink/copy, but phase0 also accepts a local path directly.
    video_arg = args.source
    if not args.source.startswith("http"):
        # Local file — pass absolute path
        video_arg = str(Path(args.source).resolve())

    # Argument LISTS, run without a shell (audit R9): the old f-string
    # commands with shell=True let a URL containing a quote break out of the
    # quoting and run arbitrary commands (reachable from the web UI).
    phases = [
        (0, ["phase0_input.py", video_arg, "--id", video_id, "--fps", str(args.fps)]),
        # Round 10: decide early whether there is music (CLAP music/speech/
        # silence on sampled windows) so later phases know what is valid
        (0.5, ["route_content.py"]
              + ([] if args.content_type == "auto" else ["--content-type", args.content_type])
              + (["--no-clap"] if args.skip_phase5 else [])),
        (1, ["phase1_shots.py", "--threshold", str(args.shot_threshold),
             "--min-scene-len", str(args.min_scene_len)]),
        (2, ["phase2_vision.py", "--model", args.model]
            + (["--provider", args.provider] if args.provider != "auto" else [])
            + (["--dry-run"] if args.vision_dry_run else [])
            + (["--max-usd", str(args.max_usd)] if args.max_usd is not None else [])),
        (3, ["phase3_camera.py", "--flow-fps", str(args.flow_fps)]),
        (3.5, ["separate_vocals.py", "--provider", args.separation]),
        (4, ["phase4_transcribe.py", "--model", args.whisper_model, "--backend", args.asr_backend]
            + (["--language", args.whisper_language] if args.whisper_language else [])),
        (5, ["phase5_audio.py"]),
        (6, ["phase6_music.py"]),
        (6.5, ["music_structure.py", "--beats", args.beats, "--sections", args.sections]),
        (7, ["phase7_sync.py"]),
        (8, ["phase8_dashboard.py"]
            + ([] if args.content_type == "auto" else ["--content-type", args.content_type])),
    ]

    from _provenance import new_run
    import time as _time
    run = new_run(sys.argv)

    py = sys.executable
    for n, cmd in phases:
        route_missing = n == 0.5 and not (out_dir / "content_route.json").exists()
        if n < args.start_from and not route_missing:
            print(f"[skip] phase {n} (start_from={args.start_from})")
            continue
        if n == 2 and args.skip_phase2:
            print(f"[skip] phase 2 (--skip-phase2)")
            continue
        if n == 5 and args.skip_phase5:
            print(f"[skip] phase 5 (--skip-phase5)")
            continue
        if n == 0.5:
            print("\n---------- route: what kind of audio is this? ----------")
        elif n == 3.5:
            print("\n---------- separate: isolate the vocals ----------")
        elif n == 6.5:
            print("\n---------- structure: beats, downbeats, sections ----------")
        else:
            print(f"\n========== phase {n} ==========")
        argv = [py, str(SCRIPTS / cmd[0]), *cmd[1:]]
        # Phase 1 needs the video path — read it from the metadata that
        # phase 0 just wrote (round-4 audit fix F3: was a guess of
        # data/raw/<video_id>.mp4, which never existed for local files)
        if n == 1:
            video_path = resolve_staged_video(video_id, out_dir)
            if video_path:
                argv += ["--video", video_path]
            else:
                print("[warn] could not resolve staged video path from metadata.json; "
                      "phase 1 will use its default (data/raw/video.mp4)")
        t0 = _time.time()
        result = subprocess.run(argv, cwd=str(REPO_ROOT), env=env)
        run["phases"].append({"phase": n, "command": ["python", *argv[1:]],
                              "returncode": result.returncode,
                              "seconds": round(_time.time() - t0, 1)})
        if result.returncode != 0:
            print(f"\n[error] phase {n} failed (returncode {result.returncode})")
            run["failed_phase"] = n
            _finish(run, out_dir, video_id)
            # 10 + phase: never 0 — `return n` made a phase-0 failure look
            # like success (exit 0) to callers such as the web UI
            return 10 + int(n)

    manifest = _finish(run, out_dir, video_id)
    dashboard = reports_dir / "dashboard.html"
    print(f"\n[ok] all phases done.")
    print(f"[ok] run manifest: {disp(manifest)}")
    print(f"[ok] data:   {disp(out_dir)}/")
    print(f"[ok] report: {disp(dashboard)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())