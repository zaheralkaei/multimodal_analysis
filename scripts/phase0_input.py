"""
Phase 0 — Input prep.
Takes a YouTube URL or local video file, stages it at data/raw/<video_id>.mp4
(round-4 audit fix: the old code always downloaded to data/raw/video.mp4, which
broke the per-video naming contract and made runs overwrite each other), then
extracts:
  - frames at 2 fps (configurable via --fps; saved as JPEGs in data/<video_id>/frames/)
  - audio track as 16 kHz mono WAV (data/<video_id>/audio.wav)
  - metadata JSON (data/<video_id>/metadata.json)

Why 2 fps? See README.md "Phase 0 — Input preparation" for the full trade-off
table. Short version: 1 fps missed fast camera motion in phase 3 (optical
flow), 5 fps is 4× the storage for marginal gain, 24 fps is overkill.
2 fps gives 0.5s time resolution which catches pan/tilt/zoom in 1-2s shots.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, fractions
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED = DATA_DIR / "processed"
if "PROCESSED_DIR" in os.environ:
    PROCESSED = Path(os.environ["PROCESSED_DIR"])
RAW = DATA_DIR / "raw"
FRAMES_DIR = PROCESSED / "frames"

from _paths import disp

VIDEO_EXTS = [".mp4", ".webm", ".mkv", ".mov", ".avi"]



def have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=10)
        return True
    except Exception:
        return False


def stage_local(source: str, video_id: str, raw_dir: Path) -> Path:
    """Copy/symlink a local video into data/raw/<video_id>.<ext> (audit F3).

    If the file already lives in RAW under the right name, it is returned
    as-is (no copy). Returns the staged path.
    """
    p = Path(source).resolve()
    if not p.exists():
        print(f"[error] file not found: {p}")
        sys.exit(1)
    staged = raw_dir / f"{video_id}{p.suffix or '.mp4'}"
    if p == staged:
        return p
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, staged)
    print(f"[info] staged local video at {disp(staged)}")
    return staged


def resolve_downloaded(video_id: str, raw_dir: Path) -> Path:
    """Find the file yt-dlp wrote for <video_id>.%(ext)s (audit F2: the
    best format's real extension is used by yt-dlp, so don't assume .mp4)."""
    candidates = []
    for ext in VIDEO_EXTS:
        f = raw_dir / f"{video_id}{ext}"
        if f.exists():
            candidates.append(f)
    if not candidates:
        print(f"[error] download finished but no file found for {video_id}.* in {raw_dir}")
        sys.exit(1)
    # Deterministic: prefer mp4 (matches the -f best[ext=mp4]/best intent),
    # then webm, etc. in VIDEO_EXTS order
    return candidates[0]


def get_video(source: str, video_id: str) -> Path:
    """Stage the video at data/raw/<video_id>.<ext> and return that path."""
    if source.startswith("http://") or source.startswith("https://"):
        RAW.mkdir(parents=True, exist_ok=True)
        out_tpl = RAW / f"{video_id}.%(ext)s"
        print(f"[info] downloading {source} via yt-dlp ...")
        try:
            subprocess.run([sys.executable, "-m", "yt_dlp", "-o", str(out_tpl),
                            "-f", "best[ext=mp4]/best", source],
                           check=True, timeout=600)
        except FileNotFoundError:
            print("[warn] yt-dlp not installed; install with `pip install yt-dlp`")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"[error] yt-dlp failed: {e}")
            sys.exit(1)
        return resolve_downloaded(video_id, RAW)
    return stage_local(source, video_id, RAW)


def extract_metadata(video: Path) -> dict:
    """Use ffprobe to extract video metadata."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(video)
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
    meta = json.loads(out.stdout)
    # Find video stream
    v = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), {})
    fmt = meta.get("format", {})
    # r_frame_rate is a rational string like "30000/1001" — parse with
    # Fraction, never eval() (round-4 audit low-severity fix)
    try:
        fps = float(fractions.Fraction(v.get("r_frame_rate", "0/1")))
    except (ValueError, ZeroDivisionError):
        fps = None
    return {
        "source_file": str(video),
        "duration_sec": float(fmt.get("duration", 0)),
        "size_bytes": int(fmt.get("size", 0)),
        "bit_rate": int(fmt.get("bit_rate", 0)),
        "video": {
            "codec": v.get("codec_name"),
            "width": v.get("width"),
            "height": v.get("height"),
            "fps": fps,
            "nb_frames": v.get("nb_frames"),
        },
    }


def extract_audio(video: Path, out_wav: Path) -> None:
    """Extract 16kHz mono PCM audio."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        "-loglevel", "error", str(out_wav)
    ]
    subprocess.run(cmd, check=True, timeout=120)
    print(f"[ok] extracted audio to {disp(out_wav)} ({out_wav.stat().st_size:,} bytes)")


def extract_frames(video: Path, out_dir: Path, fps: int = 1) -> int:
    """Extract one frame every N seconds. Returns frame count.

    Clears any existing frame_*.jpg before extracting to avoid counter
    overlap and stale frames from a previous video.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale frames (from a previous run with different video or fps)
    stale = sorted(out_dir.glob("frame_*.jpg"))
    if stale:
        for f in stale:
            f.unlink()
        print(f"[info] cleared {len(stale)} stale frames from {disp(out_dir)}/")
    cmd = [
        "ffmpeg", "-y", "-i", str(video),
        "-vf", f"fps={fps}",
        "-q:v", "2",  # high quality JPEG
        "-loglevel", "error",
        str(out_dir / "frame_%05d.jpg")
    ]
    subprocess.run(cmd, check=True, timeout=300)
    frames = sorted(out_dir.glob("frame_*.jpg"))
    print(f"[ok] extracted {len(frames)} frames at {fps}fps to {disp(out_dir)}/")
    return len(frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="YouTube URL or local video path")
    parser.add_argument("--fps", type=int, default=2, help="frames per second to extract")
    parser.add_argument("--id", default="video",
                        help="video ID for staging (data/raw/<id>.<ext>) and folder naming; "
                             "run_pipeline always passes this explicitly")
    args = parser.parse_args()

    if not have_ffmpeg():
        print("[error] ffmpeg not found. Install via `choco install ffmpeg` or from https://ffmpeg.org/")
        return 1

    PROCESSED.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    video = get_video(args.source, args.id)
    print(f"[ok] video: {disp(video)} ({video.stat().st_size:,} bytes)")

    meta = extract_metadata(video)
    print(f"[info] duration: {meta['duration_sec']:.1f}s, "
          f"{meta['video']['width']}x{meta['video']['height']}, "
          f"{meta['video']['fps']:.2f}fps" if meta['video']['fps'] else "")

    # extract audio + frames in parallel (sequential here for simplicity)
    audio_wav = PROCESSED / "audio.wav"
    extract_audio(video, audio_wav)
    n_frames = extract_frames(video, FRAMES_DIR, args.fps)
    meta["frames_extracted"] = n_frames
    meta["frame_fps"] = args.fps
    meta["audio_path"] = disp(audio_wav)
    meta["frames_dir"] = disp(FRAMES_DIR) + "/"

    meta_path = PROCESSED / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {disp(meta_path)}")

    print("\n[next] Phase 1: python scripts/phase1_shots.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
