"""
Phase 0 — Input prep.
Takes a YouTube URL or local video file and extracts:
  - frames at 2 fps (configurable via --fps; saved as JPEGs in <PROCESSED>/frames/)
  - audio track as 16 kHz mono WAV (<PROCESSED>/audio.wav)
  - metadata JSON (<PROCESSED>/metadata.json)

Why 2 fps? See README.md "Phase 0 — Input preparation" for the full trade-off
table. Short version: 1 fps missed fast camera motion in phase 3 (optical
flow), 5 fps is 4× the storage for marginal gain, 24 fps is overkill.
2 fps gives 0.5s time resolution which catches pan/tilt/zoom in 1-2s shots.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from fractions import Fraction
from pathlib import Path

from common import FRAMES_DIR, PROCESSED, RAW, display_path, record_run


def ffmpeg_timeout(duration_sec: float, per_sec: float, minimum: int) -> int:
    """Timeout that scales with video length (fixed timeouts failed on long videos)."""
    return int(max(minimum, duration_sec * per_sec))


def have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=10)
        return True
    except Exception:
        return False


def get_video(source: str, video_id: str = "video") -> Path:
    """Return path to local video file. Downloads if source is a URL.

    Downloads go to data/raw/<video_id>.mp4. yt-dlp skips the download when
    that file already exists, so each video needs its own name — a shared
    name would silently reuse the previous video.
    """
    RAW.mkdir(parents=True, exist_ok=True)
    if source.startswith("http://") or source.startswith("https://"):
        # try yt-dlp first
        try:
            out = RAW / f"{video_id}.mp4"
            print(f"[info] downloading {source} via yt-dlp ...")
            subprocess.run([sys.executable, "-m", "yt_dlp", "-o", str(out),
                           "-f", "best[ext=mp4]/best", source],
                          check=True, timeout=3600)
            return out
        except FileNotFoundError:
            print("[warn] yt-dlp not installed; install with `pip install yt-dlp`")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"[error] yt-dlp failed: {e}")
            sys.exit(1)
    else:
        p = Path(source)
        if not p.exists():
            print(f"[error] file not found: {p}")
            sys.exit(1)
        return p


def _parse_rate(rate: str | None) -> float | None:
    """Parse an ffprobe rate like '24000/1001' into a float."""
    try:
        r = Fraction(rate)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return float(r) if r else None


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
    return {
        "source_file": str(video.resolve()),
        "duration_sec": float(fmt.get("duration", 0)),
        "size_bytes": int(fmt.get("size", 0)),
        "bit_rate": int(fmt.get("bit_rate", 0)),
        "video": {
            "codec": v.get("codec_name"),
            "width": v.get("width"),
            "height": v.get("height"),
            "fps": _parse_rate(v.get("r_frame_rate")),
            "nb_frames": v.get("nb_frames"),
        },
    }


def extract_audio(video: Path, out_wav: Path, timeout: int = 120) -> None:
    """Extract 16kHz mono PCM audio."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        "-loglevel", "error", str(out_wav)
    ]
    subprocess.run(cmd, check=True, timeout=timeout)
    print(f"[ok] extracted audio to {display_path(out_wav)} ({out_wav.stat().st_size:,} bytes)")


def extract_frames(video: Path, out_dir: Path, fps: int = 1, timeout: int = 300) -> int:
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
        print(f"[info] cleared {len(stale)} stale frames from {display_path(out_dir)}/")
    cmd = [
        "ffmpeg", "-y", "-i", str(video),
        "-vf", f"fps={fps}",
        "-q:v", "2",  # high quality JPEG
        "-loglevel", "error",
        str(out_dir / "frame_%05d.jpg")
    ]
    subprocess.run(cmd, check=True, timeout=timeout)
    frames = sorted(out_dir.glob("frame_*.jpg"))
    print(f"[ok] extracted {len(frames)} frames at {fps}fps to {display_path(out_dir)}/")
    return len(frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="YouTube URL or local video path")
    parser.add_argument("--fps", type=int, default=2, help="frames per second to extract")
    parser.add_argument("--video-id", default="video",
                        help="file name (without .mp4) for URL downloads in data/raw/ (default: video)")
    args = parser.parse_args()

    if not have_ffmpeg():
        print("[error] ffmpeg not found. Install via `choco install ffmpeg` or from https://ffmpeg.org/")
        return 1

    PROCESSED.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    video = get_video(args.source, args.video_id)
    print(f"[ok] video: {video} ({video.stat().st_size:,} bytes)")

    meta = extract_metadata(video)
    print(f"[info] duration: {meta['duration_sec']:.1f}s, "
          f"{meta['video']['width']}x{meta['video']['height']}, "
          f"{meta['video']['fps']:.2f}fps" if meta['video']['fps'] else "")

    # extract audio + frames in parallel (sequential here for simplicity)
    audio_wav = PROCESSED / "audio.wav"
    duration = meta["duration_sec"]
    extract_audio(video, audio_wav, ffmpeg_timeout(duration, 0.5, 120))
    n_frames = extract_frames(video, FRAMES_DIR, args.fps, ffmpeg_timeout(duration, 2.0 * args.fps, 300))
    meta["frames_extracted"] = n_frames
    meta["frame_fps"] = args.fps
    meta["audio_path"] = display_path(audio_wav)
    meta["frames_dir"] = display_path(FRAMES_DIR) + "/"

    meta_path = PROCESSED / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {display_path(meta_path)}")
    record_run(0, inputs=[video], outputs=[meta_path, audio_wav, FRAMES_DIR],
               params={"fps": args.fps, "source": args.source})

    print("\n[next] Phase 1: python scripts/phase1_shots.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
