"""
Phase 0 — Input prep.
Takes a YouTube URL or local video file and extracts:
  - frames at 1 fps (saved as PNGs in data/processed/frames/)
  - audio track as 16kHz mono WAV (data/processed/audio.wav)
  - metadata JSON (data/processed/metadata.json)
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED = DATA_DIR / "processed"
RAW = DATA_DIR / "raw"
FRAMES_DIR = PROCESSED / "frames"


def have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=10)
        return True
    except Exception:
        return False


def get_video(source: str) -> Path:
    """Return path to local video file. Downloads if source is a URL."""
    RAW.mkdir(parents=True, exist_ok=True)
    if source.startswith("http://") or source.startswith("https://"):
        # try yt-dlp first
        try:
            out = RAW / "video.mp4"
            print(f"[info] downloading {source} via yt-dlp ...")
            subprocess.run(["yt-dlp", "-o", str(out), "-f", "best[ext=mp4]/best", source],
                          check=True, timeout=600)
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
        "source_file": str(video),
        "duration_sec": float(fmt.get("duration", 0)),
        "size_bytes": int(fmt.get("size", 0)),
        "bit_rate": int(fmt.get("bit_rate", 0)),
        "video": {
            "codec": v.get("codec_name"),
            "width": v.get("width"),
            "height": v.get("height"),
            "fps": eval(v.get("r_frame_rate", "0/1")) if v.get("r_frame_rate") else None,
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
    print(f"[ok] extracted audio to {out_wav.relative_to(REPO_ROOT)} ({out_wav.stat().st_size:,} bytes)")


def extract_frames(video: Path, out_dir: Path, fps: int = 1) -> int:
    """Extract one frame every N seconds. Returns frame count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video),
        "-vf", f"fps={fps}",
        "-q:v", "2",  # high quality JPEG
        "-loglevel", "error",
        str(out_dir / "frame_%05d.jpg")
    ]
    subprocess.run(cmd, check=True, timeout=300)
    frames = sorted(out_dir.glob("frame_*.jpg"))
    print(f"[ok] extracted {len(frames)} frames at {fps}fps to {out_dir.relative_to(REPO_ROOT)}/")
    return len(frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="YouTube URL or local video path")
    parser.add_argument("--fps", type=int, default=1, help="frames per second to extract")
    args = parser.parse_args()

    if not have_ffmpeg():
        print("[error] ffmpeg not found. Install via `choco install ffmpeg` or from https://ffmpeg.org/")
        return 1

    PROCESSED.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    video = get_video(args.source)
    print(f"[ok] video: {video.relative_to(REPO_ROOT)} ({video.stat().st_size:,} bytes)")

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
    meta["audio_path"] = str(audio_wav.relative_to(REPO_ROOT))
    meta["frames_dir"] = str(FRAMES_DIR.relative_to(REPO_ROOT)) + "/"

    meta_path = PROCESSED / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {meta_path.relative_to(REPO_ROOT)}")

    print("\n[next] Phase 1: python scripts/phase1_shots.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
