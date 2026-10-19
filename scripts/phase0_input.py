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
from _tools import ffmpeg_exe, ffprobe_exe

VIDEO_EXTS = [".mp4", ".webm", ".mkv", ".mov", ".avi"]



def have_ffmpeg() -> bool:
    exe = ffmpeg_exe()
    if not exe:
        return False
    try:
        subprocess.run([exe, "-version"], capture_output=True, check=True, timeout=10)
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


def get_video(source: str, video_id: str) -> tuple[Path, str | None]:
    """Stage the video at data/raw/<video_id>.<ext> and return (path, source_url).

    source_url is the original URL for downloads (recorded in metadata.json so
    the dashboard can deep-link to video timestamps — UI audit U2), else None
    for local files.
    """
    if source.startswith("http://") or source.startswith("https://"):
        RAW.mkdir(parents=True, exist_ok=True)
        out_tpl = RAW / f"{video_id}.%(ext)s"
        print(f"[info] downloading {source} via yt-dlp ...")
        # Up to 720p video + best audio, merged with our ffmpeg; falls back to
        # a progressive single file (often only 360p) when merging is impossible
        # --no-playlist: a link copied from a playlist (…&list=…) otherwise
        # downloads every video in it (audit R9)
        cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist", "-o", str(out_tpl)]
        ff = ffmpeg_exe()
        if ff:
            cmd += ["--ffmpeg-location", ff,
                    "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
                    "--merge-output-format", "mp4"]
        else:
            cmd += ["-f", "best[ext=mp4]/best"]
        try:
            subprocess.run(cmd + [source], check=True, timeout=3600)
        except FileNotFoundError:
            print("[warn] yt-dlp not installed; install with `pip install yt-dlp`")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"[error] yt-dlp failed: {e}")
            sys.exit(1)
        return resolve_downloaded(video_id, RAW), source
    return stage_local(source, video_id, RAW), None


def extract_metadata(video: Path) -> dict:
    """Extract video metadata with ffprobe, or OpenCV when ffprobe is absent."""
    probe = ffprobe_exe()
    if probe is None:
        return extract_metadata_cv2(video)
    cmd = [
        probe, "-v", "quiet", "-print_format", "json",
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


def extract_metadata_cv2(video: Path) -> dict:
    """ffprobe-free fallback (imageio-ffmpeg ships no ffprobe)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ").lower() or None
    meta = {
        "source_file": str(video),
        "duration_sec": (n / fps) if fps else 0.0,
        "size_bytes": video.stat().st_size,
        "bit_rate": 0,
        "video": {
            "codec": codec,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None,
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None,
            "fps": fps,
            "nb_frames": n or None,
        },
        "probe": "opencv",
    }
    cap.release()
    return meta


# Two audio files: speech models want 16 kHz; music models (CLAP @48 kHz,
# librosa spectral features) need the full band — upsampling the 16 kHz file
# leaves nothing above 8 kHz (methodology review R2).
AUDIO_SPEECH = ("audio.wav", 16000)
AUDIO_MUSIC = ("audio_music.wav", 48000)


def media_timeout(duration_sec: float) -> int:
    """Subprocess timeout that grows with the video: 5 min, or 2x the
    duration for long inputs (audit R9: a fixed 300 s killed ffmpeg on
    feature-length films)."""
    return int(max(300, 2 * (duration_sec or 0)))


def extract_audio(video: Path, out_wav: Path, sample_rate: int = 16000,
                  timeout: int = 300) -> None:
    """Extract mono PCM audio at the given sample rate."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_exe() or "ffmpeg", "-y", "-i", str(video),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-acodec", "pcm_s16le",
        "-loglevel", "error", str(out_wav)
    ]
    subprocess.run(cmd, check=True, timeout=timeout)
    print(f"[ok] extracted {sample_rate} Hz audio to {disp(out_wav)} ({out_wav.stat().st_size:,} bytes)")


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
        print(f"[info] cleared {len(stale)} stale frames from {disp(out_dir)}/")
    cmd = [
        ffmpeg_exe() or "ffmpeg", "-y", "-i", str(video),
        "-vf", f"fps={fps}",
        "-q:v", "2",  # high quality JPEG
        "-loglevel", "error",
        str(out_dir / "frame_%05d.jpg")
    ]
    subprocess.run(cmd, check=True, timeout=timeout)
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

    video, source_url = get_video(args.source, args.id)
    print(f"[ok] video: {disp(video)} ({video.stat().st_size:,} bytes)")

    meta = extract_metadata(video)
    if source_url:
        meta["source_url"] = source_url  # dashboard deep-links (UI audit U2)
    print(f"[info] duration: {meta['duration_sec']:.1f}s, "
          f"{meta['video']['width']}x{meta['video']['height']}, "
          f"{meta['video']['fps']:.2f}fps" if meta['video']['fps'] else "")

    # extract audio + frames in parallel (sequential here for simplicity)
    audio_wav = PROCESSED / AUDIO_SPEECH[0]
    tmo = media_timeout(meta.get("duration_sec", 0))
    extract_audio(video, audio_wav, AUDIO_SPEECH[1], timeout=tmo)
    music_wav = PROCESSED / AUDIO_MUSIC[0]
    extract_audio(video, music_wav, AUDIO_MUSIC[1], timeout=tmo)
    n_frames = extract_frames(video, FRAMES_DIR, args.fps, timeout=tmo)
    meta["frames_extracted"] = n_frames
    meta["frame_fps"] = args.fps
    meta["audio_path"] = disp(audio_wav)
    meta["audio_music_path"] = disp(music_wav)
    meta["frames_dir"] = disp(FRAMES_DIR) + "/"

    meta_path = PROCESSED / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {disp(meta_path)}")

    print("\n[next] Phase 1: python scripts/phase1_shots.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
