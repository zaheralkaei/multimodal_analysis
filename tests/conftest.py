"""Shared fixtures: synthetic videos whose ground truth is known exactly.

Nothing here downloads models or calls an API, so the suite runs in CI.
Tests that need ffmpeg are skipped when it is not installed.
"""
from __future__ import annotations
import shutil, subprocess, sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

W, H = 320, 240  # output frame size of every synthetic clip


def textured_still(path: Path, channel: int | None = None, seed: int = 0, size=(1280, 720)) -> Path:
    """Random rectangles/circles: lots of trackable corners.

    With ``channel`` set (0=B, 1=G, 2=R), all shapes are drawn in that colour
    channel only, so the frame's dominant channel identifies which shot it
    came from.
    """
    import cv2
    rng = np.random.default_rng(seed)
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = 40
    for _ in range(900):
        level = int(rng.integers(90, 255))
        color = [0, 0, 0]
        if channel is None:
            color = [int(v) for v in rng.integers(0, 255, 3)]
        else:
            color[channel] = level
        x, y, r = int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(4, 30))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), r, color, -1)
        else:
            cv2.rectangle(img, (x, y), (x + r, y + r), color, -1)
    if channel is not None:  # background in the same channel, so the mean is dominated by it
        img[..., channel] = np.maximum(img[..., channel], 60)
    cv2.imwrite(str(path), img)
    return path


# ffmpeg filters that move a 320x240 window over a 1280x720 still
MOTIONS = {
    "static":    "crop=320:240:400:200",
    "pan-right": "crop=320:240:x='200+t*60':y=200",
    "pan-left":  "crop=320:240:x='600-t*60':y=200",
    "tilt-down": "crop=320:240:x=400:y='100+t*40'",
    "tilt-up":   "crop=320:240:x=400:y='400-t*40'",
    "zoom-in":   "zoompan=z='1+0.004*on':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=320x240:fps=24",
    "zoom-out":  "zoompan=z='1.5-0.004*on':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=320x240:fps=24",
    "handheld":  "crop=320:240:x='400+25*sin(t*11)':y='200+25*cos(t*13)'",
}


def make_clip(still: Path, motion: str, seconds: float, out: Path) -> Path:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-framerate", "24",
                    "-t", str(seconds), "-i", str(still), "-vf", MOTIONS[motion],
                    "-pix_fmt", "yuv420p", "-r", "24", str(out)], check=True)
    return out


def extract_frames(video: Path, out_dir: Path, fps: float = 2) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vf", f"fps={fps}",
                    "-q:v", "2", str(out_dir / "frame_%05d.jpg")], check=True)
    return out_dir


def click_track(path: Path, seconds: float, bpm: float = 120, sr: int = 16000) -> Path:
    """C-major chord with a click on every beat (beat 0 at t=0)."""
    import soundfile as sf
    t = np.arange(int(sr * seconds)) / sr
    y = 0.08 * sum(np.sin(2 * np.pi * f * t) for f in (261.63, 329.63, 392.0))
    n = 400
    click = 0.8 * np.exp(-np.arange(n) / 60) * np.sin(2 * np.pi * 2000 * np.arange(n) / sr)
    for b in np.arange(0, seconds, 60 / bpm):
        i = int(b * sr)
        y[i:i + n] += click[: len(y[i:i + n])]
    sf.write(str(path), y.astype("float32"), sr)
    return path


# The smoke-test video: 8 shots of 2.5 s, cuts every 2.5 s = exactly on the
# 120 BPM beat grid. Each shot uses one colour channel and one camera motion.
SMOKE_SHOTS = [  # (dominant channel, motion)
    (2, "static"), (1, "pan-right"), (0, "tilt-down"), (2, "zoom-in"),
    (1, "pan-left"), (0, "static"), (2, "tilt-up"), (1, "zoom-out"),
]
SMOKE_SHOT_SEC = 2.5


@pytest.fixture(scope="session")
def smoke_video(tmp_path_factory) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("smoke")
    clips = []
    for i, (channel, motion) in enumerate(SMOKE_SHOTS):
        still = textured_still(d / f"still_{i}.png", channel=channel, seed=i)
        clips.append(make_clip(still, motion, SMOKE_SHOT_SEC, d / f"clip_{i}.mp4"))
    listing = d / "clips.txt"
    listing.write_text("".join(f"file '{c}'\n" for c in clips))
    silent = d / "video_only.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c", "copy", str(silent)], check=True)
    audio = click_track(d / "music.wav", SMOKE_SHOT_SEC * len(SMOKE_SHOTS))
    out = d / "Smoke Test (synthetic).mp4"  # spaces and parentheses on purpose
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(silent), "-i", str(audio),
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)], check=True)
    return out
