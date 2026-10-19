"""Locate external media tools (ffmpeg / ffprobe).

ffmpeg is often not on PATH on Windows. Lookup order:
  1. $FFMPEG_BINARY / $FFPROBE_BINARY
  2. PATH (shutil.which)
  3. ffprobe only: a sibling of the resolved ffmpeg binary
  4. ffmpeg only: the binary bundled with the `imageio-ffmpeg` package
     (ships ffmpeg but NOT ffprobe — phase 0 then probes with OpenCV)
"""
from __future__ import annotations
import os, shutil
from pathlib import Path


def ffmpeg_exe() -> str | None:
    env = os.environ.get("FFMPEG_BINARY")
    if env and Path(env).exists():
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass
    return None


def ffprobe_exe() -> str | None:
    env = os.environ.get("FFPROBE_BINARY")
    if env and Path(env).exists():
        return env
    found = shutil.which("ffprobe")
    if found:
        return found
    ff = ffmpeg_exe()
    if ff:
        p = Path(ff)
        for name in ("ffprobe.exe", "ffprobe"):
            cand = p.with_name(name)
            if cand.exists():
                return str(cand)
    return None
