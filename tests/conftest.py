"""Shared test helpers: load scripts/ modules with custom PROCESSED_DIR env.

Phase scripts read PROCESSED_DIR / REPORTS_DIR at import time, so tests must
set the env var BEFORE importing. `load_phase()` handles that (and reloads if
the module was already imported with different env).
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_phase(name: str, tmp_path: Path | None = None, module_name: str | None = None):
    """Import a scripts/ module, optionally pointed at a tmp PROCESSED_DIR."""
    if tmp_path is not None:
        os.environ["PROCESSED_DIR"] = str(tmp_path)
    mod = importlib.import_module(module_name or name)
    return importlib.reload(mod)


@pytest.fixture()
def tmp_processed(tmp_path):
    """A tmp dir standing in for data/<video_id>/, with env wired."""
    proc = tmp_path / "data" / "vid1"
    proc.mkdir(parents=True)
    old = os.environ.get("PROCESSED_DIR")
    os.environ["PROCESSED_DIR"] = str(proc)
    yield proc
    if old is None:
        os.environ.pop("PROCESSED_DIR", None)
    else:
        os.environ["PROCESSED_DIR"] = old


@pytest.fixture()
def fake_metadata(tmp_processed):
    """Standard 2fps metadata.json like phase0 writes, plus a frames dir."""
    import json

    frames = tmp_processed / "frames"
    frames.mkdir()
    # 100 frames at 2 fps = 50 s of video
    for i in range(1, 101):
        (frames / f"frame_{i:05d}.jpg").write_bytes(b"\xff\xd8fake")
    meta = {
        "source_file": "data/raw/vid1.mp4",
        "duration_sec": 50.0,
        "video": {"codec": "h264", "width": 320, "height": 240, "fps": 24.0},
        "frames_extracted": 100,
        "frame_fps": 2,
    }
    (tmp_processed / "metadata.json").write_text(json.dumps(meta))
    return tmp_processed