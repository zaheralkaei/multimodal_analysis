"""Run manifest for reproducible analyses (round 8).

run_pipeline.py appends one entry per run to data/<video_id>/run_manifest.json:
git commit (+ dirty flag), Python/platform, versions of every analysis
package, the full CLI arguments, each phase's exact command line, exit code
and wall time, and the SHA-256 of the input video. Model names, prompts,
thresholds and parameters live in each phase's own *_stats.json file, which
the manifest lists — together they are what a methods section needs.
"""
from __future__ import annotations
import datetime, hashlib, json, platform, subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PACKAGES = ["numpy", "scipy", "opencv-python-headless", "scenedetect", "librosa",
            "soundfile", "faster-whisper", "ctranslate2", "transformers", "torch",
            "plotly", "pandas", "yt-dlp", "imageio-ffmpeg"]

STATS_FILES = ["metadata.json", "shot_detection_stats.json", "shot_vision_stats.json",
               "shot_camera_stats.json", "transcript_stats.json", "audio_clap_stats.json",
               "music_summary.json", "sync_stats.json"]


def package_versions() -> dict:
    from importlib import metadata
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    commit = run("rev-parse", "HEAD")
    dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
    return {"commit": commit or None, "dirty": dirty}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def new_run(argv: list[str]) -> dict:
    return {
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "argv": argv,
        "git": git_state(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "phases": [],
    }


def finish_run(run: dict, processed: Path, video: Path | None) -> Path:
    run["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    if video is not None:
        run["input_video"] = {"path": str(video), "sha256": sha256_file(video)}
    run["stats_files"] = [f for f in STATS_FILES if (processed / f).exists()]
    path = processed / "run_manifest.json"
    runs = []
    if path.exists():
        try:
            runs = json.loads(path.read_text(encoding="utf-8")).get("runs", [])
        except (json.JSONDecodeError, AttributeError):
            runs = []
    runs.append(run)
    path.write_text(json.dumps({"runs": runs}, indent=2) + "\n", encoding="utf-8")
    return path
