"""Shared paths, constants and helpers for the pipeline phases.

Every phase imports from here instead of re-deriving paths, tag lists and
frame-timing rules, so they cannot drift apart.

Output folders:
  PROCESSED  — per-video data (env PROCESSED_DIR, default data/processed)
  REPORTS    — per-video dashboard (env REPORTS_DIR, default reports)

Provenance: each phase calls ``record_run`` at the end. It stores the phase's
arguments, parameters, git commit and a fingerprint of every input and output
file in ``PROCESSED/run_info.json``. ``run_pipeline.py`` uses that record to
skip phases whose inputs and arguments have not changed.
"""
from __future__ import annotations
import hashlib, json, math, os, platform, subprocess, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW = DATA_DIR / "raw"
PROCESSED = Path(os.environ["PROCESSED_DIR"]) if "PROCESSED_DIR" in os.environ else DATA_DIR / "processed"
REPORTS = Path(os.environ["REPORTS_DIR"]) if "REPORTS_DIR" in os.environ else REPO_ROOT / "reports"
FRAMES_DIR = PROCESSED / "frames"
RUN_INFO = PROCESSED / "run_info.json"

# ---------------------------------------------------------------------------
# CLAP vocabulary (phase 5). Each group gets its own softmax, so mood scores
# are not diluted by section/instrument prompts.
MOOD_TAGS = [
    "happy and bright", "sad and melancholic", "aggressive and intense",
    "romantic and tender", "triumphant and epic", "calm and peaceful",
    "tense and anxious", "dreamy and ethereal", "dark and ominous",
    "playful and whimsical", "lonely and introspective", "powerful and confident",
]
SECTION_TAGS = [
    "intro", "verse", "chorus", "bridge", "outro", "instrumental break", "vocal only",
]
INSTRUMENT_TAGS = [
    "acoustic guitar", "electric guitar", "piano", "drums and percussion",
    "bass guitar", "synthesizer", "strings orchestra", "vocal only no instruments",
]
TAG_GROUPS = {"mood": MOOD_TAGS, "section": SECTION_TAGS, "instrument": INSTRUMENT_TAGS}
ALL_TAGS = MOOD_TAGS + SECTION_TAGS + INSTRUMENT_TAGS

# Camera vocabulary shared by phase 3 (optical flow) and phase 2 (VLM).
CAMERA_LABELS = ["static", "pan", "tilt", "zoom-in", "zoom-out", "handheld"]

# "Cut on beat" tolerances reported by phase 7 (seconds). The first one is the
# headline number shown in the dashboard.
BEAT_TOLERANCES_SEC = [0.1, 0.05, 0.2]


# ---------------------------------------------------------------------------
# Metadata and frame timing

def load_metadata(processed: Path | None = None) -> dict:
    """Phase 0's metadata.json (empty dict if missing)."""
    path = (processed or PROCESSED) / "metadata.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def frame_fps(processed: Path | None = None) -> float:
    """Rate at which phase 0 extracted frames (falls back to 1 fps)."""
    return float(load_metadata(processed).get("frame_fps") or 1)


def frame_for_time(t: float, fps: float, n_extracted: int = 0) -> int:
    """1-indexed frame_%05d.jpg number nearest to time t.

    ffmpeg's fps filter emits output frame k (0-indexed) at t = k / fps and
    names it frame_{k+1}.
    """
    idx = int(round(t * fps)) + 1
    if n_extracted:
        idx = min(idx, n_extracted)
    return max(1, idx)


def frames_in_range(start: float, end: float, fps: float) -> range:
    """1-indexed frame numbers whose timestamps fall in [start, end)."""
    return range(math.ceil(start * fps) + 1, math.ceil(end * fps) + 1)


def frame_path(idx: int, frames_dir: Path | None = None) -> Path:
    return (frames_dir or FRAMES_DIR) / f"frame_{idx:05d}.jpg"


def count_frames(frames_dir: Path | None = None) -> int:
    return len(list((frames_dir or FRAMES_DIR).glob("frame_*.jpg")))


def display_path(p: Path | str) -> str:
    """Repo-relative path when possible (sources may live outside the repo)."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# Provenance and change detection

def fingerprint(path: Path) -> list | None:
    """Cheap change detector: [size, mtime_ns] for files, [n, newest mtime] for dirs."""
    path = Path(path)
    if not path.exists():
        return None
    if path.is_dir():
        files = [f for f in path.iterdir() if f.is_file()]
        return [len(files), max((f.stat().st_mtime_ns for f in files), default=0)]
    st = path.stat()
    return [st.st_size, st.st_mtime_ns]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=5)
        commit = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=REPO_ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
        return commit + ("-dirty" if dirty else "") if commit else "unknown"
    except Exception:
        return "unknown"


def code_hashes() -> dict[str, str]:
    """sha1 of every module from scripts/ that the running phase has imported.

    Stored with each run, so editing a phase (or a helper it uses) re-runs that
    phase and, through changed outputs, everything downstream — while editing
    an unrelated script does not re-run expensive phases like the vision model.
    """
    scripts = Path(__file__).resolve().parent
    files = {Path(sys.argv[0]).resolve()} if sys.argv and sys.argv[0].endswith(".py") else set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f and Path(f).resolve().parent == scripts:
            files.add(Path(f).resolve())
    return {display_path(p): hashlib.sha1(p.read_bytes()).hexdigest()
            for p in sorted(files) if p.exists() and p.parent == scripts}


def load_run_info(processed: Path | None = None) -> dict:
    path = (processed or PROCESSED) / "run_info.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def record_run(phase: int, inputs: list[Path], outputs: list[Path],
               params: dict | None = None, argv: list[str] | None = None) -> None:
    """Store what this phase ran with, so results are traceable and re-runs skippable."""
    info = load_run_info()
    info[f"phase{phase}"] = {
        "argv": list(sys.argv[1:] if argv is None else argv),
        "params": params or {},
        "inputs": {display_path(p): fingerprint(p) for p in inputs},
        "outputs": {display_path(p): fingerprint(p) for p in outputs},
        "code": code_hashes(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    RUN_INFO.parent.mkdir(parents=True, exist_ok=True)
    RUN_INFO.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def is_up_to_date(phase: int, argv: list[str], processed: Path | None = None) -> tuple[bool, str]:
    """True if phase ran with the same argv and its inputs/outputs are unchanged."""
    rec = load_run_info(processed).get(f"phase{phase}")
    if not rec:
        return False, "never ran"
    if rec.get("argv") != list(argv):
        return False, "arguments changed"
    if "code" not in rec:
        return False, "no code version recorded"
    for p, digest in rec["code"].items():
        path = REPO_ROOT / p
        if not path.exists() or hashlib.sha1(path.read_bytes()).hexdigest() != digest:
            return False, f"code changed: {p}"
    for kind in ("inputs", "outputs"):
        for p, fp in rec.get(kind, {}).items():
            path = Path(p) if Path(p).is_absolute() else REPO_ROOT / p
            if fingerprint(path) != fp:
                return False, f"{kind[:-1]} changed: {p}"
    return True, "up to date"
