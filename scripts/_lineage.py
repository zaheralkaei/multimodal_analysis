"""Lineage: which inputs, settings and code produced each output (round 10).

Every step calls `record()` after writing its outputs. It stores, in
data/<video_id>/lineage.json:

    {step: {"inputs":  {path: sha256},     # files the step read
            "outputs": {path: sha256},     # files the step wrote
            "params":  {...},              # settings that change results
            "code":    sha256 of the script,
            "time":    iso timestamp}}

`check()` re-hashes the files and reports two kinds of inconsistency:

  - stale:       an INPUT changed after the step ran (e.g. phase 1 re-ran
                 with other settings, so phase 2's captions describe an
                 old shot list) -> the step's outputs no longer belong to
                 the current upstream data.
  - overwritten: an OUTPUT changed after the step recorded it (another run
                 or a manual edit replaced it without recording lineage).

Phase 7 refuses to join a stream whose step is stale or overwritten, and the
QA report lists them. Hashes are cached per (path, size, mtime) so a file is
read once per process.
"""
from __future__ import annotations
import datetime, hashlib, json
from pathlib import Path

LINEAGE_FILE = "lineage.json"
REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_PREFIX = "@repo/"

_CACHE: dict[tuple, str] = {}


def file_sha(path) -> str | None:
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return None
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if key in _CACHE:
        return _CACHE[key]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    _CACHE[key] = h.hexdigest()
    return _CACHE[key]


def _key(path, processed: Path) -> str:
    """Portable path: relative to the video folder, else to the repository
    (round 13: absolute paths made every step 'stale' after the project
    folder was moved or cloned), else absolute."""
    p = Path(path).resolve()
    try:
        return p.relative_to(Path(processed).resolve()).as_posix()
    except ValueError:
        pass
    try:
        return REPO_PREFIX + p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(p)


def _resolve(key: str, processed: Path) -> Path:
    if key.startswith(REPO_PREFIX):
        return REPO_ROOT / key[len(REPO_PREFIX):]
    p = Path(key)
    return p if p.is_absolute() else processed / p


def load(processed: Path) -> dict:
    path = Path(processed) / LINEAGE_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def record(processed, step: str, inputs: list, outputs: list,
           params: dict | None = None, script=None) -> dict:
    processed = Path(processed)
    entry = {
        "inputs": {_key(p, processed): file_sha(p) for p in inputs},
        "outputs": {_key(p, processed): file_sha(p) for p in outputs if Path(p).exists()},
        "params": params or {},
        "code": file_sha(script) if script else None,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    data = load(processed)
    data[step] = entry
    (processed / LINEAGE_FILE).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return entry


def check(processed, steps: list[str] | None = None) -> dict:
    """{'has_lineage': bool, 'problems': [{step, kind, file}], 'ok_steps': [...]}"""
    processed = Path(processed)
    data = load(processed)
    out = {"has_lineage": bool(data), "problems": [], "ok_steps": [], "missing": []}
    for step, rec in data.items():
        if steps is not None and step not in steps:
            continue
        bad = False
        for key, sha in rec.get("inputs", {}).items():
            cur = file_sha(_resolve(key, processed))
            if cur is None:
                # deleted (e.g. the raw video removed to save space) is NOT
                # changed: it cannot be verified, but nothing contradicts it
                out["missing"].append({"step": step, "file": key})
            elif cur != sha:
                out["problems"].append({"step": step, "kind": "stale", "file": key})
                bad = True
        for key, sha in rec.get("outputs", {}).items():
            if file_sha(_resolve(key, processed)) != sha:
                out["problems"].append({"step": step, "kind": "overwritten", "file": key})
                bad = True
        if not bad:
            out["ok_steps"].append(step)
    return out


def bad_steps(result: dict) -> set[str]:
    return {p["step"] for p in result.get("problems", [])}
