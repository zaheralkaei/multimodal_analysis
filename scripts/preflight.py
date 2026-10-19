"""
Preflight - check that a real (paid) run can succeed, without spending money.

    python scripts/preflight.py                      # offline checks
    python scripts/preflight.py --online             # + OpenRouter model/price check (free)
    python scripts/preflight.py --online --id rtwpk9rb1Dc   # + cost for that video's shots

Checks:
  - Python version and every analysis package (import + version)
  - ffmpeg (PATH / FFMPEG_BINARY / imageio-ffmpeg) and ffprobe (optional)
  - free disk space; local model caches (CLAP, Whisper) - else a download
  - .env: OPENROUTER_API_KEY present (never printed), VISION_MODEL
  - --online: the model slug exists on OpenRouter, accepts IMAGE input, and
    its price; an upper-bound cost for N shots (from data/<id>/shots.json or
    --shots). Uses only the public GET /models endpoint - no key, no charge.

Exit code 0 = ready, 1 = a blocking problem was found.
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _paths import REPO_ROOT, disp
from _env import load_env
from _tools import ffmpeg_exe, ffprobe_exe

REQUIRED = {  # import name -> pip name
    "numpy": "numpy", "pandas": "pandas", "cv2": "opencv-python-headless",
    "scenedetect": "scenedetect", "librosa": "librosa", "soundfile": "soundfile",
    "faster_whisper": "faster-whisper", "transformers": "transformers",
    "torch": "torch", "plotly": "plotly", "PIL": "Pillow", "flask": "Flask",
    "yt_dlp": "yt-dlp",
}

OK, WARN, FAIL = "ok", "warn", "FAIL"


class Report:
    def __init__(self):
        self.items: list[tuple[str, str, str]] = []

    def add(self, status: str, what: str, detail: str = ""):
        self.items.append((status, what, detail))
        mark = {"ok": "[ok]  ", "warn": "[warn]", "FAIL": "[FAIL]"}[status]
        print(f"{mark} {what}" + (f" - {detail}" if detail else ""))

    @property
    def blocking(self) -> bool:
        return any(s == FAIL for s, _, _ in self.items)


def check_python(r: Report):
    v = sys.version_info
    r.add(OK if v >= (3, 11) else FAIL, f"Python {v.major}.{v.minor}.{v.micro}",
          "" if v >= (3, 11) else "3.11+ required")


def check_packages(r: Report):
    from importlib import import_module, metadata
    for mod, pip in REQUIRED.items():
        try:
            import_module(mod)
            try:
                ver = metadata.version(pip)
            except metadata.PackageNotFoundError:
                ver = "?"
            r.add(OK, f"{pip} {ver}")
        except Exception as e:
            r.add(FAIL, f"{pip} not importable", f"pip install {pip} ({type(e).__name__})")


def check_tools(r: Report):
    ff = ffmpeg_exe()
    r.add(OK if ff else FAIL, "ffmpeg", ff or "not found - install ffmpeg or `pip install imageio-ffmpeg`")
    fp = ffprobe_exe()
    r.add(OK if fp else WARN, "ffprobe",
          fp or "not found - phase 0 falls back to OpenCV for metadata (no bit rate)")


def check_disk(r: Report, min_gb: float = 5.0):
    free = shutil.disk_usage(REPO_ROOT).free / 1e9
    r.add(OK if free >= min_gb else WARN, f"disk free {free:.1f} GB",
          "" if free >= min_gb else f"recommend >= {min_gb} GB (models + frames)")


def check_model_caches(r: Report, whisper_model: str):
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    clap = hub / "models--laion--clap-htsat-fused"
    r.add(OK if clap.exists() else WARN, "CLAP model cached",
          "" if clap.exists() else "phase 5 will download ~0.6 GB on first run")
    wh = hub / f"models--Systran--faster-whisper-{whisper_model}"
    r.add(OK if wh.exists() else WARN, f"Whisper '{whisper_model}' cached",
          "" if wh.exists() else "phase 4 will download it on first run")


def check_env(r: Report) -> str:
    load_env()
    key = os.environ.get("OPENROUTER_API_KEY", "")
    placeholder = key in ("", "your-key-here")
    r.add(WARN if placeholder else OK, "OPENROUTER_API_KEY",
          "missing - phase 2 needs it (or use --skip-phase2 / --provider mock for tests)"
          if placeholder else f"set ({len(key)} chars, not shown)")
    model = os.environ.get("VISION_MODEL", "google/gemma-4-31b-it")
    r.add(OK, f"VISION_MODEL = {model}")
    return model


def check_online(r: Report, model: str, n_shots: int | None, max_usd: float):
    import urllib.request
    base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=20) as resp:
            data = json.loads(resp.read()).get("data", [])
    except Exception as e:
        r.add(WARN, "OpenRouter model list", f"unreachable: {e}")
        return
    m = next((x for x in data if x.get("id") == model), None)
    if m is None:
        close = [x["id"] for x in data if model.split("/")[-1].split("-")[0] in x.get("id", "")][:5]
        r.add(FAIL, f"model '{model}' not on OpenRouter", f"similar: {close}")
        return
    mods = (m.get("architecture") or {}).get("input_modalities") or []
    r.add(OK if "image" in mods else FAIL, f"'{model}' accepts images",
          f"input modalities: {mods}")
    params = set(m.get("supported_parameters") or [])
    missing = [p for p in ("temperature", "seed", "response_format", "max_tokens") if p not in params]
    r.add(OK if not missing else WARN, "request parameters supported",
          "all of temperature/seed/response_format/max_tokens" if not missing
          else f"not supported: {missing} (reproducibility or JSON mode weaker)")
    pr = m.get("pricing") or {}
    r.add(OK, f"price per M tokens: in ${float(pr.get('prompt') or 0) * 1e6:.3f}, "
              f"out ${float(pr.get('completion') or 0) * 1e6:.3f}"
              + (f", per image ${float(pr.get('image') or 0):.5f}" if pr.get("image") else ""))
    if n_shots:
        import phase2_vision as p2
        est = p2.estimate_cost(n_shots + 1, pr)
        r.add(OK if est is not None and est <= max_usd else WARN,
              f"estimated phase-2 cost for {n_shots} shots <= ${est:.3f}",
              f"budget ${max_usd:.2f}" + ("" if est <= max_usd else " - raise --max-usd"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--online", action="store_true", help="also query OpenRouter's public model list (free)")
    ap.add_argument("--id", default=None, help="video id: count shots from data/<id>/shots.json")
    ap.add_argument("--shots", type=int, default=None, help="number of shots for the cost estimate")
    ap.add_argument("--model", default=None, help="override VISION_MODEL")
    ap.add_argument("--whisper-model", default="small")
    ap.add_argument("--max-usd", type=float, default=float(os.environ.get("VISION_MAX_USD", "1.0")))
    args = ap.parse_args()

    r = Report()
    print("== environment ==")
    check_python(r)
    check_tools(r)
    check_disk(r)
    print("== packages ==")
    check_packages(r)
    print("== models ==")
    check_model_caches(r, args.whisper_model)
    model = check_env(r)
    if args.model:
        model = args.model
    n_shots = args.shots
    if args.id and n_shots is None:
        sp = REPO_ROOT / "data" / args.id / "shots.json"
        if sp.exists():
            n_shots = len(json.loads(sp.read_text(encoding="utf-8")))
            r.add(OK, f"{disp(sp)}: {n_shots} shots")
        else:
            r.add(WARN, f"{disp(sp)} not found", "run phases 0-1 first for a shot count")
    if args.online:
        print("== OpenRouter (public, no charge) ==")
        check_online(r, model, n_shots, args.max_usd)

    n_fail = sum(1 for s, _, _ in r.items if s == FAIL)
    n_warn = sum(1 for s, _, _ in r.items if s == WARN)
    print(f"\n{'READY' if not r.blocking else 'NOT READY'}: {n_fail} blocking, {n_warn} warnings")
    return 1 if r.blocking else 0


if __name__ == "__main__":
    sys.exit(main())
