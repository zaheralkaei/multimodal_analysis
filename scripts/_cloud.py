"""Shared layer for hosted model APIs (round 11, audio stack).

Heavy models run on hosted services; this module gives every phase the same
plumbing:

  - keys:    read from .env / environment (FAL_KEY, GROQ_API_KEY,
             REPLICATE_API_TOKEN); never printed or logged.
  - cache:   responses are stored under data/cache/<service>/<key> where the
             key hashes the INPUT CONTENT + request parameters, so a re-run
             never pays twice and never reuses an answer for other audio.
  - retry:   exponential backoff on 408/429/5xx and network errors.
  - costs:   every paid call (and every cache hit) is appended to
             data/<video_id>/cloud_usage.json with an estimated price;
             `budget_ok()` refuses calls beyond AUDIO_MAX_USD (default 1.0).

Prices are estimates from the providers' published rates (checked
2026-10-20) and are labelled as such in the log.
"""
from __future__ import annotations
import datetime, hashlib, json, os, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "cache"

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}

# Published rates used for estimates (USD)
PRICES = {
    "fal_demucs_per_sec": 0.0007,          # fal.ai fal-ai/demucs
    "groq_whisper_turbo_per_hour": 0.04,   # Groq whisper-large-v3-turbo, min 10 s/request
    "groq_whisper_v3_per_hour": 0.111,
    "replicate_allin1_per_run": 0.11,      # sakemin/all-in-one-music-structure-analyzer (A100)
}


class CloudError(RuntimeError):
    pass


def load_env() -> None:
    try:
        from _env import load_env as _load
        _load()
    except ImportError:
        pass


def get_key(name: str) -> str | None:
    load_env()
    v = os.environ.get(name, "").strip()
    return v if v and v not in ("your-key-here",) else None


# ------------------------------------------------------------------ hashing / cache

def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cache_key(*parts) -> str:
    return sha_bytes(json.dumps(parts, sort_keys=True, default=str).encode())[:32]


def cache_path(service: str, key: str, suffix: str = ".json") -> Path:
    return CACHE_DIR / service / f"{key}{suffix}"


def cache_get_json(service: str, key: str):
    p = cache_path(service, key)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def cache_put_json(service: str, key: str, data) -> Path:
    p = cache_path(service, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


# ------------------------------------------------------------------ HTTP

def request(method: str, url: str, *, headers=None, json_body=None, data=None, files=None,
            timeout: int = 300, retries: int = 4, expect_json: bool = True):
    """requests call with backoff. Raises CloudError with a short message."""
    import requests
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            r = requests.request(method, url, headers=headers, json=json_body, data=data,
                                 files=files, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == retries:
                raise CloudError(f"{method} {url.split('?')[0]}: network error {e}") from None
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
        if r.status_code in RETRY_STATUS and attempt < retries:
            ra = r.headers.get("Retry-After", "")
            time.sleep(float(ra) if ra.replace(".", "", 1).isdigit() else delay)
            delay = min(delay * 2, 60)
            continue
        if r.status_code >= 400:
            raise CloudError(f"{method} {url.split('?')[0]}: HTTP {r.status_code}: {r.text[:300]}")
        if not expect_json:
            return r.content
        try:
            return r.json()
        except ValueError:
            raise CloudError(f"{method} {url.split('?')[0]}: response is not JSON") from None
    raise CloudError("unreachable")


def download(url: str, dest: Path, timeout: int = 600) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(request("GET", url, timeout=timeout, expect_json=False))
    return dest


# ------------------------------------------------------------------ cost log

def usage_path(processed: Path) -> Path:
    return Path(processed) / "cloud_usage.json"


def log_usage(processed: Path, service: str, operation: str, est_usd: float,
              cached: bool, detail: dict | None = None) -> None:
    p = usage_path(processed)
    try:
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"calls": []}
    except json.JSONDecodeError:
        data = {"calls": []}
    data["calls"].append({
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "service": service, "operation": operation, "cached": cached,
        "est_usd": 0.0 if cached else round(est_usd, 5), "detail": detail or {},
        "price_basis": "provider list price, estimate",
    })
    data["total_est_usd"] = round(sum(c["est_usd"] for c in data["calls"]), 5)
    p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def spent(processed: Path) -> float:
    p = usage_path(processed)
    if not p.exists():
        return 0.0
    try:
        return float(json.loads(p.read_text(encoding="utf-8")).get("total_est_usd", 0.0))
    except (json.JSONDecodeError, ValueError):
        return 0.0


def budget_ok(processed: Path, est_usd: float) -> tuple[bool, str]:
    cap = float(os.environ.get("AUDIO_MAX_USD", "1.0"))
    already = spent(processed)
    if already + est_usd > cap:
        return False, (f"estimated ${est_usd:.3f} would exceed the audio budget "
                       f"(${already:.3f} already spent of AUDIO_MAX_USD=${cap:.2f})")
    return True, ""


# ------------------------------------------------------------------ audio helpers

def encode_audio(src: Path, dest: Path, fmt: str, sample_rate: int | None = None,
                 mono: bool = False, bitrate: str = "192k") -> Path:
    """Re-encode with ffmpeg (mp3 for uploads, flac for lossless ASR)."""
    import subprocess
    from _tools import ffmpeg_exe
    cmd = [ffmpeg_exe() or "ffmpeg", "-y", "-loglevel", "error", "-i", str(src)]
    if mono:
        cmd += ["-ac", "1"]
    if sample_rate:
        cmd += ["-ar", str(sample_rate)]
    if fmt == "mp3":
        cmd += ["-c:a", "libmp3lame", "-b:a", bitrate]
    elif fmt == "flac":
        cmd += ["-c:a", "flac"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd + [str(dest)], check=True, timeout=600)
    return dest


def data_uri(path: Path, mime: str) -> str:
    import base64
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()
