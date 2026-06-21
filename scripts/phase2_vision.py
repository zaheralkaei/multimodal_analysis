"""
Phase 2 — Per-shot visual analysis with a vision-language model.

Reads:  data/processed/shots.json (from Phase 1)
        data/processed/frames/frame_*.jpg (from Phase 0)
Writes: data/processed/shot_vision.csv — one row per shot with 8 Q&A columns

Uses Ollama for vision-language inference. Supports both:
  - Local ollama (http://localhost:11434) — works offline, model must fit in RAM
  - Ollama cloud (https://ollama.com/api) — needs OLLAMA_API_KEY in .env

Endpoint selected via OLLAMA_BASE_URL env var (defaults to local).
Model selected via --model flag (defaults to OLLAMA_VISION_MODEL env var
or "gemma3:4b" if neither is set).

Each question is asked separately with temperature=0 and seed=42 for
reproducibility (see Phase 2 of taylor-swift-lyrics-nlp audit pattern).
"""
from __future__ import annotations
import argparse, base64, csv, json, os, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"

# Load .env for OLLAMA_API_KEY, OLLAMA_BASE_URL, etc.
sys.path.insert(0, str(Path(__file__).parent))
try:
    from _env import load_env
    load_env()
except ImportError:
    pass


# Each question gets its own column in the output CSV. Order matters.
QUESTIONS = [
    ("caption",      "Describe this scene in 1-2 sentences. Be specific about what is visible."),
    ("camera",       "What is the camera doing? Answer with ONE of: static, pan, tilt, zoom-in, zoom-out, dolly, tracking, handheld, or unknown."),
    ("emotion",      "What is the dominant emotion shown by the people in this frame? Answer with one or two words (e.g. 'sad', 'joyful', 'anxious', 'neutral', 'angry', 'contemplative')."),
    ("colors",       "List the 3 most prominent colors in this frame, comma-separated (e.g. 'blue, white, brown')."),
    ("entities",     "List the main objects and people visible, comma-separated. Be concise (max 8 items)."),
    ("location",     "Is this scene indoor or outdoor? Also describe the setting in 3-5 words."),
    ("lighting",     "Describe the lighting in 3-5 words (e.g. 'harsh sunlight', 'dim warm interior', 'neon-lit night')."),
    ("composition",  "Describe the visual composition/framing in 3-5 words (e.g. 'close-up face', 'wide aerial shot', 'over-shoulder medium')."),
]


def get_endpoint() -> tuple[str, dict]:
    """Return (url, headers) for the ollama endpoint.

    Reads OLLAMA_BASE_URL from env (default: http://localhost:11434).
    If URL is the cloud (https://ollama.com), adds Authorization header.
    """
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    # If base already has /api, don't add it again
    if base.endswith("/api"):
        url = f"{base}/generate"
    else:
        url = f"{base}/api/generate"
    headers = {"Content-Type": "application/json"}
    if "ollama.com" in base:
        api_key = os.environ.get("OLLAMA_API_KEY", "")
        if not api_key:
            raise ValueError("OLLAMA_BASE_URL points to ollama.com but OLLAMA_API_KEY is not set")
        headers["Authorization"] = f"Bearer {api_key}"
    return url, headers


def call_ollama(model: str, prompt: str, image_b64: str, timeout: int = 120,
                seed: int = 42) -> tuple[str, float]:
    """Call ollama /api/generate. Returns (response_text, latency_seconds)."""
    import urllib.request
    url, headers = get_endpoint()
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0, "seed": seed, "num_predict": 300},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        result = json.loads(r.read())
    return result.get("response", "").strip(), time.time() - t0


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def analyze_shots(model: str, shots: list[dict], frames_dir: Path,
                  out_csv: Path) -> tuple[list[dict], dict]:
    """For each shot, ask all questions about its mid frame. Saves incrementally."""
    import csv
    rows = []
    stats = {"calls": 0, "errors": 0, "total_seconds": 0.0}

    # Resume support: load any existing rows from out_csv.
    # DEDUPE: keep LAST row per shot_idx (in case prior resume appended dupes).
    existing = {}
    if out_csv.exists():
        with out_csv.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    existing[int(r["shot_idx"])] = r
                except (ValueError, KeyError):
                    pass
        if existing:
            print(f"[info] resuming from existing CSV: {len(existing)} shots already done")
            # Rewrite the CSV without duplicates (in case prior run appended dupes)
            cols = ["shot_idx", "start_sec", "end_sec", "duration_sec", "mid_frame"] + [q[0] for q in QUESTIONS]
            tmp_rows = sorted(existing.values(), key=lambda r: int(r["shot_idx"]))
            with out_csv.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in tmp_rows:
                    w.writerow(r)

    cols = ["shot_idx", "start_sec", "end_sec", "duration_sec", "mid_frame"] + [q[0] for q in QUESTIONS]

    # Open CSV in append mode (or write header if new)
    file_exists = out_csv.exists()
    f_out = out_csv.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f_out, fieldnames=cols)
    if not file_exists:
        writer.writeheader()

    try:
        for i, shot in enumerate(shots):
            if i in existing:
                rows.append(existing[i])
                continue
            mid_rel = shot["mid_frame_path"]
            mid_path = REPO_ROOT / mid_rel
            if not mid_path.exists():
                print(f"[warn] shot {i}: mid-frame missing: {mid_path}", flush=True)
                continue
            b64 = encode_image(mid_path)
            row = {
                "shot_idx": i,
                "start_sec": shot["start_sec"],
                "end_sec": shot["end_sec"],
                "duration_sec": shot["duration_sec"],
                "mid_frame": mid_rel,
            }
            for qname, qprompt in QUESTIONS:
                try:
                    ans, secs = call_ollama(model, qprompt, b64)
                    row[qname] = ans
                    stats["calls"] += 1
                    stats["total_seconds"] += secs
                except Exception as e:
                    row[qname] = f"[error: {e}]"
                    stats["errors"] += 1
            rows.append(row)
            writer.writerow(row)
            f_out.flush()  # write to disk immediately so we can resume on crash
            if (i + 1) % 5 == 0 or i == len(shots) - 1:
                print(f"  [{i+1}/{len(shots)}] shots analyzed "
                      f"(avg {stats['total_seconds']/max(1, stats['calls']):.1f}s/call)",
                      flush=True)
    finally:
        f_out.close()
    return rows, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_model = os.environ.get("VISION_MODEL", "gemma3:4b")
    parser.add_argument("--model", default=default_model,
                       help=f"Ollama vision model name (default: {default_model}). "
                            f"Use OLLAMA_VISION_MODEL env var to set a different default. "
                            f"Cloud options: qwen2.5vl:72b-cloud, qwen2.5vl:32b-cloud, gemma3:27b-cloud.")
    args = parser.parse_args()

    # Show config
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    has_key = bool(os.environ.get("OLLAMA_API_KEY"))
    print(f"[info] endpoint: {base}")
    print(f"[info] auth: {'bearer token' if has_key else 'no auth (local)'}")
    print(f"[info] model: {args.model}")

    shots_path = PROCESSED / "shots.json"
    if not shots_path.exists():
        print(f"[error] shots.json not found at {shots_path}")
        print("  run phase 1 first: python scripts/phase1_shots.py")
        return 1

    shots = json.loads(shots_path.read_text(encoding="utf-8"))
    print(f"[info] loaded {len(shots)} shots", flush=True)

    # Quick health check
    print(f"[info] testing model ...", flush=True)
    try:
        from PIL import Image
        dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
        dummy_path = REPO_ROOT / "data" / "processed" / "_healthcheck.jpg"
        dummy.save(dummy_path)
        b64 = base64.b64encode(dummy_path.read_bytes()).decode()
        # Cloud models need longer timeout for first call (cold start)
        timeout = 180 if "ollama.com" in base else 60
        resp, secs = call_ollama(args.model, "Reply with the word 'ready' only.", b64, timeout=timeout)
        print(f"[ok] model responded in {secs:.1f}s: {resp[:50]!r}", flush=True)
        dummy_path.unlink(missing_ok=True)
    except Exception as e:
        print(f"[error] model health check failed: {e}")
        print(f"  - if using cloud: check OLLAMA_API_KEY in .env")
        print(f"  - if using local: ensure ollama is running and model is pulled: ollama pull {args.model}")
        return 1

    print(f"\n[info] analyzing {len(shots)} shots × {len(QUESTIONS)} questions = "
          f"{len(shots) * len(QUESTIONS)} total calls", flush=True)
    out_csv = PROCESSED / "shot_vision.csv"
    rows, stats = analyze_shots(args.model, shots, PROCESSED / "frames", out_csv)

    # Final summary (CSV was written incrementally during analysis)
    print(f"\n[ok] wrote {out_csv.relative_to(REPO_ROOT)} ({len(rows)} rows, "
          f"{stats['calls']} successful calls, {stats['errors']} errors)")
    print(f"[stats] total time: {stats['total_seconds']:.0f}s "
          f"({stats['total_seconds']/60:.1f} min, "
          f"avg {stats['total_seconds']/max(1,stats['calls']):.1f}s/call)")

    print(f"\n[next] Phase 3: python scripts/phase3_camera.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
