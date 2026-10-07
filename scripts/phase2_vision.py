"""
Phase 2 — Per-shot visual analysis with a vision-language model.

Reads:  <PROCESSED>/shots.json (from Phase 1)
        <PROCESSED>/frames/frame_*.jpg (from Phase 0)
Writes: <PROCESSED>/shot_vision.csv — one row per shot with 8 answer columns
        <PROCESSED>/shot_vision_meta.json — model, endpoint, prompt settings

Uses Ollama for vision-language inference. Supports both:
  - Local ollama (http://localhost:11434) — works offline, model must fit in RAM
  - Ollama cloud (https://ollama.com/api) — needs OLLAMA_API_KEY in .env

Round-3 changes:
  - Three key frames per shot (15% / 50% / 85%, from Phase 1) instead of only
    the mid-frame, so the model can see motion and change within the shot.
  - Structured output: the request carries a JSON schema (Ollama ``format``)
    whose ``camera`` and ``emotion`` fields are enums, so answers come from the
    fixed vocabularies. If the endpoint rejects a schema, we fall back to plain
    JSON mode and normalize the emotion with _normalize_emotion.
  - Requests run in parallel (--workers, default 4).
  - The model name is recorded (shot_vision_meta.json, run_info.json).

Each request uses temperature=0 and seed=42 for reproducibility.
"""
from __future__ import annotations
import argparse, base64, csv, json, os, sys, threading, time, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import CAMERA_LABELS, PROCESSED, REPO_ROOT, display_path, record_run

# Load .env for OLLAMA_API_KEY, OLLAMA_BASE_URL, etc.
try:
    from _env import load_env
    load_env()
except ImportError:
    pass

from _normalize_emotion import list_canonical, normalize_emotion

EMOTIONS = list_canonical()

# Each question gets its own column in the output CSV.
QUESTIONS = [
    ("caption",      "Describe the shot in 1-2 sentences. Be specific about what is visible."),
    ("camera",       f"What is the camera doing across the frames? One of: {', '.join(CAMERA_LABELS)}."),
    ("emotion",      f"Dominant emotion of people in frame. One of: {', '.join(EMOTIONS)}."),
    ("colors",       "3 most prominent colors, comma-separated (e.g. 'blue, white, brown')."),
    ("entities",     "Main objects and people, comma-separated. Max 8 items, concise."),
    ("location",     "indoor or outdoor + setting in 3-5 words (e.g. 'indoor bedroom', 'outdoor beach at sunset')."),
    ("lighting",     "Lighting in 3-5 words (e.g. 'harsh sunlight', 'dim warm interior', 'neon-lit night')."),
    ("composition",  "Visual composition/framing in 3-5 words "
                     "(e.g. 'close-up face', 'wide aerial shot', 'over-shoulder medium')."),
]
CSV_COLS = ["shot_idx", "start_sec", "end_sec", "duration_sec", "mid_frame", "n_images"] + [q[0] for q in QUESTIONS]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {"type": "string"},
        "camera": {"type": "string", "enum": CAMERA_LABELS},
        "emotion": {"type": "string", "enum": EMOTIONS},
        "colors": {"type": "string"},
        "entities": {"type": "string"},
        "location": {"type": "string"},
        "lighting": {"type": "string"},
        "composition": {"type": "string"},
    },
    "required": [q[0] for q in QUESTIONS],
}


def build_prompt(n_images: int) -> str:
    frames = ("You are shown 1 frame from a shot of a music video." if n_images == 1 else
              f"You are shown {n_images} frames from ONE shot of a music video, in time order "
              "(early, middle, late). Treat them as one shot.")
    lines = [frames, "", "Respond with ONLY a JSON object with these keys:"]
    lines += [f"- {name}: {q}" for name, q in QUESTIONS]
    lines.append("No text before or after the JSON. No markdown code fences.")
    return "\n".join(lines)


def get_endpoint() -> tuple[str, dict]:
    """Return (url, headers) for the ollama endpoint."""
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    url = f"{base}/generate" if base.endswith("/api") else f"{base}/api/generate"
    headers = {"Content-Type": "application/json"}
    if "ollama.com" in base:
        api_key = os.environ.get("OLLAMA_API_KEY", "")
        if not api_key:
            raise ValueError("OLLAMA_BASE_URL points to ollama.com but OLLAMA_API_KEY is not set")
        headers["Authorization"] = f"Bearer {api_key}"
    return url, headers


def call_ollama(model: str, prompt: str, images_b64: list[str], timeout: int = 120,
                seed: int = 42, num_predict: int = 1500, fmt=None) -> tuple[str, float]:
    """Call ollama /api/generate. Returns (response_text, latency_seconds)."""
    import urllib.request
    url, headers = get_endpoint()
    payload = {
        "model": model,
        "prompt": prompt,
        "images": images_b64,
        "stream": False,
        "options": {"temperature": 0, "seed": seed, "num_predict": num_predict},
    }
    if fmt is not None:
        payload["format"] = fmt
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        result = json.loads(r.read())
    return result.get("response", "").strip(), time.time() - t0


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def parse_json_response(raw: str) -> dict:
    """Parse the model's response as JSON. Recovers from code fences and chatter."""
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = text[:-3].strip()
    if "{" in text:
        start, end = text.index("{"), text.rfind("}")
        if end > start:
            text = text[start:end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"_parse_error": raw[:200]}
    if not isinstance(data, dict):
        return {"_parse_error": raw[:200]}

    cleaned = {}
    for key, _ in QUESTIONS:
        v = data.get(key, "")
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v)
        cleaned[key] = str(v).strip()
    return cleaned


def clean_answers(parsed: dict) -> dict:
    """Force camera/emotion onto the fixed vocabularies (needed when schema mode is off)."""
    out = dict(parsed)
    if out.get("emotion"):
        out["emotion"] = normalize_emotion(out["emotion"])
    cam = (out.get("camera") or "").strip().lower()
    if cam and cam not in CAMERA_LABELS:
        family = cam.split("-")[0]  # pan-left → pan
        out["camera"] = family if family in CAMERA_LABELS else "other"
    return out


def shot_images(shot: dict) -> list[Path]:
    """Key frames for a shot (falls back to the mid-frame for old shots.json files)."""
    rels = shot.get("key_frame_paths") or [shot["mid_frame_path"]]
    seen, paths = set(), []
    for rel in rels:
        p = REPO_ROOT / rel
        if rel not in seen and p.exists():
            seen.add(rel)
            paths.append(p)
    return paths


def row_is_reusable(row: dict, shot: dict) -> bool:
    """A previous row can be reused if it succeeded and belongs to the same shot."""
    try:
        return (not (row.get("caption") or "").startswith(("[error", "[parse_error"))
                and abs(float(row["start_sec"]) - float(shot["start_sec"])) <= 1e-3
                and row.get("mid_frame") == shot["mid_frame_path"]
                and int(row.get("n_images") or 1) == len(shot_images(shot)))
    except (KeyError, ValueError, TypeError):
        return False


class Analyzer:
    """Holds the per-run state shared by worker threads."""

    def __init__(self, model: str, use_schema: bool, timeout: int):
        self.model = model
        self.use_schema = use_schema
        self.timeout = timeout
        self.lock = threading.Lock()
        self.stats = {"calls": 0, "errors": 0, "parse_errors": 0, "total_seconds": 0.0}

    def _call(self, prompt: str, images: list[str]) -> tuple[str, float]:
        if self.use_schema:
            try:
                return call_ollama(self.model, prompt, images, timeout=self.timeout, fmt=RESPONSE_SCHEMA)
            except urllib.error.HTTPError as e:
                if e.code not in (400, 422, 501):
                    raise
                with self.lock:
                    if self.use_schema:
                        print(f"[warn] endpoint rejected the JSON schema (HTTP {e.code}); "
                              "falling back to plain JSON + emotion normalization", flush=True)
                    self.use_schema = False
        return call_ollama(self.model, prompt, images, timeout=self.timeout, fmt="json")

    def analyze(self, i: int, shot: dict) -> dict | None:
        paths = shot_images(shot)
        if not paths:
            print(f"[warn] shot {i}: no frames found for {shot['mid_frame_path']}", flush=True)
            return None
        row = {"shot_idx": i, "start_sec": shot["start_sec"], "end_sec": shot["end_sec"],
               "duration_sec": shot["duration_sec"], "mid_frame": shot["mid_frame_path"],
               "n_images": len(paths)}
        try:
            raw, secs = self._call(build_prompt(len(paths)), [encode_image(p) for p in paths])
            parsed = parse_json_response(raw)
            with self.lock:
                self.stats["calls"] += 1
                self.stats["total_seconds"] += secs
                if "_parse_error" in parsed:
                    self.stats["parse_errors"] += 1
            if "_parse_error" in parsed:
                row["caption"] = f"[parse_error: {parsed['_parse_error']}]"
            else:
                row.update(clean_answers(parsed))
        except Exception as e:
            row["caption"] = f"[error: {e}]"
            with self.lock:
                self.stats["errors"] += 1
        return row


def analyze_shots(model: str, shots: list[dict], out_csv: Path, workers: int = 4,
                  use_schema: bool = True, timeout: int = 120) -> tuple[list[dict], dict]:
    """Analyze every shot not already done. Saves incrementally; resumable."""
    existing = {}
    if out_csv.exists():
        with out_csv.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    idx = int(r["shot_idx"])
                except (KeyError, ValueError):
                    continue
                if idx < len(shots) and row_is_reusable(r, shots[idx]):
                    existing[idx] = r
        print(f"[info] resuming: {len(existing)}/{len(shots)} shots reusable from {display_path(out_csv)}")

    rows = dict(existing)

    def write_all() -> None:
        tmp = out_csv.with_suffix(".csv.tmp")
        with tmp.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
            w.writeheader()
            for idx in sorted(rows):
                w.writerow(rows[idx])
        tmp.replace(out_csv)

    write_all()
    todo = [(i, s) for i, s in enumerate(shots) if i not in existing]
    analyzer = Analyzer(model, use_schema, timeout)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(analyzer.analyze, i, s) for i, s in todo]
        for fut in as_completed(futures):
            row = fut.result()
            done += 1
            if row is not None:
                rows[row["shot_idx"]] = row
                write_all()  # incremental save: a crash loses at most in-flight shots
            if done % 5 == 0 or done == len(todo):
                st = analyzer.stats
                print(f"  [{done}/{len(todo)}] shots analyzed "
                      f"(avg {st['total_seconds'] / max(1, st['calls']):.1f}s/call, "
                      f"{st['parse_errors']} parse errors, {st['errors']} errors)", flush=True)
    stats = dict(analyzer.stats, schema_mode=analyzer.use_schema)
    return [rows[i] for i in sorted(rows)], stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_model = os.environ.get("VISION_MODEL", "gemma3:4b")
    parser.add_argument("--model", default=default_model,
                        help=f"Ollama vision model name (default: {default_model}). "
                             f"Cloud options include gemini-3-flash-preview, gemma3:27b.")
    parser.add_argument("--workers", type=int, default=4, help="parallel requests (default 4)")
    parser.add_argument("--no-schema", dest="use_schema", action="store_false",
                        help="don't send a JSON schema (for endpoints without structured output)")
    args = parser.parse_args()

    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    print(f"[info] endpoint: {base}")
    print(f"[info] auth: {'bearer token' if os.environ.get('OLLAMA_API_KEY') else 'no auth (local)'}")
    print(f"[info] model: {args.model}, workers: {args.workers}")

    shots_path = PROCESSED / "shots.json"
    if not shots_path.exists():
        print(f"[error] shots.json not found at {shots_path}")
        print("  run phase 1 first: python scripts/phase1_shots.py")
        return 1
    shots = json.loads(shots_path.read_text(encoding="utf-8"))
    print(f"[info] loaded {len(shots)} shots", flush=True)

    timeout = 180 if "ollama.com" in base else 120
    print("[info] testing model ...", flush=True)
    try:
        from PIL import Image
        dummy_path = PROCESSED / "_healthcheck.jpg"
        Image.new("RGB", (224, 224), color=(128, 128, 128)).save(dummy_path)
        resp, secs = call_ollama(args.model, "Reply with the word 'ready' only.",
                                 [encode_image(dummy_path)], timeout=timeout, num_predict=50)
        dummy_path.unlink(missing_ok=True)
        print(f"[ok] model responded in {secs:.1f}s: {resp[:50]!r}", flush=True)
    except Exception as e:
        print(f"[error] model health check failed: {e}")
        print("  - if using cloud: check OLLAMA_API_KEY in .env")
        print(f"  - if using local: ensure ollama is running and model is pulled: ollama pull {args.model}")
        return 1

    out_csv = PROCESSED / "shot_vision.csv"
    rows, stats = analyze_shots(args.model, shots, out_csv, args.workers, args.use_schema, timeout)

    meta = {"model": args.model, "endpoint": base, "schema_mode": stats["schema_mode"],
            "images_per_shot": len(shots[0].get("key_frame_paths") or [1]) if shots else 0,
            "temperature": 0, "seed": 42}
    meta_path = PROCESSED / "shot_vision_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"\n[ok] wrote {display_path(out_csv)} ({len(rows)} rows, "
          f"{stats['calls']} calls, {stats['parse_errors']} parse errors, {stats['errors']} errors)")
    print(f"[stats] total model time: {stats['total_seconds']:.0f}s "
          f"(avg {stats['total_seconds'] / max(1, stats['calls']):.1f}s/call)")
    record_run(2, inputs=[shots_path], outputs=[out_csv, meta_path], params=meta)
    print("\n[next] Phase 3: python scripts/phase3_camera.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
