"""
Phase 2 — Per-shot visual analysis with a vision-language model.

Reads:  data/<video_id>/shots.json (from Phase 1)
        data/<video_id>/frames/frame_*.jpg (from Phase 0)
Writes: data/<video_id>/shot_vision.csv — one row per shot with 8 Q&A columns
        data/<video_id>/shot_vision_stats.json — provider/model/metrics

Model backends (round-4 audit):
  - OpenRouter (default): OpenAI-compatible /v1/chat/completions.
    Needs OPENROUTER_API_KEY in .env. Default model: google/gemma-4-31b-it.
  - Ollama: /api/generate against localhost:11434 or Ollama cloud
    (OLLAMA_BASE_URL + OLLAMA_API_KEY). Kept as a fallback.

The provider is chosen by --provider:
  auto  → OpenRouter if OPENROUTER_API_KEY is set, else Ollama
  openrouter | ollama → force that backend

Round-2 design (kept): single combined JSON-mode prompt per shot (8 questions
in one call), temperature=0, seed=42, stop tokens, markdown-fence stripping.

Round-4 fixes:
  - The model/provider actually used is written to shot_vision_stats.json so
    later phases (7/8) can report it instead of hardcoding a name (AUDIT_R4 F4).
  - Resume re-analyzes a shot when its mid-frame changed (e.g. after an fps
    fix in phase 1) instead of silently keeping stale analysis.
"""
from __future__ import annotations
import argparse, base64, csv, datetime, json, os, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
from _paths import disp
PROCESSED = REPO_ROOT / "data" / "processed"
if "PROCESSED_DIR" in os.environ:
    PROCESSED = Path(os.environ["PROCESSED_DIR"])

# Load .env for API keys (OPENROUTER_API_KEY, OLLAMA_API_KEY, etc.)
sys.path.insert(0, str(Path(__file__).parent))
try:
    from _env import load_env
    load_env()
except ImportError:
    pass

try:
    from _normalize_emotion import normalize_emotion as _normalize_emotion_raw
    NORMALIZE_EMOTIONS = True
except ImportError:
    NORMALIZE_EMOTIONS = False
    _normalize_emotion_raw = lambda x: x  # passthrough


# Each question gets its own column in the output CSV.
QUESTIONS = [
    ("caption",      "Describe this scene in 1-2 sentences. Be specific about what is visible."),
    ("camera",       "What is the camera doing? ONE word from: static, pan, tilt, zoom-in, zoom-out, dolly, tracking, handheld, unknown."),
    ("emotion",      "Dominant emotion of people in frame. ONE OR TWO WORDS from: joyful, sad, angry, fearful, surprised, disgusted, neutral, contemplative, sensual, energetic, melancholic, anxious, playful, romantic, intense, confident."),
    ("colors",       "3 most prominent colors, comma-separated (e.g. 'blue, white, brown')."),
    ("entities",     "Main objects and people, comma-separated. Max 8 items, concise."),
    ("location",     "indoor or outdoor + setting in 3-5 words (e.g. 'indoor bedroom', 'outdoor beach at sunset')."),
    ("lighting",     "Lighting in 3-5 words (e.g. 'harsh sunlight', 'dim warm interior', 'neon-lit night')."),
    ("composition",  "Visual composition/framing in 3-5 words (e.g. 'close-up face', 'wide aerial shot', 'over-shoulder medium')."),
]

# JSON-mode instruction prepended to every prompt
JSON_INSTRUCTION = """Respond with ONLY a JSON object in this exact schema:
{
  "caption": "<your 1-2 sentence description>",
  "camera": "<one word from the list>",
  "emotion": "<one or two words>",
  "colors": "<comma-separated>",
  "entities": "<comma-separated>",
  "location": "<indoor/outdoor + 3-5 word setting>",
  "lighting": "<3-5 words>",
  "composition": "<3-5 words>"
}
Do not add any text before or after the JSON. No markdown code fences."""

STOP_SEQUENCES = ["\n\n", "###", "Question:", "---"]


def build_combined_prompt() -> str:
    """Build the full JSON-mode prompt (used for ALL shots)."""
    parts = [JSON_INSTRUCTION, "\n\nImage to analyze:"]
    for qname, qprompt in QUESTIONS:
        parts.append(f"- {qname}: {qprompt}")
    return "\n".join(parts)


COMBINED_PROMPT = build_combined_prompt()


def resolve_provider(provider: str) -> str:
    """auto → openrouter if OPENROUTER_API_KEY set, else ollama."""
    if provider != "auto":
        return provider
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    return "ollama"


def endpoint_for(provider: str) -> str:
    """Human-readable endpoint identifier for logs and stats."""
    if provider == "openrouter":
        return os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    return base


def call_openrouter(model: str, prompt: str, image_b64: str, timeout: int = 180,
                    seed: int = 42, num_predict: int = 1500) -> tuple[str, float]:
    """OpenRouter (OpenAI-compatible) chat completion with one image."""
    import urllib.request
    url = os.environ.get("OPENROUTER_BASE_URL",
                         "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions"
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set (add it to .env)")
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }],
        "temperature": 0,
        "seed": seed,
        "max_tokens": num_predict,
        "stop": STOP_SEQUENCES,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        # OpenRouter app attribution (optional but recommended)
        "HTTP-Referer": "https://github.com/zaheralkaei/multimodal_analysis",
        "X-Title": "multimodal_analysis",
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        result = json.loads(r.read())
    try:
        text = result["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"unexpected OpenRouter response shape: {result}") from None
    return text.strip(), time.time() - t0


def call_ollama(model: str, prompt: str, image_b64: str, timeout: int = 120,
                seed: int = 42, num_predict: int = 1500) -> tuple[str, float]:
    """Call Ollama /api/generate (local or cloud)."""
    import urllib.request
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    url = f"{base}/generate" if base.endswith("/api") else f"{base}/api/generate"
    headers = {"Content-Type": "application/json"}
    if "ollama.com" in base:
        api_key = os.environ.get("OLLAMA_API_KEY", "")
        if not api_key:
            raise ValueError("OLLAMA_BASE_URL points to ollama.com but OLLAMA_API_KEY is not set")
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_predict": num_predict,
            "stop": STOP_SEQUENCES,
        },
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        result = json.loads(r.read())
    return result.get("response", "").strip(), time.time() - t0


def call_llm(provider: str, model: str, prompt: str, image_b64: str,
             timeout: int = 120, seed: int = 42, num_predict: int = 1500) -> tuple[str, float]:
    """Dispatch to the configured backend."""
    if provider == "openrouter":
        return call_openrouter(model, prompt, image_b64, timeout=timeout, seed=seed,
                               num_predict=num_predict)
    return call_ollama(model, prompt, image_b64, timeout=timeout, seed=seed,
                       num_predict=num_predict)


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def parse_json_response(raw: str) -> dict:
    """Try to parse the model's response as JSON. Recover from common failures."""
    text = raw.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3].strip()
    # Try to find the first { and last }
    if "{" in text:
        start = text.index("{")
        end = text.rfind("}")
        if end > start:
            text = text[start:end+1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"_parse_error": raw[:200]}

    cleaned = {}
    for q in QUESTIONS:
        key = q[0]
        if key in data:
            v = data[key]
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v)
            cleaned[key] = str(v).strip()
        else:
            cleaned[key] = ""
    return cleaned


def analyze_shots(provider: str, model: str, shots: list[dict], frames_dir: Path,
                  out_csv: Path) -> tuple[list[dict], dict]:
    """For each shot, ask the combined JSON-mode prompt once. Saves incrementally."""
    rows = []
    stats = {"calls": 0, "errors": 0, "parse_errors": 0, "total_seconds": 0.0, "skipped": []}

    # Resume support: load any existing rows from out_csv. A shot is re-analyzed
    # when its mid_frame changed (AUDIT_R4: phase 1 mid-frame fix can shift
    # every mid-frame — stale rows must not survive).
    existing: dict[int, dict] = {}
    if out_csv.exists():
        with out_csv.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    existing[int(r["shot_idx"])] = r
                except (ValueError, KeyError):
                    pass
        if existing:
            print(f"[info] resuming from existing CSV: {len(existing)} shots already done")

    cols = ["shot_idx", "start_sec", "end_sec", "duration_sec", "mid_frame"] + [q[0] for q in QUESTIONS]

    with out_csv.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=cols)
        writer.writeheader()
        for i, shot in enumerate(shots):
            mid_rel = shot.get("mid_frame_path", "")
            stale_row = existing.get(i)
            if stale_row is not None and stale_row.get("mid_frame") == mid_rel:
                rows.append(stale_row)
                writer.writerow(stale_row)
                continue
            mid_path = REPO_ROOT / mid_rel
            if not mid_rel or not mid_path.exists():
                print(f"[warn] shot {i}: mid-frame missing: {mid_path}", flush=True)
                stats["skipped"].append(i)
                continue
            b64 = encode_image(mid_path)
            row = {
                "shot_idx": i,
                "start_sec": shot["start_sec"],
                "end_sec": shot["end_sec"],
                "duration_sec": shot["duration_sec"],
                "mid_frame": mid_rel,
            }
            try:
                raw, secs = call_llm(provider, model, COMBINED_PROMPT, b64)
                parsed = parse_json_response(raw)
                if "_parse_error" in parsed:
                    row["caption"] = f"[parse_error: {parsed['_parse_error']}]"
                    stats["parse_errors"] += 1
                else:
                    for qname, _ in QUESTIONS:
                        v = parsed.get(qname, "")
                        if qname == "emotion" and NORMALIZE_EMOTIONS and v:
                            v = _normalize_emotion_raw(v)
                        row[qname] = v
                stats["calls"] += 1
                stats["total_seconds"] += secs
            except Exception as e:
                row["caption"] = f"[error: {e}]"
                stats["errors"] += 1
            rows.append(row)
            writer.writerow(row)
            f_out.flush()
            if (i + 1) % 5 == 0 or i == len(shots) - 1:
                print(f"  [{i+1}/{len(shots)}] shots analyzed "
                      f"(avg {stats['total_seconds']/max(1, stats['calls']):.1f}s/call, "
                      f"{stats['parse_errors']} parse errors)",
                      flush=True)
    return rows, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_model = os.environ.get("VISION_MODEL", "google/gemma-4-31b-it")
    parser.add_argument("--model", default=default_model,
                        help=f"Vision model name (default: {default_model}). "
                             "OpenRouter example: google/gemma-4-31b-it; "
                             "Ollama examples: gemma3:27b, gemma3:4b (local).")
    parser.add_argument("--provider", default="auto", choices=["auto", "openrouter", "ollama"],
                        help="Backend (default: auto — OpenRouter if OPENROUTER_API_KEY set, else Ollama)")
    args = parser.parse_args()

    provider = resolve_provider(args.provider)
    endpoint = endpoint_for(provider)
    has_key = bool(os.environ.get("OPENROUTER_API_KEY") if provider == "openrouter"
                   else os.environ.get("OLLAMA_API_KEY"))
    print(f"[info] provider: {provider}")
    print(f"[info] endpoint: {endpoint}")
    print(f"[info] auth: {'bearer token' if has_key else 'no auth (local)'}")
    print(f"[info] model: {args.model}")
    if provider == "openrouter" and not has_key:
        print("[error] OPENROUTER_API_KEY is not set. Add it to .env (see .env.example).")
        return 1

    shots_path = PROCESSED / "shots.json"
    if not shots_path.exists():
        print(f"[error] shots.json not found at {shots_path}")
        print("  run phase 1 first: python scripts/phase1_shots.py")
        return 1

    shots = json.loads(shots_path.read_text(encoding="utf-8"))
    print(f"[info] loaded {len(shots)} shots", flush=True)

    # Quick health check
    print(f"[info] testing model ...", flush=True)
    dummy_path = PROCESSED / "_healthcheck.jpg"
    try:
        from PIL import Image
        dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
        dummy.save(dummy_path)
        b64 = base64.b64encode(dummy_path.read_bytes()).decode()
        timeout = 180 if provider == "openrouter" else 60
        resp, secs = call_llm(provider, args.model, "Reply with the word 'ready' only.",
                              b64, timeout=timeout, num_predict=50)
        print(f"[ok] model responded in {secs:.1f}s: {resp[:50]!r}", flush=True)
    except Exception as e:
        print(f"[error] model health check failed: {e}")
        if provider == "openrouter":
            print(f"  - check OPENROUTER_API_KEY in .env and that '{args.model}' exists on openrouter.ai")
        else:
            print(f"  - if using cloud: check OLLAMA_API_KEY in .env")
            print(f"  - if using local: ensure ollama is running and model is pulled: ollama pull {args.model}")
        return 1
    finally:
        dummy_path.unlink(missing_ok=True)

    print(f"\n[info] analyzing {len(shots)} shots with combined JSON prompt = "
          f"{len(shots)} total calls", flush=True)
    out_csv = PROCESSED / "shot_vision.csv"
    rows, stats = analyze_shots(provider, args.model, shots, PROCESSED / "frames", out_csv)

    # Persist what was ACTUALLY used — phases 7/8 read this (AUDIT_R4 F4)
    stats.update({
        "provider": provider,
        "model": args.model,
        "endpoint": endpoint,
        "n_shots": len(shots),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    stats_path = PROCESSED / "shot_vision_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(f"\n[ok] wrote {disp(out_csv)} ({len(rows)} rows, "
          f"{stats['calls']} successful calls, {stats['parse_errors']} parse errors, "
          f"{stats['errors']} errors)")
    if stats["skipped"]:
        print(f"[warn] shots skipped (missing mid-frame): {stats['skipped']}")
    print(f"[stats] total time: {stats['total_seconds']:.0f}s "
          f"({stats['total_seconds']/60:.1f} min, "
          f"avg {stats['total_seconds']/max(1,stats['calls']):.1f}s/call)")

    print(f"\n[next] Phase 3: python scripts/phase3_camera.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())