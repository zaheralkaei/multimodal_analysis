"""
Phase 2 — Per-shot visual analysis with a vision-language model.

Reads:  data/processed/shots.json (from Phase 1)
        data/processed/frames/frame_*.jpg (from Phase 0)
Writes: data/processed/shot_vision.csv — one row per shot with caption + camera + emotion + colors + entities + location + lighting + composition

Uses Ollama for vision-language inference. Default model: gemma3:4b (works
locally with limited RAM). To use Qwen2.5-VL on a machine with more RAM, pass
--model qwen2.5vl:7b (or :3b).

Each question is asked separately with temperature=0 for reproducibility and
seed=42 (ollama supports seed; see Phase 2 of taylor-swift-lyrics-nlp audit).
"""
from __future__ import annotations
import argparse, base64, csv, json, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"


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


def call_ollama(model: str, prompt: str, image_b64: str, timeout: int = 120,
                seed: int = 42) -> tuple[str, float]:
    """Call ollama /api/generate. Returns (response_text, latency_seconds)."""
    import urllib.request
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0, "seed": seed, "num_predict": 80},
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        result = json.loads(r.read())
    return result.get("response", "").strip(), time.time() - t0


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def analyze_shots(model: str, shots: list[dict], frames_dir: Path) -> tuple[list[dict], dict]:
    """For each shot, ask all questions about its mid frame."""
    rows = []
    stats = {"calls": 0, "errors": 0, "total_seconds": 0.0}
    for i, shot in enumerate(shots):
        mid_rel = shot["mid_frame_path"]
        mid_path = REPO_ROOT / mid_rel
        if not mid_path.exists():
            print(f"[warn] shot {i}: mid-frame missing: {mid_path}")
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
        if (i + 1) % 5 == 0 or i == len(shots) - 1:
            print(f"  [{i+1}/{len(shots)}] shots analyzed "
                  f"(avg {stats['total_seconds']/max(1, stats['calls']):.1f}s/call)")
    return rows, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gemma3:4b",
                       help="Ollama vision model name (default: gemma3:4b). "
                            "Use qwen2.5vl:7b on a machine with >=48GB RAM.")
    args = parser.parse_args()

    shots_path = PROCESSED / "shots.json"
    if not shots_path.exists():
        print(f"[error] shots.json not found at {shots_path}")
        print("  run phase 1 first: python scripts/phase1_shots.py")
        return 1

    shots = json.loads(shots_path.read_text(encoding="utf-8"))
    print(f"[info] loaded {len(shots)} shots")
    print(f"[info] using model: {args.model}")

    # Quick health check
    print(f"[info] testing model ...")
    try:
        from PIL import Image
        dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
        dummy_path = REPO_ROOT / "data" / "processed" / "_healthcheck.jpg"
        dummy.save(dummy_path)
        b64 = base64.b64encode(dummy_path.read_bytes()).decode()
        resp, secs = call_ollama(args.model, "Reply with the word 'ready' only.", b64, timeout=180)
        print(f"[ok] model responded in {secs:.1f}s: {resp[:50]!r}")
        dummy_path.unlink(missing_ok=True)
    except Exception as e:
        print(f"[error] model health check failed: {e}")
        print(f"  ensure ollama is running and model is pulled: ollama pull {args.model}")
        return 1

    print(f"\n[info] analyzing {len(shots)} shots × {len(QUESTIONS)} questions = "
          f"{len(shots) * len(QUESTIONS)} total calls")
    rows, stats = analyze_shots(args.model, shots, PROCESSED / "frames")

    # Write CSV
    out_csv = PROCESSED / "shot_vision.csv"
    cols = ["shot_idx", "start_sec", "end_sec", "duration_sec", "mid_frame"] + [q[0] for q in QUESTIONS]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\n[ok] wrote {out_csv.relative_to(REPO_ROOT)} ({len(rows)} rows, "
          f"{stats['calls']} successful calls, {stats['errors']} errors)")
    print(f"[stats] total time: {stats['total_seconds']:.0f}s "
          f"({stats['total_seconds']/60:.1f} min, "
          f"avg {stats['total_seconds']/max(1,stats['calls']):.1f}s/call)")

    print(f"\n[next] Phase 3: python scripts/phase3_camera.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
