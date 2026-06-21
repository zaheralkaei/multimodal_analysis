"""
Phase 5 — Audio tagging with CLAP (via transformers).

Reads:  data/processed/audio.wav (from Phase 0)
Writes: data/processed/audio_clap.csv — per-window CLAP similarity scores
        data/processed/audio_clap.json — same as JSON

Uses laion/clap-htsat-fused via transformers (no separate CLAP install needed).

For each 5-second audio window, compute similarity against a fixed vocabulary
of mood / section / instrument tags. Output is a per-time feature matrix
you can correlate with visual analysis.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"


# Fixed vocabulary — chosen for music-video analysis.
MOOD_TAGS = [
    "happy and bright",
    "sad and melancholic",
    "aggressive and intense",
    "romantic and tender",
    "triumphant and epic",
    "calm and peaceful",
    "tense and anxious",
    "dreamy and ethereal",
    "dark and ominous",
    "playful and whimsical",
    "lonely and introspective",
    "powerful and confident",
]
SECTION_TAGS = [
    "intro",
    "verse",
    "chorus",
    "bridge",
    "outro",
    "instrumental break",
    "vocal only",
]
INSTRUMENT_TAGS = [
    "acoustic guitar",
    "electric guitar",
    "piano",
    "drums and percussion",
    "bass guitar",
    "synthesizer",
    "strings orchestra",
    "vocal only no instruments",
]
ALL_TAGS = MOOD_TAGS + SECTION_TAGS + INSTRUMENT_TAGS


def load_clap():
    """Load CLAP model + processor from HuggingFace."""
    from transformers import ClapModel, ClapProcessor
    model_name = "laion/clap-htsat-fused"
    print(f"[info] loading {model_name} ...")
    processor = ClapProcessor.from_pretrained(model_name)
    model = ClapModel.from_pretrained(model_name)
    model.eval()
    return model, processor


def slice_audio(audio_path: Path, window_sec: float = 5.0):
    """Yield (start_sec, end_sec, audio_array, sample_rate) per window."""
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(str(audio_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # mono
    # CLAP requires 48kHz; resample if needed
    if sr != 48000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=48000)
        sr = 48000
    win = int(window_sec * sr)  # sr is now 48000 after resampling
    for i in range(0, len(data), win):
        chunk = data[i : i + win]
        if len(chunk) < win // 2:  # skip trailing half-windows
            break
        yield round(i / sr, 3), round((i + len(chunk)) / sr, 3), chunk, sr


def compute_window_similarities(model, processor, audio_chunk, sr) -> dict:
    """Return {tag: similarity} for ALL_TAGS on a single audio chunk."""
    import numpy as np
    import torch
    inputs = processor(
        audio=[audio_chunk],
        sampling_rate=sr,
        text=ALL_TAGS,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    # logits_per_audio is (1, n_tags); softmax → probabilities
    probs = outputs.logits_per_audio.softmax(dim=1).cpu().numpy()[0]
    return {tag: float(p) for tag, p in zip(ALL_TAGS, probs)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", type=float, default=5.0, help="window size in seconds")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    audio_path = PROCESSED / "audio.wav"
    if not audio_path.exists():
        print(f"[error] audio not found at {audio_path}")
        return 1

    model, processor = load_clap()

    import csv
    out_csv = PROCESSED / "audio_clap.csv"
    out_json = PROCESSED / "audio_clap.json"

    rows = []
    t0 = time.time()
    print(f"[info] slicing audio into {args.window}s windows ...")
    n_windows = 0
    for start, end, chunk, sr in slice_audio(audio_path, args.window):
        sims = compute_window_similarities(model, processor, chunk, sr)
        row = {
            "start_sec": start,
            "end_sec": end,
            **sims,
        }
        rows.append(row)
        n_windows += 1
        if n_windows % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{n_windows}] windows ({elapsed:.0f}s, {elapsed/n_windows:.1f}s/window)")
    elapsed = time.time() - t0

    cols = ["start_sec", "end_sec"] + ALL_TAGS
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\n[ok] wrote {out_csv.relative_to(REPO_ROOT)} ({len(rows)} windows × {len(ALL_TAGS)} tags)")
    print(f"[stats] {elapsed:.0f}s total ({elapsed/max(1,len(rows)):.1f}s/window)")

    out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    # Quick sanity check
    if rows:
        max_mood = max(MOOD_TAGS, key=lambda t: sum(r[t] for r in rows)/len(rows))
        max_section = max(SECTION_TAGS, key=lambda t: sum(r[t] for r in rows)/len(rows))
        print(f"\n[sanity] avg-highest mood: '{max_mood}'")
        print(f"[sanity] avg-highest section: '{max_section}'")

        # Tag activity report (round-2 audit finding):
        # 21/27 tags had variance ~0 in Tyla run. A tag is "active" if
        # its max score across windows exceeds the threshold. This is a
        # signal-to-noise check: tags that never score > 0.05 are noise.
        ACTIVE_THRESHOLD = 0.05
        active = []
        inactive = []
        for tag in ALL_TAGS:
            max_score = max(float(r[tag]) for r in rows)
            if max_score > ACTIVE_THRESHOLD:
                active.append((tag, max_score))
            else:
                inactive.append((tag, max_score))
        print(f"\n[active] {len(active)}/{len(ALL_TAGS)} tags have max score > {ACTIVE_THRESHOLD}:")
        for tag, mx in sorted(active, key=lambda x: -x[1]):
            print(f"  ✓ {tag:<30} max={mx:.3f}")
        if inactive:
            print(f"[inactive] {len(inactive)} tags below threshold (will appear as flat-zero in dashboard):")
            for tag, mx in sorted(inactive, key=lambda x: -x[1]):
                print(f"  · {tag:<30} max={mx:.3f}")

    print(f"\n[next] Phase 6: python scripts/phase6_music.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
