"""
Phase 5 — Audio tagging with CLAP (via transformers).

Reads:  <PROCESSED>/audio.wav (from Phase 0)
Writes: <PROCESSED>/audio_clap.csv  — per-window tag probabilities
        <PROCESSED>/audio_clap.json — same as JSON

Uses laion/clap-htsat-fused via transformers (no separate CLAP install needed).

For each 5-second audio window, CLAP scores the audio against three fixed
vocabularies (common.TAG_GROUPS): mood, song section, instrument.

Round-3 change: each group gets its OWN softmax. Previously all 27 prompts
competed in one softmax, so a window that strongly matched an instrument
prompt squeezed every mood score towards zero, and mood scores rose and fell
with unrelated instrument/section prompts. Now the 12 mood probabilities of a
window sum to 1, as do the 7 section and 8 instrument probabilities.
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path

from common import ALL_TAGS, MOOD_TAGS, PROCESSED, SECTION_TAGS, TAG_GROUPS, display_path, record_run

MODEL_NAME = "laion/clap-htsat-fused"


def group_softmax(logits, groups: dict[str, list[str]], tags: list[str]):
    """Softmax over each tag group separately. logits: (n_windows, n_tags) array."""
    import numpy as np
    logits = np.asarray(logits, dtype=np.float64)
    probs = np.zeros_like(logits)
    for group_tags in groups.values():
        idx = [tags.index(t) for t in group_tags]
        g = logits[:, idx]
        g = np.exp(g - g.max(axis=1, keepdims=True))
        probs[:, idx] = g / g.sum(axis=1, keepdims=True)
    return probs


def load_clap(device: str):
    """Load CLAP model + processor from HuggingFace."""
    from transformers import ClapModel, ClapProcessor
    print(f"[info] loading {MODEL_NAME} on {device} ...")
    processor = ClapProcessor.from_pretrained(MODEL_NAME)
    model = ClapModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return model, processor


def slice_audio(audio_path: Path, window_sec: float = 5.0):
    """Yield (start_sec, end_sec, audio_array, sample_rate) per window."""
    import soundfile as sf
    data, sr = sf.read(str(audio_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # mono
    if sr != 48000:  # CLAP expects 48 kHz
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=48000)
        sr = 48000
    win = int(window_sec * sr)
    for i in range(0, len(data), win):
        chunk = data[i: i + win]
        if len(chunk) < win // 2:  # skip trailing half-windows
            break
        yield round(i / sr, 3), round((i + len(chunk)) / sr, 3), chunk, sr


def score_windows(model, processor, chunks: list, sr: int, device: str):
    """(n_chunks, n_tags) per-group probabilities for a batch of audio chunks."""
    import torch
    inputs = processor(audio=chunks, sampling_rate=sr, text=ALL_TAGS,
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits_per_audio.cpu().numpy()  # (n_chunks, n_tags)
    return group_softmax(logits, TAG_GROUPS, ALL_TAGS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", type=float, default=5.0, help="window size in seconds")
    parser.add_argument("--batch", type=int, default=8, help="windows per forward pass")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    audio_path = PROCESSED / "audio.wav"
    if not audio_path.exists():
        print(f"[error] audio not found at {audio_path}")
        return 1

    model, processor = load_clap(args.device)

    rows = []
    t0 = time.time()
    print(f"[info] slicing audio into {args.window}s windows ...")
    batch = []

    def flush():
        probs = score_windows(model, processor, [c for _, _, c in batch], 48000, args.device)
        for (start, end, _), p in zip(batch, probs):
            rows.append({"start_sec": start, "end_sec": end,
                         **{tag: round(float(v), 5) for tag, v in zip(ALL_TAGS, p)}})
        batch.clear()
        elapsed = time.time() - t0
        print(f"  [{len(rows)}] windows ({elapsed:.0f}s, {elapsed / len(rows):.2f}s/window)")

    for start, end, chunk, _sr in slice_audio(audio_path, args.window):
        batch.append((start, end, chunk))
        if len(batch) >= args.batch:
            flush()
    if batch:
        flush()
    elapsed = time.time() - t0

    out_csv = PROCESSED / "audio_clap.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["start_sec", "end_sec"] + ALL_TAGS)
        w.writeheader()
        w.writerows(rows)
    out_json = PROCESSED / "audio_clap.json"
    out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\n[ok] wrote {display_path(out_csv)} ({len(rows)} windows × {len(ALL_TAGS)} tags)")
    print(f"[stats] {elapsed:.0f}s total ({elapsed / max(1, len(rows)):.2f}s/window)")

    if rows:
        def avg(t):
            return sum(r[t] for r in rows) / len(rows)
        print(f"\n[sanity] avg-highest mood: '{max(MOOD_TAGS, key=avg)}'")
        print(f"[sanity] avg-highest section: '{max(SECTION_TAGS, key=avg)}'")
        # Signal check: a tag whose probability barely moves across windows carries
        # no time information. With per-group softmax, the uniform level is
        # 1/len(group), so report the spread (max - min) instead of the max.
        print("\n[spread] max-min probability per tag across windows:")
        for group, tags in TAG_GROUPS.items():
            spreads = sorted(((max(r[t] for r in rows) - min(r[t] for r in rows), t) for t in tags), reverse=True)
            flat = [t for s, t in spreads if s < 0.05]
            print(f"  {group}: {len(tags) - len(flat)}/{len(tags)} tags vary by ≥0.05"
                  + (f" (flat: {', '.join(flat)})" if flat else ""))

    record_run(5, inputs=[audio_path], outputs=[out_csv, out_json],
               params={"model": MODEL_NAME, "window_sec": args.window, "softmax": "per-group"})
    print("\n[next] Phase 6: python scripts/phase6_music.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
