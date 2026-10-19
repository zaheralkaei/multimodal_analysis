"""
Phase 5 — Audio tagging with CLAP (via transformers).

Reads:  data/<video_id>/audio.wav (from Phase 0)
Writes: data/<video_id>/audio_clap.csv — per-window CLAP similarity scores
        data/<video_id>/audio_clap.json — same as JSON

Uses laion/clap-htsat-fused via transformers (no separate CLAP install needed).

For each 5-second audio window, compute similarity against a fixed vocabulary
of mood / section / instrument tags (from _clap_tags.py, shared with phases
7 and 8). Output is a per-time feature matrix you can correlate with visual
analysis.

Round 8 fixes (methodology review R2/R3):
  - Reads the 48 kHz full-band `audio_music.wav` from phase 0. The old input
    was the 16 kHz speech file upsampled to 48 kHz, i.e. nothing above 8 kHz.
  - Softmax is taken WITHIN each tag group (mood | section | instrument).
    One softmax over all 27 tags made "piano" compete with "sad", so a
    window's mood probabilities depended on how instrument-like it sounded.
  - Raw cosine similarities are written too (audio_clap_cosine.csv): the
    softmax only ranks tags against each other, it is not presence.
  - Section tags from an isolated 5 s window remain a weak signal; they are
    kept for continuity but should not be reported as song structure.
"""
from __future__ import annotations
from _clap_tags import ALL_TAGS, MOOD_TAGS, SECTION_TAGS, INSTRUMENT_TAGS, TAG_GROUPS

import argparse, json, os, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
from _paths import disp
PROCESSED = REPO_ROOT / "data" / "processed"
if "PROCESSED_DIR" in os.environ:
    PROCESSED = Path(os.environ["PROCESSED_DIR"])


def load_clap(device: str = "cpu"):
    """Load CLAP model + processor from HuggingFace on the given device."""
    from transformers import ClapModel, ClapProcessor
    model_name = "laion/clap-htsat-fused"
    print(f"[info] loading {model_name} on {device} ...")
    processor = ClapProcessor.from_pretrained(model_name)
    model = ClapModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, processor


def pick_audio(processed: Path) -> tuple[Path, bool]:
    """Prefer the full-band music file; fall back (flagged) to the 16 kHz one."""
    music = processed / "audio_music.wav"
    if music.exists():
        return music, False
    return processed / "audio.wav", True


def group_softmax(logits, groups: dict, tags: list):
    """Softmax over each tag group separately. logits: (n_windows, n_tags)."""
    import numpy as np
    logits = np.asarray(logits, dtype=np.float64)
    out = np.zeros_like(logits)
    for names in groups.values():
        idx = [tags.index(t) for t in names]
        sub = logits[:, idx]
        sub = sub - sub.max(axis=1, keepdims=True)
        e = np.exp(sub)
        out[:, idx] = e / e.sum(axis=1, keepdims=True)
    return out


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


def compute_batch_logits(model, processor, chunks: list, sr: int):
    """(n_chunks, n_tags) CLAP logits (= cosine * logit_scale) and the scale."""
    import torch
    inputs = processor(audio=chunks, sampling_rate=sr, text=ALL_TAGS,
                       return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scale = float(model.logit_scale_a.exp())
    return outputs.logits_per_audio.detach().cpu().numpy(), scale


def compute_window_similarities(model, processor, audio_chunk, sr) -> dict:
    """{tag: per-group probability} for one chunk (kept for callers/tests)."""
    logits, _ = compute_batch_logits(model, processor, [audio_chunk], sr)
    probs = group_softmax(logits, TAG_GROUPS, ALL_TAGS)[0]
    return {tag: float(p) for tag, p in zip(ALL_TAGS, probs)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", type=float, default=5.0, help="window size in seconds")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    audio_path, band_limited = pick_audio(PROCESSED)
    if not audio_path.exists():
        print(f"[error] audio not found at {audio_path}")
        return 1
    if band_limited:
        print("[warn] audio_music.wav missing — using the 16 kHz speech track "
              "(no content above 8 kHz). Re-run phase 0 for full-band audio.")
    print(f"[info] audio: {disp(audio_path)}")

    model, processor = load_clap(device=args.device)

    import csv
    out_csv = PROCESSED / "audio_clap.csv"
    out_json = PROCESSED / "audio_clap.json"

    import numpy as np
    t0 = time.time()
    print(f"[info] slicing audio into {args.window}s windows ...")
    windows = list(slice_audio(audio_path, args.window))
    all_logits, scale = [], 1.0
    for b in range(0, len(windows), 8):
        batch = windows[b:b + 8]
        logits, scale = compute_batch_logits(model, processor, [w[2] for w in batch], batch[0][3])
        all_logits.append(logits)
        print(f"  [{min(b + 8, len(windows))}/{len(windows)}] windows "
              f"({time.time() - t0:.0f}s)", flush=True)
    logits = np.concatenate(all_logits) if all_logits else np.zeros((0, len(ALL_TAGS)))
    probs = group_softmax(logits, TAG_GROUPS, ALL_TAGS)
    cosine = logits / scale
    rows, cos_rows = [], []
    for (start, end, _c, _sr), pr, cs in zip(windows, probs, cosine):
        rows.append({"start_sec": start, "end_sec": end,
                     **{t: round(float(v), 5) for t, v in zip(ALL_TAGS, pr)}})
        cos_rows.append({"start_sec": start, "end_sec": end,
                         **{t: round(float(v), 5) for t, v in zip(ALL_TAGS, cs)}})
    elapsed = time.time() - t0

    cols = ["start_sec", "end_sec"] + ALL_TAGS
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\n[ok] wrote {disp(out_csv)} ({len(rows)} windows × {len(ALL_TAGS)} tags)")
    print(f"[stats] {elapsed:.0f}s total ({elapsed/max(1,len(rows)):.1f}s/window)")

    out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (PROCESSED / "audio_clap_cosine.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(cos_rows)
    (PROCESSED / "audio_clap_stats.json").write_text(json.dumps({
        "model": "laion/clap-htsat-fused",
        "audio_file": disp(audio_path),
        "band_limited_16k": band_limited,
        "window_sec": args.window,
        "normalization": "softmax within each tag group (mood | section | instrument)",
        "logit_scale": round(scale, 4),
        "tag_groups": TAG_GROUPS,
        "n_windows": len(rows),
    }, indent=2) + "\n", encoding="utf-8")

    # Quick sanity check
    if rows:
        max_mood = max(MOOD_TAGS, key=lambda t: sum(r[t] for r in rows)/len(rows))
        max_section = max(SECTION_TAGS, key=lambda t: sum(r[t] for r in rows)/len(rows))
        print(f"\n[sanity] avg-highest mood: '{max_mood}'")
        print(f"[sanity] avg-highest section: '{max_section}'")

        # Tag activity report (round-2 audit finding): a tag is "active" if
        # it ever wins clearly — max score above twice its group's uniform
        # level (2/len(group)); probabilities are per-group since round 8.
        group_of = {t: g for g, names in TAG_GROUPS.items() for t in names}
        active = []
        inactive = []
        for tag in ALL_TAGS:
            max_score = max(float(r[tag]) for r in rows)
            if max_score > 2.0 / len(TAG_GROUPS[group_of[tag]]):
                active.append((tag, max_score))
            else:
                inactive.append((tag, max_score))
        print(f"\n[active] {len(active)}/{len(ALL_TAGS)} tags exceed 2x their group's uniform level:")
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
