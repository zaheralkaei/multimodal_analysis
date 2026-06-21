"""
Phase 6 — Music structure analysis with librosa.

Reads:  data/processed/audio.wav (from Phase 0)
Writes: data/processed/music_features.csv — per-second features
        data/processed/music_summary.json — global stats (tempo, key, beats)

Extracts:
  - tempo (BPM)
  - beat positions (downbeats)
  - estimated musical key
  - per-second: RMS energy, spectral centroid (brightness), spectral contrast,
    zero-crossing rate (noisiness)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"


Krumhansl_SCHMUCKLER = {
    # Major profile
    "C":  [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    # Minor profile (natural minor)
    "Cm": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
}


def estimate_key(chroma_mean):
    """Estimate musical key using Krumhansl-Schmuckler."""
    import numpy as np
    PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    best_score = -np.inf
    best_key = "C major"
    # rotate chroma and try major + minor for each
    for tonic_idx in range(12):
        chroma_rotated = np.roll(chroma_mean, -tonic_idx)
        # Compare to major
        maj_score = np.corrcoef(chroma_rotated, Krumhansl_SCHMUCKLER["C"])[0, 1]
        if maj_score > best_score:
            best_score = maj_score
            best_key = f"{PITCH_NAMES[tonic_idx]} major"
        # Compare to minor
        min_score = np.corrcoef(chroma_rotated, Krumhansl_SCHMUCKLER["Cm"])[0, 1]
        if min_score > best_score:
            best_score = min_score
            best_key = f"{PITCH_NAMES[tonic_idx]} minor"
    return best_key, float(best_score)


def analyze_music(audio_path: Path):
    """Extract music features. Returns (per_second_rows, summary_dict)."""
    import librosa
    import numpy as np
    import csv

    print(f"[info] loading {audio_path.name} with librosa ...")
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"[info] duration: {duration:.1f}s, sr: {sr}")

    # Tempo + beat positions
    print("[info] extracting tempo + beats ...")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    print(f"[info] tempo: {float(tempo):.1f} BPM, {len(beat_times)} beats")

    # Key estimation via chroma
    print("[info] estimating key ...")
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    key, key_corr = estimate_key(chroma_mean)
    print(f"[info] estimated key: {key} (corr {key_corr:.2f})")

    # Per-second features
    print("[info] extracting per-second features ...")
    n_seconds = int(duration)
    rows = []
    for sec in range(n_seconds):
        start = sec
        end = sec + 1
        y_seg = y[sr * start : sr * end]
        if len(y_seg) < sr // 2:
            continue
        rms = float(librosa.feature.rms(y=y_seg).mean())
        centroid = float(librosa.feature.spectral_centroid(y=y_seg, sr=sr).mean())
        zcr = float(librosa.feature.zero_crossing_rate(y_seg).mean())
        contrast = float(librosa.feature.spectral_contrast(y=y_seg, sr=sr).mean())
        # beat in this second?
        beats_in_sec = [round(float(b), 3) for b in beat_times if start <= b < end]
        rows.append({
            "second": sec,
            "start_sec": start,
            "end_sec": end,
            "rms_energy": round(rms, 5),
            "spectral_centroid": round(centroid, 1),
            "zero_crossing_rate": round(zcr, 5),
            "spectral_contrast": round(contrast, 3),
            "n_beats": len(beats_in_sec),
        })
    summary = {
        "duration_sec": round(duration, 3),
        "sample_rate": sr,
        "tempo_bpm": round(float(tempo), 2),
        "n_beats": len(beat_times),
        "key": key,
        "key_correlation": round(key_corr, 3),
        "beat_times": [round(float(b), 3) for b in beat_times.tolist()],
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args = parser.parse_args()

    audio_path = PROCESSED / "audio.wav"
    if not audio_path.exists():
        print(f"[error] audio not found at {audio_path}")
        return 1

    rows, summary = analyze_music(audio_path)

    # Write per-second CSV
    import csv
    out_csv = PROCESSED / "music_features.csv"
    cols = ["second", "start_sec", "end_sec", "rms_energy", "spectral_centroid",
            "zero_crossing_rate", "spectral_contrast", "n_beats"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[ok] wrote {out_csv.relative_to(REPO_ROOT)} ({len(rows)} rows)")

    # Write summary JSON (without beat_times in printed summary for brevity)
    out_json = PROCESSED / "music_summary.json"
    printed_summary = {k: v for k, v in summary.items() if k != "beat_times"}
    printed_summary["n_beat_times_in_json"] = len(summary["beat_times"])
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_json.relative_to(REPO_ROOT)}")

    print(f"\n[stats] tempo: {summary['tempo_bpm']} BPM, key: {summary['key']}, "
          f"{summary['n_beats']} beats across {summary['duration_sec']:.0f}s")

    print(f"\n[next] Phase 7: python scripts/phase7_sync.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
