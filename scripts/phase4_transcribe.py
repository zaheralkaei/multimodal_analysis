"""
Phase 4 — Speech/lyrics transcription with faster-whisper.

Reads:  data/processed/audio.wav (from Phase 0)
Writes: data/processed/transcript.csv — per-segment transcript with timestamps
        data/processed/transcript.json — same as JSON

Uses faster-whisper (CTranslate2 backend, ~4x faster than openai-whisper).
Default model: small.en (good for English music, ~2GB RAM). For better
lyrics quality use medium or large-v3.

Note: Whisper is trained on speech, not music. On sung lyrics it often works
"well enough" but quality varies. For music videos with heavy reverb or
rap, results may be noisy — check the output manually.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"
if "PROCESSED_DIR" in os.environ:
    PROCESSED = Path(os.environ["PROCESSED_DIR"])


def transcribe(audio_path: Path, model_size: str = "small",
               language: str | None = None, beam_size: int = 5) -> list[dict]:
    """Run faster-whisper. Returns list of segment dicts.

    Default model is 'small' (multilingual, auto-detects language).
    Use 'small.en' for English-only (smaller, faster).
    """
    from faster_whisper import WhisperModel

    print(f"[info] loading whisper model: {model_size} (cpu int8)")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print(f"[info] transcribing {audio_path.name} ...")
    t0 = time.time()
    segments, info = model.transcribe(
        str(audio_path), language=language, beam_size=beam_size,
        vad_filter=True,  # skip non-speech segments
    )
    out = []
    for seg in segments:
        out.append({
            "start_sec": round(seg.start, 3),
            "end_sec": round(seg.end, 3),
            "duration_sec": round(seg.end - seg.start, 3),
            "text": seg.text.strip(),
            "avg_logprob": round(seg.avg_logprob, 3),
            "no_speech_prob": round(seg.no_speech_prob, 3),
            "compression_ratio": round(seg.compression_ratio, 3),
        })
    elapsed = time.time() - t0
    print(f"[ok] {len(out)} segments, language={info.language}, "
          f"prob={info.language_probability:.2f}, duration={info.duration:.0f}s")
    print(f"[stats] transcribed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="small.en",
                       help="whisper model size: tiny.en, base.en, small.en, medium.en, large-v3")
    parser.add_argument("--language", default=None,
                       help="force language (e.g. 'en'); default = auto-detect")
    args = parser.parse_args()

    audio_path = PROCESSED / "audio.wav"
    if not audio_path.exists():
        print(f"[error] audio not found at {audio_path}")
        print("  run phase 0 first: python scripts/phase0_input.py <source>")
        return 1

    segments = transcribe(audio_path, args.model, args.language)

    # Write CSV
    import csv
    out_csv = PROCESSED / "transcript.csv"
    cols = ["start_sec", "end_sec", "duration_sec", "text", "avg_logprob",
            "no_speech_prob", "compression_ratio"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for seg in segments:
            w.writerow(seg)
    print(f"[ok] wrote {out_csv.relative_to(REPO_ROOT)} ({len(segments)} rows)")

    # Write JSON
    out_json = PROCESSED / "transcript.json"
    out_json.write_text(json.dumps(segments, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_json.relative_to(REPO_ROOT)}")

    # Sample stats
    if segments:
        non_empty = [s for s in segments if s["text"]]
        avg_conf = sum(s["avg_logprob"] for s in segments) / len(segments)
        print(f"\n[stats] {len(non_empty)}/{len(segments)} segments have text "
              f"(avg logprob {avg_conf:.2f}; "
              f"avg no-speech prob {sum(s['no_speech_prob'] for s in segments)/len(segments):.2f})")
            # Daydreaming is famously quiet/whispered - Whisper often returns 0 segments.
    # This is a real characteristic, not a bug.
    if not segments:
        print("\n[note] 0 segments detected. Daydreaming is whispered/sung at low volume")
        print("       — Whisper's VAD often filters out all speech. See README for this finding.")
    elif segments[:5]:
            print("\n[sample] first 5 segments:")
            for s in segments[:5]:
                print(f"  {s['start_sec']:>6.2f}-{s['end_sec']:>6.2f}s  {s['text'][:80]}")

    print(f"\n[next] Phase 5: python scripts/phase5_audio.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
