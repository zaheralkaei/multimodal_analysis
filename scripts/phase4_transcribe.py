"""
Phase 4 — Speech/lyrics transcription with faster-whisper.

Reads:  data/<video_id>/audio.wav (from Phase 0)
Writes: data/<video_id>/transcript.csv — per-segment transcript with timestamps
        data/<video_id>/transcript.json — same as JSON

Uses faster-whisper (CTranslate2 backend, ~4x faster than openai-whisper).
Default model: small (multilingual, auto-detects language — good for music
videos in any language, ~2GB RAM). For English-only speed use small.en;
for better lyrics quality use medium or large-v3.

Note: Whisper is trained on speech, not music. On sung lyrics it often works
"well enough" but quality varies. For music videos with heavy reverb or
rap, results may be noisy — check the output manually.

Round 8 (analysis validity):
  - Every segment gets a `suspect` flag using Whisper's own decoding
    heuristics (the thresholds openai/whisper uses to reject a decode):
      compression_ratio > 2.4  (repetition loops), or
      no_speech_prob > 0.6 AND avg_logprob < -1.0  (text over non-speech).
    Suspect segments are KEPT in the files (auditable) but phase 7 excludes
    them from lyric counts.
  - condition_on_previous_text=False: stops one hallucination from seeding
    the next window (the classic repeated-line loop on music).
  - Word-level timestamps are stored in transcript.json (`words`).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
from _paths import disp
PROCESSED = REPO_ROOT / "data" / "processed"
if "PROCESSED_DIR" in os.environ:
    PROCESSED = Path(os.environ["PROCESSED_DIR"])


COMPRESSION_RATIO_MAX = 2.4
NO_SPEECH_MIN = 0.6
LOGPROB_MIN = -1.0


def is_suspect(seg: dict) -> bool:
    """Whisper's own failure heuristics (see module docstring)."""
    if float(seg.get("compression_ratio", 0) or 0) > COMPRESSION_RATIO_MAX:
        return True
    return (float(seg.get("no_speech_prob", 0) or 0) > NO_SPEECH_MIN
            and float(seg.get("avg_logprob", 0) or 0) < LOGPROB_MIN)


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
        condition_on_previous_text=False,
        word_timestamps=True,
    )
    out = []
    for seg in segments:
        row = {
            "start_sec": round(seg.start, 3),
            "end_sec": round(seg.end, 3),
            "duration_sec": round(seg.end - seg.start, 3),
            "text": seg.text.strip(),
            "avg_logprob": round(seg.avg_logprob, 3),
            "no_speech_prob": round(seg.no_speech_prob, 3),
            "compression_ratio": round(seg.compression_ratio, 3),
        }
        row["suspect"] = is_suspect(row)
        row["words"] = [{"start": round(w.start, 3), "end": round(w.end, 3),
                         "word": w.word, "p": round(w.probability, 3)}
                        for w in (seg.words or [])]
        out.append(row)
    elapsed = time.time() - t0
    print(f"[ok] {len(out)} segments, language={info.language}, "
          f"prob={info.language_probability:.2f}, duration={info.duration:.0f}s")
    print(f"[stats] transcribed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    transcribe.info = {"language": info.language,
                       "language_probability": round(info.language_probability, 3),
                       "model": model_size, "beam_size": beam_size,
                       "vad_filter": True, "condition_on_previous_text": False}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="small",
                        help="whisper model size (default: small, multilingual; "
                             "English-only alternatives: tiny.en, base.en, small.en, medium.en; "
                             "higher quality: large-v3)")
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
            "no_speech_prob", "compression_ratio", "suspect"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for seg in segments:
            w.writerow(seg)
    print(f"[ok] wrote {disp(out_csv)} ({len(segments)} rows)")

    # Write JSON
    out_json = PROCESSED / "transcript.json"
    out_json.write_text(json.dumps(segments, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[ok] wrote {disp(out_json)}")
    n_suspect = sum(1 for s in segments if s["suspect"])
    stats = dict(getattr(transcribe, "info", {}))
    stats.update({"n_segments": len(segments), "n_suspect": n_suspect,
                  "suspect_rule": {"compression_ratio_max": COMPRESSION_RATIO_MAX,
                                   "no_speech_min": NO_SPEECH_MIN,
                                   "logprob_min": LOGPROB_MIN},
                  "audio_file": disp(audio_path)})
    (PROCESSED / "transcript_stats.json").write_text(json.dumps(stats, indent=2) + "\n",
                                                     encoding="utf-8")
    from _lineage import record
    record(PROCESSED, "phase4", [audio_path],
           [out_csv, out_json, PROCESSED / "transcript_stats.json"],
           {"model": args.model, "language": args.language}, __file__)
    if n_suspect:
        print(f"[warn] {n_suspect}/{len(segments)} segments flagged suspect "
              f"(likely hallucination; excluded from lyric stats in phase 7)")

    # Sample stats
    if segments:
        non_empty = [s for s in segments if s["text"]]
        avg_conf = sum(s["avg_logprob"] for s in segments) / len(segments)
        print(f"\n[stats] {len(non_empty)}/{len(segments)} segments have text "
              f"(avg logprob {avg_conf:.2f}; "
              f"avg no-speech prob {sum(s['no_speech_prob'] for s in segments)/len(segments):.2f})")
    if not segments:
        print("\n[note] 0 segments detected — no speech/vocals found by the VAD, or the")
        print("       vocals are too quiet/processed for Whisper. Check the audio manually.")
    elif segments[:5]:
            print("\n[sample] first 5 segments:")
            for s in segments[:5]:
                print(f"  {s['start_sec']:>6.2f}-{s['end_sec']:>6.2f}s  {s['text'][:80]}")

    print(f"\n[next] Phase 5: python scripts/phase5_audio.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
