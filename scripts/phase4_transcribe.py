"""
Phase 4 — Speech/lyrics transcription with faster-whisper.

Reads:  <PROCESSED>/audio.wav (from Phase 0)
Writes: <PROCESSED>/transcript.csv  — per-segment transcript with timestamps
        <PROCESSED>/transcript.json — segments, each with word-level timestamps
        <PROCESSED>/transcript_words.csv — one row per word (start, end, prob)
        <PROCESSED>/vocals.wav — isolated vocals (only with --separate-vocals)

Uses faster-whisper (CTranslate2 backend, ~4x faster than openai-whisper).
Default model: small (multilingual, auto-detects language). For better
lyrics quality use medium or large-v3.

Whisper is trained on speech, not music. Sung vocals over loud instruments are
its main failure mode. --separate-vocals first runs Demucs (htdemucs) to strip
the instruments and transcribes only the vocal stem, which typically recovers
many lyrics that VAD otherwise drops. It needs `pip install demucs` and
downloads ~80 MB of weights on first use; slow on CPU (~1x real time).
"""
from __future__ import annotations
import argparse, csv, json, subprocess, sys, time
from pathlib import Path

from common import PROCESSED, display_path, record_run

DEFAULT_MODEL = "small"  # keep in sync with run_pipeline.py --whisper-model


def separate_vocals(audio_path: Path, out_path: Path) -> Path:
    """Run Demucs two-stem separation and return the vocals WAV path."""
    try:
        import demucs  # noqa: F401
    except ImportError:
        raise SystemExit("[error] --separate-vocals needs demucs: pip install demucs") from None
    tmp = out_path.parent / "_demucs"
    print("[info] separating vocals with demucs (htdemucs) ...")
    subprocess.run([sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs",
                    "-o", str(tmp), str(audio_path)], check=True)
    stem = tmp / "htdemucs" / audio_path.stem / "vocals.wav"
    stem.replace(out_path)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[ok] vocals stem: {display_path(out_path)}")
    return out_path


def transcribe(audio_path: Path, model_size: str = DEFAULT_MODEL,
               language: str | None = None, beam_size: int = 5) -> tuple[list[dict], dict]:
    """Run faster-whisper. Returns (segments with words, info dict)."""
    from faster_whisper import WhisperModel

    print(f"[info] loading whisper model: {model_size} (cpu int8)")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print(f"[info] transcribing {audio_path.name} ...")
    t0 = time.time()
    segments, info = model.transcribe(
        str(audio_path), language=language, beam_size=beam_size,
        vad_filter=True,  # skip non-speech segments
        word_timestamps=True,
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
            "words": [{"start_sec": round(w.start, 3), "end_sec": round(w.end, 3),
                       "word": w.word.strip(), "probability": round(w.probability, 3)}
                      for w in (seg.words or [])],
        })
    elapsed = time.time() - t0
    print(f"[ok] {len(out)} segments, language={info.language}, "
          f"prob={info.language_probability:.2f}, duration={info.duration:.0f}s")
    print(f"[stats] transcribed in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    return out, {"language": info.language, "language_probability": round(info.language_probability, 3)}


SEGMENT_COLS = ["start_sec", "end_sec", "duration_sec", "text", "avg_logprob",
                "no_speech_prob", "compression_ratio"]


def write_outputs(segments: list[dict], out_dir: Path) -> list[Path]:
    out_csv = out_dir / "transcript.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SEGMENT_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(segments)
    words_csv = out_dir / "transcript_words.csv"
    with words_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["segment_idx", "start_sec", "end_sec", "word", "probability"])
        w.writeheader()
        for i, seg in enumerate(segments):
            for word in seg.get("words", []):
                w.writerow({"segment_idx": i, **word})
    out_json = out_dir / "transcript.json"
    out_json.write_text(json.dumps(segments, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return [out_csv, words_csv, out_json]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="whisper model: tiny, base, small, medium, large-v3 (or *.en variants)")
    parser.add_argument("--language", default=None,
                        help="force language (e.g. 'en'); default = auto-detect")
    parser.add_argument("--separate-vocals", action="store_true",
                        help="isolate vocals with Demucs before transcribing (needs `pip install demucs`)")
    args = parser.parse_args()

    audio_path = PROCESSED / "audio.wav"
    if not audio_path.exists():
        print(f"[error] audio not found at {audio_path}")
        print("  run phase 0 first: python scripts/phase0_input.py <source>")
        return 1

    source = audio_path
    if args.separate_vocals:
        source = separate_vocals(audio_path, PROCESSED / "vocals.wav")

    segments, info = transcribe(source, args.model, args.language)
    outputs = write_outputs(segments, PROCESSED)
    for p in outputs:
        print(f"[ok] wrote {display_path(p)}")

    if segments:
        n_words = sum(len(s["words"]) for s in segments)
        avg_conf = sum(s["avg_logprob"] for s in segments) / len(segments)
        print(f"\n[stats] {len(segments)} segments, {n_words} words "
              f"(avg logprob {avg_conf:.2f}; "
              f"avg no-speech prob {sum(s['no_speech_prob'] for s in segments) / len(segments):.2f})")
        print("\n[sample] first 5 segments:")
        for s in segments[:5]:
            print(f"  {s['start_sec']:>6.2f}-{s['end_sec']:>6.2f}s  {s['text'][:80]}")
    else:
        print("\n[note] 0 segments detected. Quiet, whispered or heavily produced vocals are often")
        print("       filtered out by Whisper's VAD — try --separate-vocals.")

    record_run(4, inputs=[audio_path], outputs=outputs,
               params={"model": args.model, "language": args.language,
                       "separate_vocals": args.separate_vocals, **info})
    print("\n[next] Phase 5: python scripts/phase5_audio.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
