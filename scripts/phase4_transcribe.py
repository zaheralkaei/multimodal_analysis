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

Round 11 (cloud audio stack):
  - --input auto uses the isolated vocal stem (vocals.wav from
    separate_vocals.py) when it exists, else the mix.
  - --backend auto uses Groq's hosted whisper-large-v3-turbo when
    GROQ_API_KEY is set (~$0.04 per audio hour, billed >= 10 s per request),
    else local faster-whisper. Long audio is sent in 10-minute chunks with
    5 s overlap; each segment is kept by the chunk that owns its midpoint.
    Responses are cached by audio content + settings.
  - Vocal gate: with a vocal stem, a segment whose stem is quieter than
    -40 dBFS throughout is flagged suspect ("no vocals") — Whisper writes
    words into silence, and on a stem silence is easy to measure.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
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


GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3-turbo"
CHUNK_SEC = 600.0
OVERLAP_SEC = 5.0
VOCAL_GATE_DBFS = -40.0


def chunk_plan(duration: float, chunk: float = CHUNK_SEC, overlap: float = OVERLAP_SEC) -> list:
    """[(cut_start, cut_end, own_start, own_end)]: audio sent per request and
    the part of the timeline whose segments that request owns."""
    if duration <= chunk:
        return [(0.0, duration, 0.0, duration)]
    plan, start = [], 0.0
    while start < duration:
        end = min(duration, start + chunk)
        own_s = 0.0 if start == 0 else start + overlap / 2
        own_e = duration if end >= duration else end - overlap / 2
        plan.append((start, end, own_s, own_e))
        if end >= duration:
            break
        start = end - overlap
    return plan


def parse_groq(resp: dict, offset: float, own: tuple) -> list:
    """verbose_json -> our segment dicts (times shifted by the chunk offset)."""
    words = resp.get("words") or []
    out = []
    for seg in resp.get("segments") or []:
        s, e = float(seg["start"]) + offset, float(seg["end"]) + offset
        mid = (s + e) / 2
        if not (own[0] <= mid < own[1]):
            continue
        row = {
            "start_sec": round(s, 3), "end_sec": round(e, 3), "duration_sec": round(e - s, 3),
            "text": (seg.get("text") or "").strip(),
            "avg_logprob": round(float(seg.get("avg_logprob", 0) or 0), 3),
            "no_speech_prob": round(float(seg.get("no_speech_prob", 0) or 0), 3),
            "compression_ratio": round(float(seg.get("compression_ratio", 0) or 0), 3),
        }
        row["words"] = [{"start": round(float(w["start"]) + offset, 3),
                         "end": round(float(w["end"]) + offset, 3),
                         "word": w.get("word", ""), "p": None}
                        for w in words
                        if float(seg["start"]) <= float(w["start"]) < float(seg["end"])]
        out.append(row)
    return out


def transcribe_groq(audio_path: Path, language: str | None, processed: Path,
                    model: str = GROQ_MODEL) -> list:
    import io
    import numpy as np
    import soundfile as sf
    import _cloud
    key = _cloud.get_key("GROQ_API_KEY")
    if not key:
        raise _cloud.CloudError("GROQ_API_KEY is not set (.env)")
    y, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != 16000:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    dur = len(y) / sr
    price = _cloud.PRICES["groq_whisper_turbo_per_hour" if "turbo" in model
                          else "groq_whisper_v3_per_hour"]
    est_total = sum(max(10.0, c[1] - c[0]) for c in chunk_plan(dur)) / 3600 * price
    ok, why = _cloud.budget_ok(processed, est_total)
    if not ok:
        raise _cloud.CloudError(why)
    out, lang = [], None
    for cut_s, cut_e, own_s, own_e in chunk_plan(dur):
        buf = io.BytesIO()
        sf.write(buf, y[int(cut_s * sr):int(cut_e * sr)], sr, format="FLAC")
        audio = buf.getvalue()
        ck = _cloud.cache_key("groq", _cloud.sha_bytes(audio), model, language, "v1")
        resp = _cloud.cache_get_json("groq", ck)
        est = max(10.0, cut_e - cut_s) / 3600 * price
        if resp is None:
            data = [("model", model), ("response_format", "verbose_json"), ("temperature", "0"),
                    ("timestamp_granularities[]", "segment"), ("timestamp_granularities[]", "word")]
            if language:
                data.append(("language", language))
            resp = _cloud.request("POST", GROQ_URL, headers={"Authorization": f"Bearer {key}"},
                                  data=data, files={"file": ("audio.flac", audio, "audio/flac")},
                                  timeout=600)
            _cloud.cache_put_json("groq", ck, resp)
            _cloud.log_usage(processed, "groq", model, est, cached=False,
                             detail={"chunk": [round(cut_s, 1), round(cut_e, 1)]})
        else:
            _cloud.log_usage(processed, "groq", model, est, cached=True,
                             detail={"chunk": [round(cut_s, 1), round(cut_e, 1)]})
        lang = lang or resp.get("language")
        out += parse_groq(resp, cut_s, (own_s, own_e))
    for row in out:
        row["suspect"] = is_suspect(row)
    transcribe_groq.info = {"language": lang, "language_probability": None, "model": model,
                            "backend": "groq", "temperature": 0, "chunk_sec": CHUNK_SEC,
                            "overlap_sec": OVERLAP_SEC}
    print(f"[ok] groq {model}: {len(out)} segments, language={lang}, "
          f"~${est_total:.4f}")
    return out


def apply_vocal_gate(segments: list, activity_csv: Path) -> int:
    """Flag segments during which the vocal stem is silent. Returns count."""
    import csv
    if not activity_csv.exists():
        return 0
    with activity_csv.open(encoding="utf-8") as f:
        act = [(float(r["start_sec"]), float(r["end_sec"]), float(r["rms_dbfs"]))
               for r in csv.DictReader(f)]
    n = 0
    for seg in segments:
        s, e = float(seg["start_sec"]), float(seg["end_sec"])
        loud = [db for a, b, db in act if a < e and b > s]
        if loud and max(loud) < VOCAL_GATE_DBFS:
            seg["suspect"] = True
            seg["suspect_reason"] = "vocal stem silent"
            n += 1
    return n


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
    parser.add_argument("--backend", default="auto", choices=["auto", "local", "groq"],
                        help="auto = groq if GROQ_API_KEY is set, else local faster-whisper")
    parser.add_argument("--input", default="auto", choices=["auto", "vocals", "mix"],
                        help="auto = vocal stem (vocals.wav) if separation ran, else the mix")
    args = parser.parse_args()

    vocals = PROCESSED / "vocals.wav"
    if args.input == "vocals" or (args.input == "auto" and vocals.exists()):
        audio_path, source = vocals, "vocal stem"
    else:
        audio_path, source = PROCESSED / "audio.wav", "mix"
    if not audio_path.exists():
        print(f"[error] audio not found at {audio_path}")
        print("  run phase 0 (and separate_vocals.py for --input vocals) first")
        return 1
    import _cloud
    backend = args.backend
    if backend == "auto":
        backend = "groq" if _cloud.get_key("GROQ_API_KEY") else "local"
    print(f"[info] transcribing the {source} ({disp(audio_path)}) with {backend}")

    if backend == "groq":
        try:
            segments = transcribe_groq(audio_path, args.language, PROCESSED)
            info = transcribe_groq.info
        except _cloud.CloudError as e:
            print(f"[error] groq transcription failed: {e}")
            return 1
    else:
        segments = transcribe(audio_path, args.model, args.language)
        info = dict(getattr(transcribe, "info", {}), backend="local")
    n_gated = apply_vocal_gate(segments, PROCESSED / "vocal_activity.csv") if source == "vocal stem" else 0
    if n_gated:
        print(f"[info] vocal gate: {n_gated} segments flagged (vocal stem silent)")

    # Write CSV
    import csv
    out_csv = PROCESSED / "transcript.csv"
    cols = ["start_sec", "end_sec", "duration_sec", "text", "avg_logprob",
            "no_speech_prob", "compression_ratio", "suspect", "suspect_reason"]
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
    stats = dict(info)
    stats.update({"n_segments": len(segments), "n_suspect": n_suspect,
                  "input": source, "n_vocal_gated": n_gated,
                  "suspect_rule": {"compression_ratio_max": COMPRESSION_RATIO_MAX,
                                   "no_speech_min": NO_SPEECH_MIN,
                                   "logprob_min": LOGPROB_MIN},
                  "audio_file": disp(audio_path)})
    (PROCESSED / "transcript_stats.json").write_text(json.dumps(stats, indent=2) + "\n",
                                                     encoding="utf-8")
    from _lineage import record
    record(PROCESSED, "phase4", [audio_path],
           [out_csv, out_json, PROCESSED / "transcript_stats.json"],
           {"backend": backend, "model": info.get("model"), "language": args.language,
            "input": source}, __file__)
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
