"""
Route — decide EARLY what kind of audio the video has (round 10).

Runs right after phase 0. Before round 10 the content type was decided only
in phase 8, after every analysis had already run, so a speech still got beat
tracking and a cut-on-beat test. Now this step listens to ~24 evenly spaced
5 s windows with CLAP's zero-shot music / speech / silence split and writes
data/<video_id>/content_route.json:

    {"audio_shares": {"music": .., "speech": .., "silence": ..},
     "audio_class": "music" | "speech" | "mixed" | "silent",
     "has_music": bool, "has_music_basis": "clap" | "override" | "unverified",
     "modules": {"beat_analysis": bool, "cut_on_beat_test": bool,
                 "music_mood": bool, ...},
     "reasons": [...]}

Later phases read it: phase 6 marks beats/key invalid without music, phase 7
skips the cut-on-beat test and masks per-shot audio mood, phase 8 shows it.

Note: CLAP files SUNG vocals under "music", so speech_share is spoken
content only. Transcription always runs.

    python scripts/route_content.py                     # auto
    python scripts/route_content.py --content-type song # explicit type wins
    python scripts/route_content.py --no-clap           # skip the model (unverified)
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from _paths import disp
from _clap_tags import AUDIO_TYPE_TAGS, window_audio_type, rms_dbfs
PROCESSED = REPO_ROOT / "data" / "processed"
if "PROCESSED_DIR" in os.environ:
    PROCESSED = Path(os.environ["PROCESSED_DIR"])

ROUTE_FILE = "content_route.json"
MUSIC_SHARE_MIN = 0.3          # same rule as _content_profile
N_WINDOWS = 24
WINDOW_SEC = 5.0

MUSIC_TYPES = {"song"}          # explicit --content-type values that imply music
NO_MUSIC_TYPES = {"speech"}


def classify_shares(shares: dict) -> str:
    m, s, q = shares.get("music", 0), shares.get("speech", 0), shares.get("silence", 0)
    if q >= 0.6:
        return "silent"
    if m >= 0.6:
        return "music"
    if s >= 0.6:
        return "speech"
    return "mixed"


def decide(shares: dict | None, content_type: str = "auto") -> dict:
    """Pure decision logic (tested): shares may be None when CLAP is skipped."""
    reasons = []
    if content_type in MUSIC_TYPES:
        has_music, basis = True, "override"
        reasons.append(f"--content-type {content_type}: music analysis on")
    elif content_type in NO_MUSIC_TYPES:
        has_music, basis = False, "override"
        reasons.append(f"--content-type {content_type}: music analysis off")
    elif shares is not None:
        has_music = shares["music"] >= MUSIC_SHARE_MIN
        basis = "clap"
        reasons.append(f"music in {shares['music'] * 100:.0f}% of sampled windows "
                       f"(threshold {MUSIC_SHARE_MIN * 100:.0f}%)")
    else:
        has_music, basis = True, "unverified"
        reasons.append("CLAP skipped — music presence unverified; music analyses run "
                       "but are flagged")
    if shares is not None and basis == "override":
        expected_music = content_type in MUSIC_TYPES
        if expected_music and shares["music"] < MUSIC_SHARE_MIN:
            reasons.append(f"WARNING: you set {content_type} but only "
                           f"{shares['music'] * 100:.0f}% of windows sound like music")
        if not expected_music and shares["music"] >= 0.7:
            reasons.append(f"WARNING: you set {content_type} but "
                           f"{shares['music'] * 100:.0f}% of windows sound like music")
    return {
        "audio_shares": shares,
        "audio_class": classify_shares(shares) if shares else "unknown",
        "has_music": has_music,
        "has_music_basis": basis,
        "content_type_override": None if content_type == "auto" else content_type,
        "modules": {
            "beat_analysis": has_music,
            "cut_on_beat_test": has_music,
            "music_mood": has_music,
            "key_detection": has_music,
            "transcription": True,
        },
        "reasons": reasons,
    }


def sample_windows(audio_path: Path, n: int = N_WINDOWS, window_sec: float = WINDOW_SEC):
    """Up to n evenly spaced full windows at 48 kHz: [(start_sec, chunk)]."""
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(str(audio_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 48000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=48000)
        sr = 48000
    win = int(window_sec * sr)
    n_full = len(data) // win
    if n_full == 0:
        return [(0.0, np.pad(data, (0, win - len(data))))], sr
    # spread m windows over the WHOLE file (round-10 calibration bug: the
    # spacing used n-1 instead of m-1, so a short video was only sampled in
    # its first windows and its speech section was never heard)
    m = min(n, n_full)
    idx = sorted({int(round(k * (n_full - 1) / max(1, m - 1))) for k in range(m)})
    return [(i * window_sec, data[i * win:(i + 1) * win]) for i in idx], sr


def measure_shares(audio_path: Path) -> tuple[dict, list]:
    import numpy as np
    from phase5_audio import load_clap, compute_batch_logits, group_softmax
    windows, sr = sample_windows(audio_path)
    model, processor = load_clap()
    logits = []
    for b in range(0, len(windows), 8):
        lg, _ = compute_batch_logits(model, processor, [w[1] for w in windows[b:b + 8]], sr,
                                     tags=AUDIO_TYPE_TAGS)
        logits.append(lg)
    probs = group_softmax(np.concatenate(logits), {"t": AUDIO_TYPE_TAGS}, AUDIO_TYPE_TAGS)
    kinds = [window_audio_type({**dict(zip(AUDIO_TYPE_TAGS, map(float, p))),
                                "rms_dbfs": rms_dbfs(w[1])})
             for p, w in zip(probs, windows)]
    shares = {t: round(kinds.count(t) / len(kinds), 3) for t in AUDIO_TYPE_TAGS}
    per_window = [{"start_sec": s, "type": k} for (s, _), k in zip(windows, kinds)]
    return shares, per_window


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--content-type", default="auto")
    ap.add_argument("--no-clap", action="store_true", help="do not load CLAP (result unverified)")
    args = ap.parse_args()

    audio = PROCESSED / "audio_music.wav"
    if not audio.exists():
        audio = PROCESSED / "audio.wav"
    if not audio.exists():
        print(f"[error] no audio in {disp(PROCESSED)} — run phase 0 first")
        return 1

    shares, per_window = (None, [])
    if not args.no_clap:
        print(f"[route] sampling up to {N_WINDOWS} x {WINDOW_SEC:g}s windows of {disp(audio)} with CLAP")
        shares, per_window = measure_shares(audio)
    route = decide(shares, args.content_type)
    meta_p = PROCESSED / "metadata.json"
    if meta_p.exists() and json.loads(meta_p.read_text(encoding="utf-8")).get("has_audio") is False:
        route["reasons"].insert(0, "WARNING: the video has no audio track (silent audio was "
                                   "substituted) — all audio results describe silence")
        route["has_music"] = False
        route["modules"].update({"beat_analysis": False, "cut_on_beat_test": False,
                                 "music_mood": False, "key_detection": False})
    route["sampled_windows"] = per_window
    route["audio_file"] = disp(audio)

    out = PROCESSED / ROUTE_FILE
    out.write_text(json.dumps(route, indent=2) + "\n", encoding="utf-8")
    from _lineage import record
    record(PROCESSED, "route", [audio], [out],
           {"content_type": args.content_type, "no_clap": args.no_clap,
            "n_windows": N_WINDOWS, "music_share_min": MUSIC_SHARE_MIN}, __file__)

    print(f"[route] audio class: {route['audio_class']} "
          f"(shares {shares}) -> music analyses {'ON' if route['has_music'] else 'OFF'} "
          f"[{route['has_music_basis']}]")
    for r in route["reasons"]:
        print(f"[route]   {r}")
    print(f"[ok] wrote {disp(out)}")
    return 0


def load_route(processed: Path) -> dict | None:
    p = Path(processed) / ROUTE_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    sys.exit(main())
