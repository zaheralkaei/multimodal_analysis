"""
Structure — beats, downbeats and song sections (round 11, audio stack).

Runs after phase 6. Reads data/<video_id>/audio_music.wav, audio_clap.csv
(where is music?) and content_route.json; writes music_structure.json:

    {"bpm", "beats", "downbeats", "beats_source", "downbeats_source",
     "segments": [{"start", "end", "label"}], "sections_source",
     "sections_labelled", "music_regions", "n_beats_masked"}

Beats and downbeats (--beats):
  beat_this   Beat This! (ISMIR 2024, MIT licence) on this machine — light, CPU,
              beats AND downbeats; default when installed
  replicate   from the All-In-One analyzer below
  librosa     fallback: beats only (no downbeats)

Sections (--sections):
  replicate   All-In-One Music Structure Analyzer (Kim & Nam) hosted on
              Replicate (sakemin/all-in-one-music-structure-analyzer):
              labelled sections (intro/verse/chorus/bridge/outro), ~$0.11 per
              run; needs REPLICATE_API_TOKEN
  local       unlabelled boundaries from a self-similarity novelty curve
              (Foote checkerboard on beat-synchronous MFCC + chroma), labels
              A/B/C by similarity, snapped to downbeats — free, approximate
  none

auto = beat_this + (replicate if a token is set, else local).
Beats outside music (CLAP windows typed speech/silence) are removed: a beat
tracker "finds" beats in speech too.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from _paths import disp
import _cloud
PROCESSED = REPO_ROOT / "data" / "processed"
if "PROCESSED_DIR" in os.environ:
    PROCESSED = Path(os.environ["PROCESSED_DIR"])

REPLICATE_MODEL = "sakemin/all-in-one-music-structure-analyzer"
REPLICATE_API = os.environ.get("REPLICATE_API_BASE", "https://api.replicate.com/v1").rstrip("/")
NOVELTY_KERNEL_BEATS = 16
MIN_SECTION_BEATS = 16


# ------------------------------------------------------------------ beats

def beats_beat_this(audio: Path) -> tuple[list, list]:
    import soundfile as sf
    from beat_this.inference import Audio2Beats
    y, sr = sf.read(str(audio), dtype="float32", always_2d=True)
    a2b = Audio2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    beats, downbeats = a2b(y.mean(axis=1), sr)
    return [round(float(b), 3) for b in beats], [round(float(d), 3) for d in downbeats]


def beats_librosa(audio: Path) -> tuple[list, list]:
    import librosa
    import numpy as np
    y, sr = librosa.load(str(audio), sr=22050, mono=True)
    _, frames = librosa.beat.beat_track(y=y, sr=sr)
    return [round(float(t), 3) for t in librosa.frames_to_time(frames, sr=sr)], []


def music_regions(clap_rows: list) -> list:
    """Merged [start, end] spans of CLAP windows typed 'music'."""
    from _clap_tags import window_audio_type
    spans = []
    for r in clap_rows:
        if window_audio_type(r) != "music":
            continue
        s, e = float(r["start_sec"]), float(r["end_sec"])
        if spans and s <= spans[-1][1] + 1e-6:
            spans[-1][1] = max(spans[-1][1], e)
        else:
            spans.append([s, e])
    return spans


def mask_to_regions(times: list, regions: list | None) -> list:
    if regions is None:
        return times
    return [t for t in times if any(a <= t < b for a, b in regions)]


def bpm_from_beats(beats: list) -> float | None:
    import numpy as np
    if len(beats) < 4:
        return None
    ibi = np.diff(beats)
    ibi = ibi[(ibi > 0.2) & (ibi < 2.0)]
    return round(60.0 / float(np.median(ibi)), 2) if len(ibi) else None


# ------------------------------------------------------------------ sections: local

def sections_local(audio: Path, beats: list, downbeats: list, regions: list | None) -> list:
    """Foote novelty on beat-synchronous MFCC+chroma; boundaries snapped to
    downbeats; labels by cosine similarity of segment means."""
    import librosa
    import numpy as np
    if len(beats) < 2 * MIN_SECTION_BEATS:
        return [{"start": beats[0], "end": beats[-1], "label": "A"}] if len(beats) >= 2 else []
    y, sr = librosa.load(str(audio), sr=22050, mono=True)
    hop = 512
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    ch = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    bf = librosa.time_to_frames(beats, sr=sr, hop_length=hop)
    feat = np.vstack([librosa.util.normalize(librosa.util.sync(mf, bf, aggregate=np.median), axis=1),
                      librosa.util.sync(ch, bf, aggregate=np.median)])
    # sync() returns one column per interval incl. the one BEFORE the first
    # beat; drop it so column k = the interval starting at beat k
    feat = feat[:, 1:]
    feat = feat / (np.linalg.norm(feat, axis=0, keepdims=True) + 1e-9)
    S = feat.T @ feat
    L = NOVELTY_KERNEL_BEATS
    g = np.exp(-0.5 * (np.linspace(-2, 2, 2 * L) ** 2))
    K = np.outer(g, g) * np.kron(np.array([[1, -1], [-1, 1]]), np.ones((L, L)))
    n = S.shape[0]
    Sp = np.pad(S, L, mode="edge")
    nov = np.array([np.sum(Sp[i:i + 2 * L, i:i + 2 * L] * K) for i in range(n)])
    nov = (nov - nov.min()) / (np.ptp(nov) + 1e-9)
    thr = nov.mean() + 0.5 * nov.std()
    peaks = []
    for i in np.argsort(-nov):
        if nov[i] < thr:
            break
        # the novelty curve is unreliable within one kernel of either end
        if i < MIN_SECTION_BEATS or i > n - MIN_SECTION_BEATS:
            continue
        if all(abs(i - p) >= MIN_SECTION_BEATS for p in peaks):
            peaks.append(int(i))
    times = sorted(beats[min(p, len(beats) - 1)] for p in peaks)
    if downbeats:
        times = sorted({min(downbeats, key=lambda d: abs(d - t)) for t in times})
    bounds = [beats[0]] + [t for t in times if beats[0] < t < beats[-1]] + [beats[-1]]
    segs = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a > 1.0:
            segs.append({"start": round(a, 3), "end": round(b, 3)})
    # labels: same letter when the mean features are very similar
    beat_arr = np.array(beats)
    means, letters = [], []
    for sg in segs:
        idx = np.where((beat_arr >= sg["start"]) & (beat_arr < sg["end"]))[0]
        idx = idx[idx < feat.shape[1]]
        m = feat[:, idx].mean(axis=1) if len(idx) else np.zeros(feat.shape[0])
        m = m / (np.linalg.norm(m) + 1e-9)
        match = next((letters[k] for k, mm in enumerate(means) if float(m @ mm) > 0.97), None)
        letters.append(match or chr(ord("A") + len(set(letters))))
        means.append(m)
        sg["label"] = letters[-1]
    return segs


# ------------------------------------------------------------------ sections: replicate

def _replicate_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def replicate_allin1(audio: Path, processed: Path) -> dict:
    """Run All-In-One on Replicate. Returns the analyzer's JSON (bpm, beats,
    downbeats, segments) plus the version id used."""
    token = _cloud.get_key("REPLICATE_API_TOKEN")
    if not token:
        raise _cloud.CloudError("REPLICATE_API_TOKEN is not set (.env)")
    model = _cloud.request("GET", f"{REPLICATE_API}/models/{REPLICATE_MODEL}",
                           headers=_replicate_headers(token), timeout=60)
    version = (model.get("latest_version") or {}).get("id")
    if not version:
        raise _cloud.CloudError(f"cannot read the latest version of {REPLICATE_MODEL}")
    ck = _cloud.cache_key("replicate_allin1", REPLICATE_API, _cloud.sha_file(audio), version)
    cached = _cloud.cache_get_json("replicate_allin1", ck)
    est = _cloud.PRICES["replicate_allin1_per_run"]
    if cached is not None:
        _cloud.log_usage(processed, "replicate", "allin1", est, cached=True, detail={"version": version})
        return cached
    ok, why = _cloud.budget_ok(processed, est)
    if not ok:
        raise _cloud.CloudError(why)
    schema = ((model.get("latest_version") or {}).get("openapi_schema") or {})
    props = (((schema.get("components") or {}).get("schemas") or {}).get("Input") or {}).get("properties") or {}
    file_field = next((k for k, v in props.items() if v.get("format") == "uri"), None) or "music_input"

    mp3 = processed / "_upload_mix.mp3"
    _cloud.encode_audio(audio, mp3, "mp3")
    up = _cloud.request("POST", f"{REPLICATE_API}/files", headers=_replicate_headers(token),
                        files={"content": ("mix.mp3", mp3.read_bytes(), "audio/mpeg")}, timeout=300,
                        idempotent=True)   # uploads are free; a duplicate file is harmless
    mp3.unlink(missing_ok=True)
    file_url = (up.get("urls") or {}).get("get")
    if not file_url:
        raise _cloud.CloudError(f"replicate file upload returned no url: keys {sorted(up)}")
    pred = _cloud.request("POST", f"{REPLICATE_API}/predictions",
                          headers={**_replicate_headers(token), "Content-Type": "application/json"},
                          json_body={"version": version, "input": {file_field: file_url}}, timeout=120)
    get_url = (pred.get("urls") or {}).get("get") or f"{REPLICATE_API}/predictions/{pred.get('id')}"
    deadline = time.time() + 1800
    while pred.get("status") not in ("succeeded", "failed", "canceled"):
        if time.time() > deadline:
            raise _cloud.CloudError("replicate prediction timed out")
        time.sleep(3)
        pred = _cloud.request("GET", get_url, headers=_replicate_headers(token), timeout=60)
    if pred.get("status") != "succeeded":
        raise _cloud.CloudError(f"replicate prediction {pred.get('status')}: {str(pred.get('error'))[:200]}")
    result = parse_allin1_output(pred.get("output"), token)
    result["replicate_version"] = version
    _cloud.cache_put_json("replicate_allin1", ck, result)
    _cloud.log_usage(processed, "replicate", "allin1", est, cached=False,
                     detail={"version": version, "prediction": pred.get("id")})
    return result


def parse_allin1_output(output, token: str | None = None) -> dict:
    """The Cog wrapper may return the analysis dict, a JSON file URL, or a
    list of file URLs (JSON + plots). Accept all three."""
    def has_fields(d):
        return isinstance(d, dict) and ("segments" in d or "beats" in d)
    if has_fields(output):
        return output
    if isinstance(output, dict):
        for v in output.values():
            if has_fields(v):
                return v
        output = [v for v in output.values() if isinstance(v, str)]
    urls = [output] if isinstance(output, str) else [u for u in (output or []) if isinstance(u, str)]
    # JSON-looking files first, then any other file (the wrapper's naming is
    # undocumented); images/audio are skipped by extension
    skip = (".png", ".jpg", ".jpeg", ".svg", ".pdf", ".wav", ".mp3", ".flac", ".npy")
    ordered = ([u for u in urls if u.split("?")[0].endswith(".json")]
               + [u for u in urls if not u.split("?")[0].endswith(".json")
                  and not u.split("?")[0].lower().endswith(skip)])
    for u in ordered:
        try:
            data = _cloud.request("GET", u, headers=_replicate_headers(token) if token else None)
        except _cloud.CloudError:
            continue
        if has_fields(data):
            return data
    raise _cloud.CloudError(f"unrecognised All-In-One output: {str(output)[:200]}")


# ------------------------------------------------------------------ main

def resolve(beats_opt: str, sections_opt: str) -> tuple[str, str]:
    if beats_opt == "auto":
        try:
            import beat_this  # noqa: F401
            beats_opt = "beat_this"
        except ImportError:
            beats_opt = "librosa"
    if sections_opt == "auto":
        sections_opt = "replicate" if _cloud.get_key("REPLICATE_API_TOKEN") else "local"
    return beats_opt, sections_opt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beats", default="auto", choices=["auto", "beat_this", "replicate", "librosa"])
    ap.add_argument("--sections", default="auto", choices=["auto", "replicate", "local", "none"])
    args = ap.parse_args()

    audio = PROCESSED / "audio_music.wav"
    if not audio.exists():
        print(f"[error] {disp(audio)} missing — run phase 0 first")
        return 1
    from route_content import load_route
    route = load_route(PROCESSED)
    out = PROCESSED / "music_structure.json"
    if route is not None and route.get("has_music") is False:
        result = {"skipped": True, "reason": "route found no music", "beats": [], "downbeats": [],
                  "segments": []}
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        from _lineage import record
        record(PROCESSED, "structure", [audio, PROCESSED / "content_route.json"], [out],
               {"skipped": True}, __file__)
        print("[structure] skipped — no music")
        return 0

    beats_opt, sections_opt = resolve(args.beats, args.sections)
    print(f"[structure] beats: {beats_opt}, sections: {sections_opt}")
    clap_path = PROCESSED / "audio_clap.csv"
    regions = None
    if clap_path.exists():
        with clap_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows and "music" in rows[0]:
            regions = music_regions(rows)

    allin1 = None
    fallback_reason = None
    try:
        if "replicate" in (beats_opt, sections_opt):
            allin1 = replicate_allin1(audio, PROCESSED)
    except _cloud.CloudError as e:
        fallback_reason = f"Replicate All-In-One failed: {str(e)[:200]}"
        print(f"[warn] {fallback_reason} — falling back to local")
        beats_opt = "beat_this" if beats_opt == "replicate" else beats_opt
        sections_opt = "local" if sections_opt == "replicate" else sections_opt
        if beats_opt == "beat_this":
            beats_opt, _ = resolve("auto", "none")

    if beats_opt == "replicate" and allin1:
        beats, downbeats = list(allin1.get("beats", [])), list(allin1.get("downbeats", []))
    elif beats_opt == "beat_this":
        beats, downbeats = beats_beat_this(audio)
    else:
        beats, downbeats = beats_librosa(audio)
    n_before = len(beats)
    beats, downbeats = mask_to_regions(beats, regions), mask_to_regions(downbeats, regions)

    if sections_opt == "replicate" and allin1:
        segments = [{"start": round(float(s["start"]), 3), "end": round(float(s["end"]), 3),
                     "label": str(s.get("label", "?"))} for s in allin1.get("segments", [])]
        labelled = True
    elif sections_opt in ("local", "replicate"):
        segments, labelled, sections_opt = sections_local(audio, beats, downbeats, regions), False, "local"
    else:
        segments, labelled = [], False

    result = {
        "bpm": bpm_from_beats(beats) if not (allin1 and beats_opt == "replicate") else allin1.get("bpm"),
        "beats": beats, "downbeats": downbeats,
        "beats_source": beats_opt, "downbeats_source": beats_opt if downbeats else None,
        "segments": segments, "sections_source": sections_opt, "sections_labelled": labelled,
        "music_regions": regions, "n_beats_masked": n_before - len(beats),
        "replicate_version": (allin1 or {}).get("replicate_version"),
        "fallback_reason": fallback_reason,
    }
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    from _lineage import record
    inputs = [audio] + [p for p in (clap_path, PROCESSED / "content_route.json") if p.exists()]
    record(PROCESSED, "structure", inputs, [out],
           {"beats": beats_opt, "sections": sections_opt,
            "replicate_version": result["replicate_version"]}, __file__)
    print(f"[ok] {len(beats)} beats ({result['n_beats_masked']} outside music removed), "
          f"{len(downbeats)} downbeats, {len(segments)} sections "
          f"({'labelled' if labelled else 'unlabelled'}), bpm {result['bpm']} -> {disp(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
