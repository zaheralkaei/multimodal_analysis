"""
Separate — isolate the vocals before transcription (round 11, audio stack).

Reads:  data/<video_id>/audio_music.wav (48 kHz mix from phase 0)
        data/<video_id>/content_route.json (is there music?)
Writes: data/<video_id>/vocals_stem.wav    — vocal stem at the stem's own rate
        data/<video_id>/vocals.wav         — same, 16 kHz mono, for speech recognition
        data/<video_id>/vocal_activity.csv — per-second vocal loudness (dBFS) + active flag
        data/<video_id>/separation_stats.json

Whisper on the full mix hears drums and synths as words; on the isolated
vocal stem it transcribes sung lyrics far better (and the per-second vocal
loudness tells later phases where nobody is singing).

Providers (--provider):
  auto         fal if FAL_KEY is set AND the route found music, else off
  fal          fal.ai fal-ai/demucs (htdemucs_ft, vocals only), queue API;
               ~$0.0007 per second of audio (≈ $0.15 for a 3.5-min song)
  local        demucs on this machine, if the `demucs` package is installed
               (slow on CPU; no cost)
  passthrough  OFFLINE TEST ONLY: "vocals" = the unchanged mix, labelled so —
               exercises the wiring without any service
  off          no separation; phase 4 transcribes the mix

Results are cached by audio content + settings (data/cache/fal_demucs/).
"""
from __future__ import annotations
import argparse, csv, json, os, shutil, subprocess, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from _paths import disp
import _cloud
PROCESSED = REPO_ROOT / "data" / "processed"
if "PROCESSED_DIR" in os.environ:
    PROCESSED = Path(os.environ["PROCESSED_DIR"])

FAL_ENDPOINT = "fal-ai/demucs"
FAL_MODEL = "htdemucs_ft"
MAX_UPLOAD_MB = 25          # data-URI upload; longer audio should not need separation
ACTIVE_DBFS = -40.0         # vocal stem louder than this in a second -> someone sings/speaks


# ------------------------------------------------------------------ providers

def separate_fal(mix: Path, out_stem: Path, processed: Path) -> dict:
    key = _cloud.get_key("FAL_KEY")
    if not key:
        raise _cloud.CloudError("FAL_KEY is not set (.env)")
    ck = _cloud.cache_key("fal_demucs", _cloud.sha_file(mix), FAL_MODEL, ["vocals"], "wav")
    cached = _cloud.cache_path("fal_demucs", ck, ".wav")
    dur = _duration(mix)
    est = dur * _cloud.PRICES["fal_demucs_per_sec"]
    if cached.exists():
        shutil.copy2(cached, out_stem)
        _cloud.log_usage(processed, "fal", "demucs", est, cached=True, detail={"sec": round(dur, 1)})
        return {"cached": True, "est_usd": 0.0}
    ok, why = _cloud.budget_ok(processed, est)
    if not ok:
        raise _cloud.CloudError(why)
    mp3 = processed / "_upload_mix.mp3"
    _cloud.encode_audio(mix, mp3, "mp3")
    size_mb = mp3.stat().st_size / 1e6
    if size_mb > MAX_UPLOAD_MB:
        mp3.unlink(missing_ok=True)
        raise _cloud.CloudError(f"mix is {size_mb:.0f} MB as MP3 (> {MAX_UPLOAD_MB} MB data-URI "
                                f"limit) — use --provider off or local for long videos")
    headers = {"Authorization": f"Key {key}", "Content-Type": "application/json"}
    body = {"audio_url": _cloud.data_uri(mp3, "audio/mpeg"), "model": FAL_MODEL,
            "stems": ["vocals"], "output_format": "wav"}
    mp3.unlink(missing_ok=True)
    sub = _cloud.request("POST", f"https://queue.fal.run/{FAL_ENDPOINT}", headers=headers,
                         json_body=body, timeout=300)
    rid = sub.get("request_id")
    status_url = sub.get("status_url") or f"https://queue.fal.run/{FAL_ENDPOINT}/requests/{rid}/status"
    response_url = sub.get("response_url") or f"https://queue.fal.run/{FAL_ENDPOINT}/requests/{rid}"
    deadline = time.time() + 1800
    while True:
        st = _cloud.request("GET", status_url, headers=headers, timeout=60)
        status = st.get("status", "")
        if status == "COMPLETED":
            break
        if status in ("FAILED", "ERROR", "CANCELLED") or time.time() > deadline:
            raise _cloud.CloudError(f"fal demucs request {rid}: status {status or 'timeout'}")
        time.sleep(3)
    res = _cloud.request("GET", response_url, headers=headers, timeout=120)
    vocals = (res.get("vocals") or {}).get("url") if isinstance(res.get("vocals"), dict) else res.get("vocals")
    if not vocals:
        raise _cloud.CloudError(f"fal demucs response has no vocals stem: keys {sorted(res)}")
    _cloud.download(vocals, out_stem)
    cached.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_stem, cached)
    _cloud.log_usage(processed, "fal", "demucs", est, cached=False,
                     detail={"sec": round(dur, 1), "model": FAL_MODEL, "request_id": rid})
    return {"cached": False, "est_usd": round(est, 4), "request_id": rid}


def separate_local(mix: Path, out_stem: Path, processed: Path) -> dict:
    try:
        import demucs  # noqa: F401
    except ImportError:
        raise _cloud.CloudError("local separation needs `pip install demucs`") from None
    tmp = processed / "_demucs"
    subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs_ft",
                    "-d", "cpu", "-o", str(tmp), str(mix)], check=True)
    found = list(tmp.rglob("vocals.wav"))
    if not found:
        raise _cloud.CloudError("demucs produced no vocals.wav")
    shutil.copy2(found[0], out_stem)
    shutil.rmtree(tmp, ignore_errors=True)
    return {"cached": False, "est_usd": 0.0}


def separate_passthrough(mix: Path, out_stem: Path, processed: Path) -> dict:
    shutil.copy2(mix, out_stem)
    return {"cached": False, "est_usd": 0.0, "note": "PASSTHROUGH test: vocals = full mix"}


PROVIDERS = {"fal": separate_fal, "local": separate_local, "passthrough": separate_passthrough}


# ------------------------------------------------------------------ helpers

def _duration(path: Path) -> float:
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def resolve_provider(requested: str, route: dict | None) -> tuple[str, str]:
    if requested != "auto":
        return requested, "requested"
    if route is not None and route.get("has_music") is False:
        return "off", "route found no music"
    if _cloud.get_key("FAL_KEY"):
        return "fal", "FAL_KEY set"
    return "off", "no FAL_KEY — transcribing the mix"


def align_and_resample(stem: Path, mix: Path, out16: Path) -> dict:
    """Make the stem exactly as long as the mix and write a 16 kHz mono copy."""
    import numpy as np
    import soundfile as sf
    import librosa
    y, sr = sf.read(str(stem), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    target = int(round(_duration(mix) * sr))
    diff_sec = (len(y) - target) / sr
    y = y[:target] if len(y) >= target else np.pad(y, (0, target - len(y)))
    sf.write(str(stem), y, sr, subtype="PCM_16")
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
    sf.write(str(out16), y16, 16000, subtype="PCM_16")
    return {"stem_sample_rate": sr, "length_diff_sec": round(diff_sec, 3)}


def vocal_activity(stem16: Path, out_csv: Path) -> dict:
    import numpy as np
    import soundfile as sf
    from _clap_tags import rms_dbfs
    y, sr = sf.read(str(stem16), dtype="float32")
    rows = []
    for s in range(int(len(y) // sr)):
        db = rms_dbfs(y[s * sr:(s + 1) * sr])
        rows.append({"second": s, "start_sec": s, "end_sec": s + 1,
                     "rms_dbfs": round(float(db), 2), "active": bool(db > ACTIVE_DBFS)})
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["second", "start_sec", "end_sec", "rms_dbfs", "active"])
        w.writeheader()
        w.writerows(rows)
    act = int(sum(1 for r in rows if r["active"]))
    return {"seconds": len(rows), "active_seconds": act,
            "active_share": round(act / len(rows), 3) if rows else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="auto", choices=["auto", "fal", "local", "passthrough", "off"])
    args = ap.parse_args()

    mix = PROCESSED / "audio_music.wav"
    if not mix.exists():
        print(f"[error] {disp(mix)} missing — run phase 0 first")
        return 1
    from route_content import load_route
    provider, why = resolve_provider(args.provider, load_route(PROCESSED))
    print(f"[separate] provider: {provider} ({why})")
    stats_path = PROCESSED / "separation_stats.json"
    stem, v16 = PROCESSED / "vocals_stem.wav", PROCESSED / "vocals.wav"
    if provider == "off":
        for p in (stem, v16, PROCESSED / "vocal_activity.csv"):
            p.unlink(missing_ok=True)          # never leave a stale stem behind
        stats = {"provider": "off", "reason": why}
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        from _lineage import record
        record(PROCESSED, "separate", [mix], [stats_path], {"provider": "off"}, __file__)
        print("[separate] off — phase 4 will transcribe the mix")
        return 0
    try:
        info = PROVIDERS[provider](mix, stem, PROCESSED)
    except (_cloud.CloudError, subprocess.CalledProcessError) as e:
        print(f"[error] separation failed: {e}")
        return 1
    info.update(align_and_resample(stem, mix, v16))
    info.update(vocal_activity(v16, PROCESSED / "vocal_activity.csv"))
    stats = {"provider": provider, "reason": why, "model": FAL_MODEL if provider == "fal" else
             ("htdemucs_ft" if provider == "local" else None),
             "is_passthrough": provider == "passthrough", "active_dbfs": ACTIVE_DBFS, **info}
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    from _lineage import record
    record(PROCESSED, "separate", [mix], [stem, v16, PROCESSED / "vocal_activity.csv", stats_path],
           {"provider": provider, "model": stats["model"]}, __file__)
    if abs(info.get("length_diff_sec", 0)) > 0.1:
        print(f"[warn] stem length differed from the mix by {info['length_diff_sec']} s — trimmed/padded")
    print(f"[ok] vocals: {disp(v16)} (active {info.get('active_share')} of seconds"
          f"{', cost ~$' + str(info['est_usd']) if info.get('est_usd') else ''}"
          f"{', cached' if info.get('cached') else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
