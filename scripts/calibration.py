"""
Calibration video — does the ASSEMBLED pipeline recover a known answer? (round 10)

Unit tests check modules one by one; this checks them together. It builds a
36-second synthetic video whose every property is known, runs the whole
pipeline on it, and scores the outputs against the ground truth.

    python scripts/calibration.py run          # make + run pipeline (offline) + score
    python scripts/calibration.py run --fake-cloud  # every hosted-service code path against
                                                    # the local fake server (free)
    python scripts/calibration.py run --cloud  # same through the REAL services: OpenRouter
                                               # vision + fal + Groq + Replicate (needs all
                                               # four keys; ~$0.17 in total)
    python scripts/calibration.py make         # only build data/raw/calib_v1.mp4 + truth
    python scripts/calibration.py score        # only score data/calib_v1/

What the video contains (seeded, so it is identical on every machine):

  time      audio                             picture
  0-20 s    synthetic music, 120 BPM          11 shots; 8 cuts exactly ON beats,
            (kick, snare, hi-hat, bass,       2 cuts deliberately OFF beat (+0.25 s);
            chords, arpeggio)                 designed camera moves: pan right/left,
                                              tilt down/up, zoom in/out, static, and
                                              a static shot with a moving subject
  20-30 s   synthetic speech (Windows TTS     one long static shot
            or espeak, if available)
  30-36 s   silence                           one static shot

Checks (all must pass; speech checks are skipped without a TTS engine):
  shots      cut precision/recall within 2 frames
  camera     designed motion recovered (pan/tilt/zoom direction, static)
  tempo      120 BPM within 3 %          beats   F-measure within 70 ms
  downbeats  bar starts (every 2 s) F-measure within 70 ms
  cut_downbeat  cut-on-downbeat test significant (7 of 10 cuts on bar starts)
  sections   a section boundary within 1 s of 10.0 s (music changes there)
  separation stem aligned with the mix and used for transcription;
             --cloud: the stem is >= 15 dB quieter than the mix in the
             instrumental part (vocals really isolated)
  cut_beat   cut-on-beat test significant (8 of 10 testable cuts on beats)
  audio_type CLAP music/speech/silence per window matches the timeline
  route      early route says music present
  transcript speech words recovered; no unflagged words in music or silence
  qa         no false heuristic alarms; no lineage problems
  coverage   camera labels for >= 85 % of shots;  sync  one row per shot
"""
from __future__ import annotations
import argparse, json, math, os, shutil, subprocess, sys, tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from _paths import disp
from _tools import ffmpeg_exe

CALIB_ID = "calib_v3"
W, H, FPS, SR = 640, 360, 24, 48000
DURATION = 36.0
BPM = 120.0
MUSIC_END, SPEECH_START, SPEECH_END = 20.0, 20.25, 30.0
SECTION_B_START = 10.0          # music changes section here (new chords, lead, busier drums)
BEATS_PER_BAR = 4
SPEECH_TEXT = ("Welcome to the calibration test. The quick brown fox jumps over the lazy dog. "
               "Every number in this report must be checked.")

# start, end, motion, expected labels (any of)
SHOTS = [
    (0.0, 2.0, "static", {"static"}),
    (2.0, 4.0, "pan_right", {"pan-right"}),
    (4.0, 6.0, "tilt_down", {"tilt-down"}),
    (6.0, 8.0, "zoom_in", {"zoom-in"}),
    (8.0, 10.0, "static_subject", {"static"}),
    (10.0, 12.0, "pan_left", {"pan-left"}),
    (12.0, 14.0, "tilt_up", {"tilt-up"}),
    (14.0, 15.5, "static", {"static"}),
    (15.5, 17.25, "static", {"static"}),
    (17.25, 18.75, "zoom_out", {"zoom-out"}),
    (18.75, 20.25, "static", {"static"}),
    (20.25, 24.6, "static", {"static"}),
    # 24.6 s lies OFF the 0.5 s frame grid: a frame-timing error makes the
    # last 2 fps frame of the previous shot show this one (QA false alarm)
    (24.6, 30.0, "static", {"static"}),
    (30.0, 36.0, "static_dim", {"static", "unknown"}),
]
RATE = 0.12   # pan/tilt: frame sizes per second; zoom: log-scale per second


# ------------------------------------------------------------------ truth

def ground_truth(speech: bool) -> dict:
    beats = [round(k * 60.0 / BPM, 3) for k in range(int(MUSIC_END * BPM / 60.0))]
    downbeats = beats[::BEATS_PER_BAR]
    cuts = [s for s, *_ in SHOTS[1:]]
    period = 60.0 / BPM
    on_beat = [c for c in cuts if c < MUSIC_END and abs((c / period) - round(c / period)) < 1e-6]
    off_beat = [c for c in cuts if c < MUSIC_END - period and c not in on_beat]
    segs = [{"start": 0.0, "end": MUSIC_END, "type": "music"}]
    if speech:
        segs.append({"start": SPEECH_START, "end": SPEECH_END, "type": "speech"})
    segs.append({"start": SPEECH_END, "end": DURATION, "type": "silence"})
    return {
        "version": CALIB_ID, "duration": DURATION, "fps": FPS, "size": [W, H],
        "tempo_bpm": BPM, "beats": beats, "downbeats": downbeats, "cuts": cuts,
        "cuts_on_downbeat": [c for c in on_beat if any(abs(c - d) < 1e-6 for d in downbeats)],
        "sections": [{"start": 0.0, "end": SECTION_B_START, "label": "A"},
                     {"start": SECTION_B_START, "end": MUSIC_END, "label": "B"}],
        "cuts_on_beat": on_beat, "cuts_off_beat": off_beat,
        "shots": [{"start": s, "end": e, "motion": m, "expected": sorted(x)}
                  for s, e, m, x in SHOTS],
        "audio_segments": segs,
        "speech": {"start": SPEECH_START, "end": SPEECH_END, "text": SPEECH_TEXT} if speech else None,
    }


# ------------------------------------------------------------------ picture

def _canvas(seed: int, w: int, h: int, dim: bool = False):
    """Smooth coloured field + sharp shapes (corners for tracking)."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(seed)
    small = rng.random((max(2, h // 40), max(2, w // 40), 3)).astype(np.float32)
    field = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    hue = (seed * 37) % 180
    base = np.zeros((h, w, 3), np.uint8)
    base[..., 0] = hue
    base[..., 1] = 140
    base[..., 2] = 200
    tint = cv2.cvtColor(base, cv2.COLOR_HSV2BGR).astype(np.float32)
    img = np.clip(0.55 * tint + 0.45 * field * 255, 0, 255).astype(np.uint8)
    for _ in range(int(w * h / 2500)):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        col = tuple(int(c) for c in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            sz = int(rng.integers(6, 28))
            cv2.rectangle(img, (x, y), (x + sz, y + int(sz * rng.uniform(0.5, 1.5))), col, -1)
        else:
            cv2.circle(img, (x, y), int(rng.integers(4, 16)), col, -1)
    if dim:
        img = (img * 0.45).astype(np.uint8)
    return img


def _frame(kind: str, canvas, t: float, subject=None):
    """Render one frame of a shot at time t (seconds since shot start)."""
    import cv2
    ch, cw = canvas.shape[:2]
    if kind in ("pan_right", "pan_left"):
        dx = RATE * W * (t - 1.0)             # centred on the shot middle: stays on canvas
        x0 = (cw - W) / 2 + (dx if kind == "pan_right" else -dx)
        M = [[1, 0, -x0], [0, 1, -(ch - H) / 2]]
        return cv2.warpAffine(canvas, __import__("numpy").float32(M), (W, H), flags=cv2.INTER_LINEAR)
    if kind in ("tilt_down", "tilt_up"):
        dy = RATE * H * (t - 1.0)
        y0 = (ch - H) / 2 + (dy if kind == "tilt_down" else -dy)
        M = [[1, 0, -(cw - W) / 2], [0, 1, -y0]]
        return cv2.warpAffine(canvas, __import__("numpy").float32(M), (W, H), flags=cv2.INTER_LINEAR)
    if kind in ("zoom_in", "zoom_out"):
        # zoom in: scale grows from 1; zoom out: scale shrinks towards 1
        s = math.exp(RATE * t) if kind == "zoom_in" else math.exp(RATE * (2.0 - t))
        # scale about the canvas centre, then take the central W x H window
        cx, cy = cw / 2, ch / 2
        M = [[s, 0, W / 2 - s * cx], [0, s, H / 2 - s * cy]]
        return cv2.warpAffine(canvas, __import__("numpy").float32(M), (W, H), flags=cv2.INTER_LINEAR)
    img = canvas[(ch - H) // 2:(ch - H) // 2 + H, (cw - W) // 2:(cw - W) // 2 + W].copy()
    if subject is not None:
        sh, sw = subject.shape[:2]
        x = int(60 + 120 * t)                  # 5 px/frame at 24 fps
        img[140:140 + sh, x:x + sw] = subject
    return img


def make_video_frames(path_avi: Path) -> None:
    import cv2
    vw = cv2.VideoWriter(str(path_avi), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    if not vw.isOpened():
        raise RuntimeError("OpenCV cannot write MJPG video")
    for i, (s, e, kind, _) in enumerate(SHOTS):
        pad = 1.6 if kind.startswith("zoom") else 1.35
        canvas = _canvas(100 + i, int(W * pad), int(H * pad), dim=(kind == "static_dim"))
        subject = _canvas(900, 90, 90) if kind == "static_subject" else None
        n0, n1 = int(round(s * FPS)), int(round(e * FPS))
        for f in range(n0, n1):
            vw.write(_frame(kind, canvas, (f - n0) / FPS, subject))
    vw.release()


# ------------------------------------------------------------------ sound

def _tts(text: str, out_wav: Path) -> bool:
    """Synthesize speech with an engine already on this machine."""
    if os.name == "nt":
        ps = ("Add-Type -AssemblyName System.Speech; "
              "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              f"$s.SetOutputToWaveFile('{out_wav}'); $s.Speak('{text}'); $s.Dispose()")
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True,
                           capture_output=True, timeout=120)
            return out_wav.exists() and out_wav.stat().st_size > 1000
        except Exception:
            pass
    for exe in ("espeak-ng", "espeak"):
        if shutil.which(exe):
            try:
                subprocess.run([exe, "-w", str(out_wav), text], check=True,
                               capture_output=True, timeout=120)
                return out_wav.exists()
            except Exception:
                pass
    return False


def make_audio(path_wav: Path, tmp: Path) -> bool:
    """Returns True when the speech section could be synthesized."""
    import numpy as np
    import soundfile as sf
    rng = np.random.default_rng(7)
    n = int(DURATION * SR)
    y = np.zeros(n, np.float32)
    t_all = np.arange(n) / SR
    period = 60.0 / BPM

    def add(start, sig):
        i = int(start * SR)
        j = min(n, i + len(sig))
        y[i:j] += sig[: j - i]

    chords_a = [[220.0, 261.63, 329.63], [174.61, 220.0, 261.63],
                [261.63, 329.63, 392.0], [196.0, 246.94, 293.66]]   # Am F C G
    chords_b = [[293.66, 349.23, 440.0], [233.08, 293.66, 349.23],
                [349.23, 440.0, 523.25], [261.63, 329.63, 392.0]]   # Dm Bb F C
    n_beats = int(MUSIC_END / period)
    for k in range(n_beats):
        tb = k * period
        in_b = tb >= SECTION_B_START - 1e-9
        tt = np.arange(int(0.25 * SR)) / SR
        kick = np.sin(2 * np.pi * (50 + 70 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 18)
        add(tb, (1.0 if k % BEATS_PER_BAR == 0 else 0.75) * kick)   # accented downbeat
        if k % 2 == 1 or in_b:
            sn = rng.standard_normal(int(0.12 * SR)) * np.exp(-np.arange(int(0.12 * SR)) / SR * 30)
            add(tb, (0.45 if in_b else 0.35) * sn)
        for h in ([0.5] if not in_b else [0.25, 0.5, 0.75]):           # B: 16th hi-hats
            hat = rng.standard_normal(int(0.03 * SR)) * np.exp(-np.arange(int(0.03 * SR)) / SR * 120)
            add(tb + period * h, 0.12 * np.diff(hat, prepend=0))
        chord = (chords_b if in_b else chords_a)[(k // BEATS_PER_BAR) % 4]
        tt = np.arange(int(period * SR)) / SR
        bass = np.sin(2 * np.pi * chord[0] / 2 * tt) * np.exp(-tt * 3)
        add(tb, 0.35 * bass)
        steps = 4 if in_b else 2                                       # B: 16th-note lead
        for q in range(steps):
            f0 = chord[(k * steps + q) % 3] * 2
            ta = np.arange(int(period / steps * SR)) / SR
            harm = range(1, 7) if in_b else range(1, 3)                # B: brighter timbre
            note = sum(np.sin(2 * np.pi * f0 * h * ta) / h for h in harm) * np.exp(-ta * 6)
            add(tb + q * period / steps, (0.10 if in_b else 0.12) * note)
    music_mask = t_all < MUSIC_END
    bar = BEATS_PER_BAR * period
    for c_i in range(int(MUSIC_END / bar)):                            # sustained chord pad
        a, b = c_i * bar, (c_i + 1) * bar
        sl = (t_all >= a) & (t_all < b)
        pad = (chords_b if a >= SECTION_B_START - 1e-9 else chords_a)[c_i % 4]
        for f0 in pad:
            y[sl] += 0.05 * np.sin(2 * np.pi * f0 * t_all[sl])
    y[~music_mask] = 0.0

    speech_ok = _tts(SPEECH_TEXT, tmp / "speech.wav")
    if speech_ok:
        import librosa
        sp, ssr = sf.read(str(tmp / "speech.wav"), dtype="float32")
        if sp.ndim > 1:
            sp = sp.mean(axis=1)
        sp = librosa.resample(sp, orig_sr=ssr, target_sr=SR)
        sp = sp[: int((SPEECH_END - SPEECH_START - 0.3) * SR)]
        sp = 0.8 * sp / (np.abs(sp).max() + 1e-9)
        add(SPEECH_START + 0.15, sp)
    y += (rng.standard_normal(n) * 1e-4).astype(np.float32)
    y = 0.8 * y / (np.abs(y).max() + 1e-9)
    sf.write(str(path_wav), y, SR, subtype="PCM_16")
    return speech_ok


def make(out_mp4: Path) -> dict:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        make_video_frames(tmp / "v.avi")
        speech = make_audio(tmp / "a.wav", tmp)
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(tmp / "v.avi"),
                        "-i", str(tmp / "a.wav"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "16", "-r", str(FPS), "-c:a", "aac", "-b:a", "192k",
                        "-shortest", str(out_mp4)], check=True, timeout=600)
    gt = ground_truth(speech)
    gt["version"] = out_mp4.stem          # scoring finds the video by this id
    gt_path = out_mp4.with_suffix(".truth.json")
    gt_path.write_text(json.dumps(gt, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] calibration video {disp(out_mp4)} (speech: {'yes' if speech else 'no TTS engine'})")
    print(f"[ok] ground truth {disp(gt_path)}")
    return gt


# ------------------------------------------------------------------ scoring

def _load_csv(p: Path) -> list:
    import csv
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def match_events(pred: list, truth: list, tol: float) -> dict:
    """Greedy one-to-one matching within tol. Returns P/R/F1 + counts."""
    used, tp = set(), 0
    for t in truth:
        best = None
        for j, p in enumerate(pred):
            if j not in used and abs(p - t) <= tol and (best is None or abs(p - t) < abs(pred[best] - t)):
                best = j
        if best is not None:
            used.add(best)
            tp += 1
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(truth) if truth else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "n_pred": len(pred), "n_truth": len(truth),
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}


def isolation_db(processed: Path) -> dict:
    """Loudness of the vocal stem vs the mix in the purely instrumental part
    (0-20 s: no voice in the calibration music) and in the speech part."""
    import numpy as np
    import soundfile as sf
    from _clap_tags import rms_dbfs
    mix, sr = sf.read(str(processed / "audio.wav"), dtype="float32")
    voc, sr2 = sf.read(str(processed / "vocals.wav"), dtype="float32")
    seg = lambda y, r, a, b: y[int(a * r):int(b * r)]
    inst = rms_dbfs(seg(mix, sr, 0.5, MUSIC_END - 0.5)) - rms_dbfs(seg(voc, sr2, 0.5, MUSIC_END - 0.5))
    sp = rms_dbfs(seg(mix, sr, SPEECH_START + 0.5, SPEECH_END - 0.5)) - \
        rms_dbfs(seg(voc, sr2, SPEECH_START + 0.5, SPEECH_END - 0.5))
    return {"instrumental_drop_db": round(float(inst), 1), "speech_drop_db": round(float(sp), 1)}


def frame_timing_errors(processed: Path, gt: dict) -> list:
    """For frames inside the pan shots (every source frame is unique there),
    find the best-matching source frame: actual minus assumed time (s)."""
    import cv2
    import numpy as np
    meta = json.loads((processed / "metadata.json").read_text(encoding="utf-8"))
    fps = int(meta.get("frame_fps") or 2)
    video = REPO_ROOT / "data" / "raw" / f"{gt['version']}.mp4"
    cap = cv2.VideoCapture(str(video))
    src = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        src.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (80, 45)).astype(np.float32))
    cap.release()
    if not src:
        return []
    src = np.stack(src)
    errs = []
    for g in gt["shots"]:
        if not g["motion"].startswith("pan"):
            continue
        for n in range(int(math.ceil(g["start"] * fps)) + 1, int(math.ceil(g["end"] * fps)) + 1):
            img = cv2.imread(str(processed / "frames" / f"frame_{n:05d}.jpg"), 0)
            if img is None:
                continue
            t_assumed = (n - 1) / fps
            if not (g["start"] + 0.1 < t_assumed < g["end"] - 0.1):
                continue
            g80 = cv2.resize(img, (80, 45)).astype(np.float32)
            k = int(np.abs(src - g80).mean(axis=(1, 2)).argmin())
            errs.append(k / gt["fps"] - t_assumed)
    return errs


def _words(text: str) -> list:
    import re
    return re.findall(r"[a-z']+", text.lower())


def score(processed: Path, gt: dict) -> dict:
    checks = []

    def add(name, passed, value, target, note=""):
        checks.append({"check": name, "status": ("pass" if passed else "FAIL") if passed is not None else "skip",
                       "value": value, "target": target, "note": note})

    shots = json.loads((processed / "shots.json").read_text(encoding="utf-8"))
    pred_cuts = [float(s["start_sec"]) for s in shots[1:]]
    m = match_events(pred_cuts, gt["cuts"], tol=2.0 / gt["fps"] + 0.01)
    add("shots", m["f1"] >= 0.9, m, "F1 >= 0.9 (+-2 frames)")

    cam = _load_csv(processed / "shot_camera.csv")
    ok = tot = 0
    wrong = []
    for g in gt["shots"]:
        best = max(cam, key=lambda r: min(g["end"], float(r["end_sec"])) - max(g["start"], float(r["start_sec"])),
                   default=None)
        if best is None:
            continue
        tot += 1
        if best["camera_motion"] in g["expected"]:
            ok += 1
        else:
            wrong.append(f"{g['start']}-{g['end']} {g['motion']}: got {best['camera_motion']}")
    acc = ok / tot if tot else 0.0
    add("camera", acc >= 0.85, {"correct": ok, "total": tot, "wrong": wrong}, "accuracy >= 0.85")

    ms = json.loads((processed / "music_summary.json").read_text(encoding="utf-8"))
    struct_path = processed / "music_structure.json"
    struct = json.loads(struct_path.read_text(encoding="utf-8")) if struct_path.exists() else {}
    tempo = float(struct.get("bpm") or ms.get("tempo_bpm") or 0)
    add("tempo", abs(tempo - gt["tempo_bpm"]) / gt["tempo_bpm"] <= 0.03, round(tempo, 1),
        f"{gt['tempo_bpm']} BPM +- 3%")
    pred_beats = [b for b in (struct.get("beats") or ms.get("beat_times", [])) if b < MUSIC_END]
    mb = match_events(pred_beats, gt["beats"], tol=0.07)
    add("beats", mb["f1"] >= 0.8, mb, "F-measure >= 0.8 (+-70 ms)")
    if gt.get("downbeats"):
        md = match_events(struct.get("downbeats") or [], gt["downbeats"], tol=0.07)
        add("downbeats", md["f1"] >= 0.8, md, "F-measure >= 0.8 (+-70 ms)")
        inner = [float(x["start"]) for x in (struct.get("segments") or [])[1:]]
        target = gt["sections"][1]["start"]
        near = [b for b in inner if abs(b - target) <= 1.0]
        add("sections", bool(near) and len(inner) - len(near) <= 1,
            {"inner_boundaries": inner, "source": struct.get("sections_source")},
            f"boundary within 1 s of {target} s, at most 1 extra")

    st = json.loads((processed / "sync_stats.json").read_text(encoding="utf-8"))
    al = st.get("cut_beat_alignment") or {}
    add("cut_beat", bool(al.get("significant")),
        {k: al.get(k) for k in ("observed_rate", "chance_rate", "p_value")},
        "cut-on-beat significant (8/10 cuts designed on the beat)")
    if gt.get("downbeats"):
        ad = st.get("cut_downbeat_alignment") or {}
        add("cut_downbeat", bool(ad.get("significant")),
            {k: ad.get(k) for k in ("observed_rate", "chance_rate", "p_value")},
            "cut-on-downbeat significant (7/10 cuts designed on bar starts)")

    sep_path = processed / "separation_stats.json"
    sep = json.loads(sep_path.read_text(encoding="utf-8")) if sep_path.exists() else {}
    tst_path = processed / "transcript_stats.json"
    tst = json.loads(tst_path.read_text(encoding="utf-8")) if tst_path.exists() else {}
    if sep and sep.get("provider") != "off":
        import soundfile as sf
        v = sf.info(str(processed / "vocals.wav"))
        m = sf.info(str(processed / "audio.wav"))
        diff = abs(v.frames / v.samplerate - m.frames / m.samplerate)
        add("separation_wiring", diff <= 0.05 and tst.get("input") == "vocal stem",
            {"provider": sep.get("provider"), "length_diff_sec": round(diff, 3),
             "transcribed": tst.get("input")}, "stem aligned (+-50 ms) and used for transcription")
        if not sep.get("is_passthrough"):
            iso = isolation_db(processed)
            add("separation_isolation", iso["instrumental_drop_db"] >= 15, iso,
                "stem >= 15 dB below the mix in the instrumental part")
    else:
        add("separation_wiring", None, sep.get("provider"), "stem used for transcription",
            "separation off")

    from _clap_tags import window_audio_type
    clap = _load_csv(processed / "audio_clap.csv")
    ok = tot = 0
    seen_ok: set = set()
    for w in clap:
        s, e = float(w["start_sec"]), float(w["end_sec"])
        seg = next((g for g in gt["audio_segments"] if g["start"] <= s and e <= g["end"]), None)
        if seg is None or "music" not in w:
            continue
        tot += 1
        kind = window_audio_type(w)
        ok += kind == seg["type"]
        if kind == seg["type"]:
            seen_ok.add(kind)
    missed = sorted({g["type"] for g in gt["audio_segments"]
                     if any(g["start"] <= float(w["start_sec"]) and float(w["end_sec"]) <= g["end"]
                            for w in clap)} - seen_ok)
    add("audio_type", (ok / tot >= 0.8 and not missed) if tot else None,
        {"correct": ok, "total": tot, "types_never_recognised": missed},
        "type matches timeline >= 80% of windows, every type recognised")

    route = json.loads((processed / "content_route.json").read_text(encoding="utf-8")) \
        if (processed / "content_route.json").exists() else {}
    shares = route.get("audio_shares") or {}
    expect_speech = bool(gt.get("speech"))
    add("route", (route.get("has_music") is True
                  and (shares.get("speech", 0) > 0 or not expect_speech)
                  and shares.get("silence", 0) > 0) if route else None,
        {"audio_class": route.get("audio_class"), "shares": shares,
         "n_windows": len(route.get("sampled_windows", []))},
        "music present; speech and silence sections also heard")

    tr = _load_csv(processed / "transcript.csv")
    usable = [r for r in tr if str(r.get("suspect", "")).lower() != "true"]
    if gt.get("speech"):
        sp = gt["speech"]
        said = " ".join(r["text"] for r in usable
                        if float(r["start_sec"]) < sp["end"] and float(r["end_sec"]) > sp["start"])
        truth_w = _words(sp["text"])
        got = set(_words(said))
        rec = sum(1 for w in truth_w if w in got) / len(truth_w)
        add("transcript_recall", rec >= 0.6, round(rec, 3), "word recall >= 0.6")
    else:
        add("transcript_recall", None, None, "word recall >= 0.6", "no TTS engine")
    stray = [r["text"] for r in usable
             if (float(r["end_sec"]) <= SPEECH_START + 0.1 or float(r["start_sec"]) >= SPEECH_END - 0.1)
             and r.get("text", "").strip()]
    add("no_stray_words", len(stray) == 0, stray, "no unflagged words in music/silence")

    qa = json.loads((processed / "qa_report.json").read_text(encoding="utf-8"))
    heur = [i for i in qa["issues"] if i["check"] in ("possible_false_cut", "possible_missed_cut")]
    add("qa_false_alarms", len(heur) == 0, [i["message"] for i in heur], "0 heuristic cut alarms")
    crit = [i for i in qa["issues"] if i["severity"] == "critical" and i["check"] != "vision_mock"]
    add("qa_critical", len(crit) == 0, [i["message"] for i in crit], "no critical issues (mock vision allowed)")
    vcov = (st.get("coverage") or {}).get("vision") or {}
    if st.get("vision_provider") not in (None, "mock"):
        # real (or fake-cloud) vision: every shot must have a parsed answer
        add("vision_answers", (vcov.get("pct") or 0) >= 90, vcov, ">= 90% shots with a valid answer")
    else:
        add("vision_answers", None, vcov, ">= 90% shots with a valid answer", "mock vision")

    # frame timing: extracted 2 fps frame N must show the source at (N-1)/fps
    errs = frame_timing_errors(processed, gt)
    add("frame_timing", (max(abs(e) for e in errs) <= 1.0 / gt["fps"] + 1e-6) if errs else None,
        {"max_abs_error_sec": round(max(abs(e) for e in errs), 3) if errs else None,
         "n_frames": len(errs)}, "frame N shows (N-1)/fps within 1 source frame")

    cov = (st.get("coverage") or {}).get("camera") or {}
    add("coverage_camera", (cov.get("pct") or 0) >= 85, cov, ">= 85% shots labelled")
    sync = _load_csv(processed / "sync_per_shot.csv")
    starts = [float(r["start_sec"]) for r in sync]
    add("sync", len(sync) == len(shots) and starts == sorted(starts),
        {"rows": len(sync), "shots": len(shots)}, "one ordered row per shot")

    n_fail = sum(c["status"] == "FAIL" for c in checks)
    return {"video": gt["version"], "passed": n_fail == 0, "n_fail": n_fail,
            "n_skip": sum(c["status"] == "skip" for c in checks), "checks": checks}


def print_report(rep: dict) -> None:
    for c in rep["checks"]:
        v = c["value"]
        vs = json.dumps(v) if not isinstance(v, str) else v
        print(f"  [{c['status']:>4}] {c['check']:<18} {vs[:120]}   target: {c['target']}"
              + (f" ({c['note']})" if c["note"] else ""))
    print(f"\n{'CALIBRATION PASSED' if rep['passed'] else 'CALIBRATION FAILED'}: "
          f"{rep['n_fail']} failed, {rep['n_skip']} skipped of {len(rep['checks'])}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["make", "score", "run"])
    ap.add_argument("--id", default=CALIB_ID)
    ap.add_argument("--cloud", action="store_true",
                    help="use the hosted audio services (needs keys; ~$0.15)")
    ap.add_argument("--fake-cloud", action="store_true",
                    help="run every hosted-service code path against scripts/fake_cloud.py (free)")
    args = ap.parse_args()
    video = REPO_ROOT / "data" / "raw" / f"{args.id}.mp4"
    truth = video.with_suffix(".truth.json")
    processed = REPO_ROOT / "data" / args.id

    if args.action in ("make", "run"):
        make(video)
    if args.action == "run":
        cmd = [sys.executable, str(Path(__file__).parent / "run_pipeline.py"), str(video),
               "--id", args.id, "--provider", "mock"]
        if args.fake_cloud:
            import fake_cloud
            gt0 = json.loads(truth.read_text(encoding="utf-8"))
            _srv, base, _state, env = fake_cloud.start(
                port=int(os.environ.get("FAKE_CLOUD_PORT", "8765")),
                vocals_silence_before=MUSIC_END, speech=gt0.get("speech"))
            os.environ.update(env)
            print(f"[fake-cloud] serving fal/Groq/Replicate/OpenRouter at {base}")
            cmd[cmd.index("mock")] = "openrouter"
            cmd += ["--separation", "fal", "--asr-backend", "groq", "--sections", "replicate"]
        elif args.cloud:
            # the first real run checks ALL four services on known answers:
            # OpenRouter vision (14 shots, ~$0.01) + fal + Groq + Replicate
            cmd[cmd.index("mock")] = "openrouter"
            cmd += ["--separation", "fal", "--asr-backend", "groq", "--sections", "replicate"]
        else:
            # offline: exercise every step's wiring without any paid service
            cmd += ["--separation", "passthrough", "--asr-backend", "local", "--sections", "local"]
        print("[run] " + " ".join(cmd[1:]))
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
        if rc != 0:
            print(f"[error] pipeline failed (exit {rc})")
            if args.fake_cloud:
                for line in _state.log:
                    print(f"[fake-cloud] {line}")
            return 2
        if args.fake_cloud and _state.log:
            print("[fake-cloud] rejected requests:")
            for line in _state.log:
                print(f"   {line}")
    if args.action in ("score", "run"):
        gt = json.loads(truth.read_text(encoding="utf-8"))
        rep = score(processed, gt)
        (processed / "calibration_report.json").write_text(json.dumps(rep, indent=2) + "\n",
                                                          encoding="utf-8")
        print_report(rep)
        return 0 if rep["passed"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
