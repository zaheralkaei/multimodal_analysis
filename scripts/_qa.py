"""Cross-module QA report (round 10).

Each module can look fine on its own while the assembled analysis is wrong:
a stale caption file, a speech analysed as music, lyrics "heard" in silence,
two shots that are really one. build_report() runs consistency checks
ACROSS modules and returns

    {"summary": {"critical": n, "warning": n, "info": n},
     "issues":  [{"severity", "check", "message", "shot_idx", "time_sec"}],
     "review_first": [{"shot_idx", "time_sec", "n_issues", "checks"}]}

Critical = do not use the affected numbers. Warning = check before relying
on it. Info = context for interpretation. The frame-histogram checks are
heuristics: they point a human reviewer at shots, they do not decide.
"""
from __future__ import annotations
from pathlib import Path

from _clap_tags import window_audio_type

SEV_RANK = {"critical": 3, "warning": 2, "info": 1}

# Heuristic thresholds (documented in the issue messages)
# A frame pair must look alike/different in BOTH colour (histogram r) and
# layout (normalised cross-correlation of a 32x18 grey thumbnail). Colour
# alone flagged 11/12 false alarms on dark footage in the round-10 check,
# where almost every histogram is "black". Frames too dark or flat to judge
# are skipped.
FALSE_CUT_HIST_CORR = 0.97     # neighbouring mid-frames this similar -> maybe one shot
FALSE_CUT_NCC = 0.85
MISSED_CUT_HIST_CORR = 0.40    # a jump this large inside a shot -> maybe a missed cut
MISSED_CUT_NCC = 0.20
MIN_LUMA, MIN_CONTRAST = 40.0, 20.0   # mean / std of grey level (0-255)
MISSED_CUT_MIN_SHOT = 2.5      # only look inside shots at least this long (s)
SILENCE_LYRIC_SHARE = 0.5      # lyric segment mostly over "silence" windows


def _issue(sev, check, msg, shot_idx=None, time_sec=None):
    return {"severity": sev, "check": check, "message": msg,
            "shot_idx": shot_idx, "time_sec": None if time_sec is None else round(float(time_sec), 2)}


def _signature(path):
    """(colour histogram, 32x18 grey thumbnail z-scored, judgeable?)"""
    import cv2
    import numpy as np
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(h, h)
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (32, 18),
                   interpolation=cv2.INTER_AREA).astype(np.float64)
    judgeable = g.mean() >= MIN_LUMA and g.std() >= MIN_CONTRAST
    z = (g - g.mean()) / (g.std() + 1e-6)
    return h, z, judgeable


def _corr(h1, h2) -> float:
    import cv2
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


def _ncc(z1, z2) -> float:
    return float((z1 * z2).mean())


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


# ------------------------------------------------------------------ checks

def check_lineage(lineage: dict) -> list:
    out = []
    if not lineage.get("has_lineage"):
        out.append(_issue("info", "lineage", "No lineage.json: outputs predate round 10, so "
                          "stale files cannot be detected. Re-run the pipeline once."))
    for p in lineage.get("problems", []):
        what = ("an input changed after it ran" if p["kind"] == "stale"
                else "its output was replaced afterwards")
        out.append(_issue("critical", "lineage",
                          f"{p['step']}: {what} ({p['file']}) — its results were NOT joined; "
                          f"re-run {p['step']}"))
    return out


def check_vision(rows: list, vision_stats: dict) -> list:
    out = []
    if vision_stats.get("is_mock") or vision_stats.get("provider") == "mock":
        out.append(_issue("critical", "vision_mock",
                          "Vision answers are MOCK test data — not usable for analysis"))
        return out
    n = len(rows)
    bad = [r for r in rows if r.get("vision_caption", "").startswith("[")]
    if n and bad:
        sev = "critical" if len(bad) / n > 0.2 else "warning"
        out.append(_issue(sev, "vision_errors",
                          f"{len(bad)}/{n} shots have no vision answer (API or parse errors)"))
    return out


def check_coverage(coverage: dict) -> list:
    out = []
    limits = [("vision", 0.9, "warning"), ("camera", 0.7, "warning"),
              ("camera_high_confidence", 0.3, "info"), ("audio_mood", 0.5, "info")]
    for key, lim, sev in limits:
        c = coverage.get(key)
        if c and c["total"] and c["pct"] is not None and c["pct"] < lim * 100:
            out.append(_issue(sev, "coverage",
                              f"{key}: only {c['valid']}/{c['total']} shots valid ({c['pct']}%)"))
    return out


def check_route(route: dict | None) -> list:
    out = []
    if route is None:
        out.append(_issue("warning", "route", "No content_route.json — music presence was not "
                          "checked before analysis (run route_content.py)"))
        return out
    if route.get("has_music_basis") == "unverified":
        out.append(_issue("warning", "route", "Music presence unverified (CLAP skipped); "
                          "beat statistics may describe a speech"))
    for r in route.get("reasons", []):
        if r.startswith("WARNING"):
            out.append(_issue("warning", "route", r))
    return out


def check_music(music_summary: dict, route: dict | None) -> list:
    out = []
    if not music_summary:
        return out
    tempo = float(music_summary.get("tempo_bpm") or 0)
    if music_summary.get("beats_valid") is False:
        out.append(_issue("info", "music", "No music detected: tempo/beat/key ignored and "
                          "cut-on-beat not tested"))
        return out
    if tempo and (tempo < 70 or tempo > 180):
        out.append(_issue("warning", "tempo", f"Tempo {tempo:.0f} BPM is unusual — possible "
                          f"half/double-tempo error of the beat tracker"))
    kc = music_summary.get("key_correlation")
    if kc is not None and float(kc) < 0.6:
        out.append(_issue("info", "key", f"Key estimate is weak (correlation {float(kc):.2f})"))
    return out


def check_transcript(transcript: list, clap: list, transcript_stats: dict) -> list:
    out = []
    lp = transcript_stats.get("language_probability")
    has_words = any(str(s.get("suspect", "")).lower() != "true" and s.get("text", "").strip()
                    for s in transcript)
    if lp is not None and float(lp) < 0.5 and has_words:
        out.append(_issue("warning", "language", f"Language detection uncertain "
                          f"({transcript_stats.get('language')}, p={float(lp):.2f}) — "
                          f"consider forcing --whisper-language"))
    typed = [r for r in clap if "silence" in r and "music" in r and "speech" in r]
    for seg in transcript:
        if str(seg.get("suspect", "")).lower() == "true" or not seg.get("text", "").strip():
            continue
        s, e = float(seg["start_sec"]), float(seg["end_sec"])
        tot = sil = 0.0
        for w in typed:
            ov = _overlap(s, e, float(w["start_sec"]), float(w["end_sec"]))
            if ov <= 0:
                continue
            tot += ov
            kind = window_audio_type(w)
            sil += ov if kind == "silence" else 0
        if tot > 0 and sil / tot >= SILENCE_LYRIC_SHARE:
            out.append(_issue("warning", "lyrics_in_silence",
                              f"Words transcribed where CLAP hears silence: "
                              f"\"{seg['text'][:50]}\" — likely hallucination", None, s))
    return out


def check_frames(rows: list, repo_root: Path, frames_dir: Path, frame_fps: int) -> list:
    """Heuristic visual checks for false and missed cuts."""
    out = []
    try:
        import cv2  # noqa: F401
    except ImportError:
        return out
    sigs = []
    for r in rows:
        mf = r.get("mid_frame", "")
        sigs.append(_signature(repo_root / mf) if mf else None)
    for i in range(1, len(rows)):
        a, b = sigs[i - 1], sigs[i]
        if a is None or b is None or not (a[2] and b[2]):
            continue
        c, n = _corr(a[0], b[0]), _ncc(a[1], b[1])
        if c >= FALSE_CUT_HIST_CORR and n >= FALSE_CUT_NCC:
            out.append(_issue("warning", "possible_false_cut",
                              f"Shots {i - 1} and {i} look nearly identical (colour r={c:.2f}, "
                              f"layout r={n:.2f}) — maybe one shot", i, rows[i]["start_sec"]))
    if not frames_dir.is_dir() or not frame_fps:
        return out
    from phase3_camera import shot_frame_range
    for r in rows:
        s, e = float(r["start_sec"]), float(r["end_sec"])
        if e - s < MISSED_CUT_MIN_SHOT:
            continue
        # stay 0.25 s away from both cuts: a frame-timing error of even one
        # source frame would otherwise compare across the cut
        first, last = shot_frame_range(s + 0.25, e - 0.25, frame_fps)
        prev = None
        for fi in range(first, last + 1):
            sg = _signature(frames_dir / f"frame_{fi:05d}.jpg")
            if sg is None:
                continue
            if prev is not None and prev[2] and sg[2]:
                c, n = _corr(prev[0], sg[0]), _ncc(prev[1], sg[1])
                if c < MISSED_CUT_HIST_CORR and n < MISSED_CUT_NCC:
                    t = (fi - 1) / frame_fps
                    out.append(_issue("warning", "possible_missed_cut",
                                      f"Shot {r['shot_idx']} changes abruptly near {t:.1f}s "
                                      f"(colour r={c:.2f}, layout r={n:.2f}) — maybe a missed cut",
                                      int(r["shot_idx"]), t))
                    break
            prev = sg
    return out


def check_people_vs_motion(rows: list) -> list:
    out = []
    for r in rows:
        if r.get("vision_emotion") != "no people":
            continue
        try:
            share = float(r.get("camera_subject_motion_share") or 0)
        except ValueError:
            continue
        if share > 0.5 and r.get("camera_motion") not in ("", "unknown"):
            out.append(_issue("info", "people_vs_motion",
                              f"Vision says no people, but {share * 100:.0f}% of tracked points "
                              f"move independently — check the frame", int(r["shot_idx"]),
                              r["start_sec"]))
    return out


def check_tempo_agreement(music_summary: dict | None, structure: dict | None) -> list:
    """Two independent beat trackers (librosa in phase 6, Beat This!/All-In-One
    in the structure step) should agree up to a factor of 2 (half/double
    tempo is the usual, mostly harmless ambiguity). Any other ratio means at
    least one tracker is wrong, so every beat-based statistic is in doubt."""
    a = float((music_summary or {}).get("tempo_bpm") or 0)
    b = float((structure or {}).get("bpm") or 0)
    if not a or not b or (music_summary or {}).get("beats_valid") is False:
        return []
    ratio = max(a, b) / min(a, b)
    if min(abs(ratio - r) / r for r in (1.0, 2.0)) <= 0.04:
        return []
    return [_issue("warning", "tempo_disagreement",
                   f"Beat trackers disagree: {b:.1f} BPM ({(structure or {}).get('beats_source')}) vs "
                   f"{a:.1f} BPM (librosa), ratio {ratio:.2f} — beat/downbeat statistics depend on "
                   f"which is right; check by ear before using them")]


def check_audio_stack(audio_stack: dict | None, structure: dict | None, route: dict | None) -> list:
    out = []
    a = audio_stack or {}
    if a.get("separation_passthrough"):
        out.append(_issue("warning", "separation_passthrough",
                          "Vocal 'stem' is the unchanged mix (offline wiring test) — lyrics were "
                          "transcribed from the full mix"))
    elif a.get("separation") in (None, "off") and route and route.get("has_music"):
        out.append(_issue("info", "separation_off",
                          "No vocal separation: lyrics were transcribed from the full mix "
                          "(set FAL_KEY or --separation local for better lyrics)"))
    st = structure or {}
    if st.get("fallback_reason"):
        out.append(_issue("warning", "structure_fallback", st["fallback_reason"]
                          + " — sections come from the local (unlabelled) method instead"))
    if st and not st.get("skipped"):
        if not st.get("downbeats"):
            out.append(_issue("info", "downbeats", "No downbeats (beat tracker without bar "
                              "positions) — cut-on-downbeat not tested"))
        if st.get("segments") and not st.get("sections_labelled"):
            out.append(_issue("info", "sections", "Song sections are unlabelled (local novelty "
                              "boundaries, letters A/B/C) — not verse/chorus"))
        if st.get("n_beats_masked"):
            out.append(_issue("info", "beats_masked", f"{st['n_beats_masked']} beats outside music "
                              f"regions removed (beat trackers 'find' beats in speech)"))
    return out


def check_resolution(coverage: dict) -> list:
    c = coverage.get("shots_shorter_than_audio_window")
    if c and c["total"] and c["valid"] / c["total"] > 0.5:
        return [_issue("info", "audio_resolution",
                       f"{c['valid']}/{c['total']} shots are shorter than the 5 s audio "
                       f"window: their audio mood is shared with neighbouring shots — "
                       f"compare at section level, not shot level")]
    return []


# ------------------------------------------------------------------ report

def build_report(*, rows, coverage, route, lineage, music_summary, transcript, clap,
                 transcript_stats, vision_stats, repo_root, frames_dir, frame_fps,
                 audio_stack=None, structure=None) -> dict:
    issues = []
    issues += check_lineage(lineage)
    issues += check_vision(rows, vision_stats)
    issues += check_coverage(coverage)
    issues += check_route(route)
    ms = dict(music_summary or {})
    if structure and structure.get("bpm"):
        ms["tempo_bpm"] = structure["bpm"]          # structure beats supersede librosa's tempo
    issues += check_music(ms, route)
    issues += check_tempo_agreement(music_summary, structure)
    issues += check_audio_stack(audio_stack, structure, route)
    issues += check_transcript(transcript, clap, transcript_stats)
    issues += check_frames(rows, Path(repo_root), Path(frames_dir), frame_fps)
    issues += check_people_vs_motion(rows)
    issues += check_resolution(coverage)
    issues.sort(key=lambda x: (-SEV_RANK[x["severity"]], x["time_sec"] if x["time_sec"] is not None else -1))

    per_shot: dict[int, dict] = {}
    for it in issues:
        if it["shot_idx"] is None:
            continue
        d = per_shot.setdefault(it["shot_idx"], {"shot_idx": it["shot_idx"], "time_sec": it["time_sec"],
                                                 "score": 0, "checks": []})
        d["score"] += SEV_RANK[it["severity"]]
        d["checks"].append(it["check"])
    review = sorted(per_shot.values(), key=lambda d: (-d["score"], d["shot_idx"]))[:20]
    summary = {s: sum(1 for i in issues if i["severity"] == s) for s in ("critical", "warning", "info")}
    return {"summary": summary, "issues": issues, "review_first": review}
