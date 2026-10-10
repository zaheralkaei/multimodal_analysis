"""
Content-type detection for the multimodal pipeline (UI audit round 6).

The pipeline originally assumed "music video". The same 8-phase extraction
applies to songs, films/episodes, vlogs and speeches — but the dashboard's
tracks, findings and caveats should follow whichever modalities actually
carry signal: a political speech has no tempo, a film has no lyrics.

detect_content_type() inspects the phase outputs and returns a profile dict:
    {"type": "song"|"speech"|"film"|"vlog"|"series"|"other",
     "confidence": float | None,   # None when explicitly overridden
     "reasons": [str, ...]}        # human-readable evidence, shown in the UI

Detection is deliberately simple, ordered and explainable — an explicit
--content-type always wins, and the reasons are rendered in the dashboard so
a wrong auto-detection is visible instead of silent.
"""
from __future__ import annotations

CONTENT_TYPES = ("auto", "song", "speech", "film", "series", "vlog", "other")


def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def detect_content_type(shots: list, music_summary: dict, transcript: list,
                        video_duration: float = 0.0,
                        clap_rows: list = None,
                        override: str = "auto") -> dict:
    """Ordered, explainable rules; first match wins."""
    if override and override != "auto":
        return {"type": override, "confidence": None,
                "reasons": ["explicitly set (--content-type)]; statistics ignored"]}

    shots = shots or []
    n = len(shots)
    durations = [_f(s.get("duration_sec")) for s in shots if s.get("duration_sec")]
    avg_shot = (sum(durations) / len(durations)) if durations else 0.0
    if not video_duration:
        video_duration = max((float(s.get("end_sec", 0)) for s in shots), default=0.0)
    duration = max(_f(video_duration), 1.0)

    # Transcript signal (speechiness)
    seg_chars = 0
    seg_dur = 0.0
    for t in transcript or []:
        seg_chars += len(t.get("text", ""))
        try:
            seg_dur += max(0.0, float(t.get("end_sec", 0)) - float(t.get("start_sec", 0)))
        except (TypeError, ValueError):
            pass
    talk_coverage = seg_dur / duration  # fraction of the video with speech
    chars_per_min = seg_chars / duration * 60.0

    # Music signal
    tempo = _f((music_summary or {}).get("tempo_bpm"))
    n_beats = int(_f((music_summary or {}).get("n_beats")))
    beats_per_sec = n_beats / duration
    has_music = bool(tempo) and n_beats >= 10

    # Camera staticness (vlog single-camera vs film's edit-heavy grammar)
    motions = [s.get("camera_motion", "") for s in shots]
    static_frac = (motions.count("static") / len(motions)) if motions else 0.0

    reasons: list[str] = []
    reasons.append(f"{n} shots (avg {avg_shot:.1f}s), "
                   f"{talk_coverage * 100:.0f}% talk coverage, "
                   f"{chars_per_min:.0f} lyric/speech chars/min")
    if has_music:
        reasons.append(f"music detected: {tempo:.0f} BPM, {beats_per_sec:.1f} beats/s")
    else:
        reasons.append("no usable tempo/beat track")
    if motions:
        reasons.append(f"camera static in {static_frac * 100:.0f}% of shots")

    # --- ordered rules ---
    # Speech first: no music + heavy talk, but a *staged* speech has very few
    # cuts (avg shot ≥ 8s). A talk-heavy single-camera vlog has short jump
    # cuts, so it falls through to the vlog rule instead.
    if not has_music and (talk_coverage >= 0.4 or chars_per_min >= 800) and avg_shot >= 8.0:
        ctype = "speech"
        reasons.append("long uncut camera takes with dense speech")
    elif has_music and (chars_per_min < 700 or tempo >= 50):
        ctype = "song"
    elif not has_music and talk_coverage >= 0.1 and static_frac >= 0.5 and n < 60:
        ctype = "vlog"
    elif avg_shot >= 6 or (not has_music and not transcript):
        ctype = "film"
    elif n >= 60:
        ctype = "film"  # series episodes share film's grammar; hard to separate
    else:
        ctype = "other"

    # Confidence: share of the strongest signal vs the rest (0..1). Ordered
    # rules make this approximate; it is displayed, not relied upon.
    strength = {
        "song": (2.0 if has_music else 0.0) + (1.0 if chars_per_min < 700 else 0.0),
        "speech": (2.0 if (not has_music) else 0.0) + (2.0 if talk_coverage >= 0.4 else 1.0),
        "film": (1.5 if avg_shot >= 6 else 0.0) + (1.0 if not transcript else 0.0),
        "vlog": (1.5 if static_frac >= 0.5 else 0.0) + (1.0 if talk_coverage >= 0.1 else 0.0),
        "series": 0.0,  # only reachable via override; auto cannot claim it
        "other": 0.5,
    }
    total = max(1.0, sum(strength.values()))
    confidence = round(strength[ctype] / total, 2)

    reasons.append(f"classified as {ctype}: strongest signal score "
                   f"{strength[ctype]:.1f} of {total:.1f}")
    return {"type": ctype, "confidence": confidence, "reasons": reasons}


def profile_line(profile: dict) -> str:
    """One-line summary for the banner/findings."""
    conf = profile.get("confidence")
    how = "explicitly set" if conf is None else "auto-detected"
    return f"{profile['type']} ({how})"


def explain(profile: dict) -> str:
    """Human-readable reasons joined for a caveat/UI bullet."""
    return "; ".join(profile.get("reasons", []))