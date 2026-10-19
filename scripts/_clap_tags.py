"""Shared CLAP tag vocabulary (round-4 audit fix: phase 5/7/8 each hand-copied
this list, and one edit would silently desync them)."""

# Fixed vocabulary — chosen for music-video analysis.
MOOD_TAGS = [
    "happy and bright",
    "sad and melancholic",
    "aggressive and intense",
    "romantic and tender",
    "triumphant and epic",
    "calm and peaceful",
    "tense and anxious",
    "dreamy and ethereal",
    "dark and ominous",
    "playful and whimsical",
    "lonely and introspective",
    "powerful and confident",
]
SECTION_TAGS = [
    "intro",
    "verse",
    "chorus",
    "bridge",
    "outro",
    "instrumental break",
    "vocal only",
]
INSTRUMENT_TAGS = [
    "acoustic guitar",
    "electric guitar",
    "piano",
    "drums and percussion",
    "bass guitar",
    "synthesizer",
    "strings orchestra",
    "vocal only no instruments",
]
# Is there music at all? (audit R9) librosa reports a "tempo" for any audio,
# speech included, so tempo cannot tell a speech from a song. CLAP's
# zero-shot music/speech split can: on the round-9 check a song scored music
# in 100 % of windows, a TTS speech in 0 %.
AUDIO_TYPE_TAGS = [
    "music",
    "speech",
    "silence",
]
ALL_TAGS = MOOD_TAGS + SECTION_TAGS + INSTRUMENT_TAGS + AUDIO_TYPE_TAGS

# Probabilities are normalized WITHIN each group (phase 5, round 8).
TAG_GROUPS = {"mood": MOOD_TAGS, "section": SECTION_TAGS, "instrument": INSTRUMENT_TAGS,
              "audio_type": AUDIO_TYPE_TAGS}


# Silence is decided by LOUDNESS, not by CLAP: on the round-10 calibration
# video CLAP called a near-silent window "speech" (p=0.88).
SILENCE_DBFS = -45.0


def rms_dbfs(chunk) -> float:
    import numpy as np
    rms = float(np.sqrt(np.mean(np.square(np.asarray(chunk, dtype=np.float64))))) if len(chunk) else 0.0
    return 20.0 * np.log10(max(rms, 1e-10))


def window_audio_type(row: dict) -> str | None:
    """'music' | 'speech' | 'silence' for one CLAP row, or None if the row
    predates the audio_type group. A window quieter than SILENCE_DBFS is
    silence; otherwise the louder of CLAP's music/speech scores decides."""
    try:
        if row.get("rms_dbfs") not in (None, "") and float(row["rms_dbfs"]) < SILENCE_DBFS:
            return "silence"
        scores = {t: float(row[t]) for t in AUDIO_TYPE_TAGS}
    except (KeyError, TypeError, ValueError):
        return None
    if row.get("rms_dbfs") not in (None, ""):
        scores.pop("silence", None)           # loud enough: not silence
    return max(scores, key=scores.get)


def music_share(rows: list) -> float | None:
    """Fraction of CLAP windows whose audio type is music (None if unknown)."""
    kinds = [window_audio_type(r) for r in rows or []]
    kinds = [k for k in kinds if k is not None]
    if not kinds:
        return None
    return kinds.count("music") / len(kinds)
