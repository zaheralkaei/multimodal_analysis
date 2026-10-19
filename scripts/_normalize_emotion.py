"""Emotion normalization.

Round-1 audit found Gemini returns 34+ distinct emotion strings
('sultry', 'sensual', 'seductive', 'sultry intent', 'sensual intensity', ...)
when our palette only has 16. This module maps synonyms → canonical labels.

Usage:
    from _normalize_emotion import normalize_emotion
    canonical = normalize_emotion("sensual intensity")  # → "sensual"

Canonical emotions (match the JSON-mode prompt in phase2_vision.py):
    joyful, sad, angry, fearful, surprised, disgusted, neutral, contemplative,
    sensual, energetic, melancholic, anxious, playful, romantic, intense, confident
"""
from __future__ import annotations
import re

# Not an emotion: the frame shows no person. Kept apart from "neutral" (audit
# R9) — counting empty landscapes as neutral faces inflated "neutral".
NO_PEOPLE = "no people"
NO_PEOPLE_SYNONYMS = ["none", "none (no people)", "no people", "no people visible",
                      "no person", "no one", "nobody", "no humans", "n/a", "not applicable"]

# Map of canonical emotion → list of synonyms (lowercase, normalized)
CANONICAL_EMOTIONS = {
    "joyful":       ["joyful", "joy", "happy", "happiness", "delighted", "cheerful", "elated", "ecstatic", "gleeful", "blissful"],
    "sad":          ["sad", "sorrow", "sorrowful", "grief", "grieving", "mourning", "downcast", "dejected", "miserable", "tearful"],
    "angry":        ["angry", "anger", "furious", "rage", "enraged", "wrathful", "irate", "livid", "hostile"],
    "fearful":      ["fearful", "afraid", "scared", "terrified", "horrified", "panicked", "alarmed", "frightened", "anguished", "in pain", "suffering"],
    "surprised":    ["surprised", "shocked", "amazed", "astonished", "stunned", "startled", "bewildered"],
    "disgusted":    ["disgusted", "disgust", "repulsed", "revulsion", "contempt"],
    "neutral":      ["neutral", "calm", "indifferent", "composed", "detached", "objective"],
    "contemplative":["contemplative", "thoughtful", "pensive", "reflective", "meditative", "introspective", "brooding", "yearning", "hopeful"],
    "sensual":      ["sensual", "sultry", "seductive", "seduct", "intimate", "lustful", "provocative", "passionate", "sultry intent", "sensual intensity", "sultry/confident"],
    "energetic":    ["energetic", "vigorous", "lively", "spirited", "dynamic", "vibrant", "exhilarated", "pumped", "excited", "energetic, intense"],
    "melancholic":  ["melancholic", "melancholy", "wistful", "longing", "mournful", "doleful", "plaintive", "blue", "forlorn"],
    "anxious":      ["anxious", "anxiety", "worried", "nervous", "uneasy", "apprehensive", "tense", "distressed", "stressed", "agitated"],
    "playful":      ["playful", "mischievous", "teasing", "flirtatious", "whimsical", "fun", "frolicsome"],
    "romantic":     ["romantic", "tender", "loving", "affectionate", "amorous", "sentimental"],
    "intense":      ["intense", "fierce", "fervent", "vehement", "intense or fierce", "intense focus", "serious intensity", "fierce", "in pain/suffering"],
    "confident":    ["confident", "assured", "self-assured", "bold", "commanding", "dominant", "empowered", "confident/seductive"],
}

# Flat lookup: lowercase syn → canonical
_SYNONYM_TO_CANONICAL: dict[str, str] = {}
for canonical, syns in CANONICAL_EMOTIONS.items():
    for s in syns:
        _SYNONYM_TO_CANONICAL[s.lower()] = canonical
for s in NO_PEOPLE_SYNONYMS:
    _SYNONYM_TO_CANONICAL[s] = NO_PEOPLE
_BY_LENGTH = sorted(_SYNONYM_TO_CANONICAL, key=len, reverse=True)


def normalize_emotion(raw: str) -> str:
    """Map a Gemini-emitted emotion string to our canonical 16.

    Returns:
        - One of the 16 canonical strings, OR
        - "other" if the input is empty or doesn't match any synonym.
    """
    if not raw:
        return "other"
    text = raw.strip().lower()
    # Strip trailing punctuation
    text = text.strip(".,;:!?*\"'")
    # Direct hit
    if text in _SYNONYM_TO_CANONICAL:
        return _SYNONYM_TO_CANONICAL[text]
    # Try stripping common suffixes
    for suffix in ["*", " — ", ": also fits", " too"]:
        if suffix in text:
            text = text.split(suffix)[0].strip()
    if text in _SYNONYM_TO_CANONICAL:
        return _SYNONYM_TO_CANONICAL[text]
    # The prompt asks for one or two words; a long answer is the model
    # explaining itself, not a label — don't guess an emotion out of it.
    if len(text.split()) > 6:
        return "other"
    # Longest synonym that appears as a WHOLE word/phrase (audit R9: plain
    # substring matching mapped "courageous"/"dangerous"/"stranger" -> angry
    # via "rage"/"anger", and "funeral" -> playful via "fun").
    for syn in _BY_LENGTH:
        if re.search(rf"(?<![a-z]){re.escape(syn)}(?![a-z])", text):
            return _SYNONYM_TO_CANONICAL[syn]
    return "other"


def list_canonical() -> list[str]:
    """Return the canonical 16 emotion labels (for downstream code)."""
    return list(CANONICAL_EMOTIONS.keys())


if __name__ == "__main__":
    # Self-test
    tests = [
        ("sensual intensity", "sensual"),
        ("sultry/confident", "sensual"),  # first word matches
        ("Sultry", "sensual"),
        ("Based on the body language and downcast expressions of the", "other"),
        ("intense or fierce", "intense"),
        ("", "other"),
        ("joyful", "joyful"),
        ("hopeful", "contemplative"),
    ]
    print("Self-test:")
    for raw, expected in tests:
        got = normalize_emotion(raw)
        ok = "OK" if got == expected else "FAIL"
        print(f"  {ok}: normalize_emotion({raw!r}) = {got!r} (expected {expected!r})")