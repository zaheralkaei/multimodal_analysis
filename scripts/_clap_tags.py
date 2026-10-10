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
ALL_TAGS = MOOD_TAGS + SECTION_TAGS + INSTRUMENT_TAGS