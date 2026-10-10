"""Round-6: content-type detection + adaptive dashboard (song/speech/film/vlog)."""
import json
import os
import sys
from pathlib import Path

import pytest

from conftest import load_phase


def _shots(n, dur=3.0, motion="pan-left"):
    return [{"shot_idx": i, "start_sec": i * dur, "end_sec": (i + 1) * dur,
             "duration_sec": dur, "camera_motion": motion} for i in range(n)]


def _transcript(segments):
    return [{"start_sec": s, "end_sec": e, "text": t} for s, e, t in segments]


class TestDetectContentType:
    def test_song(self):
        p = _profile_mod().detect_content_type(
            shots=_shots(60, 3.0),
            music_summary={"tempo_bpm": 128, "n_beats": 430},
            transcript=_transcript([(0, 2, "la la la")] * 10),
            video_duration=180.0)
        assert p["type"] == "song"
        assert p["confidence"] is not None

    def test_speech_no_music_high_talk(self):
        p = _profile_mod().detect_content_type(
            shots=_shots(10, 30.0),           # long shots, static camera
            music_summary={},                  # no tempo
            transcript=_transcript([(i * 30.0, i * 30.0 + 27, "x" * 400) for i in range(10)]),
            video_duration=300.0)
        assert p["type"] == "speech"
        # reasons shown in the UI must explain the call
        assert any("talk coverage" in r for r in p["reasons"])

    def test_film_no_music_no_speech_long_shots(self):
        p = _profile_mod().detect_content_type(
            shots=_shots(40, 8.0), music_summary={}, transcript=[],
            video_duration=320.0)
        assert p["type"] == "film"

    def test_vlog_static_camera_moderate_talk(self):
        p = _profile_mod().detect_content_type(
            shots=_shots(30, 5.0, motion="static"),
            music_summary={},
            transcript=_transcript([(i * 5.0, i * 5.0 + 2.0, "hey guys") for i in range(30)]),
            video_duration=150.0)
        assert p["type"] == "vlog"

    def test_override_beats_auto(self):
        p = _profile_mod().detect_content_type(
            shots=_shots(60), music_summary={"tempo_bpm": 128, "n_beats": 430},
            transcript=[], video_duration=180.0, override="series")
        assert p["type"] == "series"
        assert p["confidence"] is None
        assert any("explicitly" in r for r in p["reasons"])

    def test_empty_data_falls_back(self):
        p = _profile_mod().detect_content_type([], {}, [], 0.0)
        assert p["type"] in ("other", "film", "speech")  # must not crash


def _profile_mod():
    import importlib
    if "_content_profile" not in sys.modules:
        return importlib.import_module("_content_profile")
    return importlib.import_module("_content_profile")


class TestAdaptiveDashboard:
    """Timeline only draws tracks that have data; profile is rendered."""

    def _run(self, tmp_path, video_id, files):
        proc = tmp_path / "data" / video_id
        proc.mkdir(parents=True)
        reports = tmp_path / "reports" / video_id
        (proc / "sync_per_shot.csv").write_text(files["sync"], encoding="utf-8")
        (proc / "sync_stats.json").write_text("{}")
        for name, content in files.items():
            if name not in ("sync",):
                (proc / name).write_text(content, encoding="utf-8")
        os.environ["PROCESSED_DIR"] = str(proc)
        os.environ["REPORTS_DIR"] = str(reports)
        mod = load_phase("phase8_dashboard")
        import io, contextlib
        old_argv = sys.argv
        sys.argv = ["phase8_dashboard.py"]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = mod.main()
        finally:
            sys.argv = old_argv
        assert rc == 0
        return (reports / "dashboard.html").read_text(encoding="utf-8")

    SYNC = ("shot_idx,start_sec,end_sec,duration_sec,vision_emotion,vision_caption,"
            "camera_motion,audio_top_mood,lyric_text,mid_frame,vision_camera_from_vlm\n"
            "0,0,4,4,joyful,cap,static,happy,b,df.jpg,static\n"
            "1,4,8,4,sad,cap2,static,happy,b,df2.jpg,static\n")

    def test_speech_hides_music_tracks(self, tmp_path):
        # staged speech: two uncut 30s takes, one transcript line, no music
        sync = ("shot_idx,start_sec,end_sec,duration_sec,vision_emotion,vision_caption,"
                "camera_motion,audio_top_mood,lyric_text,mid_frame,vision_camera_from_vlm\n"
                "0,0,30,30,joyful,cap,static,happy,b,df.jpg,static\n"
                "1,30,60,30,sad,cap2,static,happy,b,df2.jpg,static\n")
        speech = self._run(tmp_path, "speechvid1", {
            "sync": sync,
            "transcript.csv": "start_sec,end_sec,text\n0.0,29.9,Ladies and gentlemen, thank you\n",
        })
        assert "Audio mood (CLAP" not in speech
        assert "Music energy (RMS" not in speech
        assert "transcripts" in speech                    # speech keeps its track
        assert "Content type: <b>speech</b>" in speech    # profile rendered + reason shown
        # music-only caveat suppressed
        assert "Cut on beat" not in speech
        assert "Krumhansl" not in speech

    def test_song_shows_all_tracks(self, tmp_path):
        song = self._run(tmp_path, "songvid11", {
            "sync": self.SYNC,
            "audio_clap.csv": ("start_sec,end_sec,happy and bright,sad and melancholic\n"
                               "0.0,5.0,0.9,0.1\n5.0,10.0,0.8,0.2\n"),
            "music_features.csv": "second,start_sec,end_sec,rms_energy,n_beats\n0,0,1,0.2,1\n",
            "transcript.csv": "start_sec,end_sec,text\n0.0,2.0,la la\n",
            "music_summary.json": json.dumps({"tempo_bpm": 128, "n_beats": 40, "beat_times": [0, 0.5]}),
        })
        assert "Audio mood (CLAP" in song
        assert "Music energy (RMS" in song
        assert "transcripts" in song
        assert "Content type: <b>song</b>" in song
        assert "Krumhansl" in song

    def test_empty_runs_only_shots_track(self, tmp_path):
        bare = self._run(tmp_path, "barevid12", {"sync": self.SYNC})
        # 1-track timeline, no empty frames
        assert "Shots timeline (color = visual emotion)" in bare
        assert "Audio mood (CLAP" not in bare
        assert "Music energy (RMS" not in bare
        assert "transcripts" not in bare