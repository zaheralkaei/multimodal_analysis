"""Round-4 audit fixes F4/F5/F6: emotion colors, dashboard guards, phase-7 join."""
import json
from pathlib import Path

import pytest

from conftest import load_phase


@pytest.fixture()
def p7(tmp_processed):
    return load_phase("phase7_sync", tmp_processed)


@pytest.fixture()
def p8(tmp_path):
    return load_phase("phase8_dashboard", tmp_path / "reports")


class TestEmotionColors:
    """AUDIT_R4 F5: all 16 canonical emotions must have real colors."""

    def test_all_canonical_covered(self, p8):
        from _normalize_emotion import list_canonical
        for e in list_canonical():
            c = p8.color_for_emotion(e)
            # neutral is the one emotion that legitimately renders gray
            assert c != "#999999" or e == "neutral", \
                f"canonical emotion {e!r} fell through to gray"

    def test_intense_not_tense_regression(self, p8):
        # Old substring matching: 'tense' in 'intense' -> tense's color.
        assert p8.color_for_emotion("intense") == "#c0392b"
        assert p8.color_for_emotion("tense") == "#ff9896"

    def test_sensual_has_color(self, p8):
        assert p8.color_for_emotion("sensual") == "#e75480"

    def test_unknown_still_gray(self, p8):
        assert p8.color_for_emotion("") == "#999999"
        assert p8.color_for_emotion("zzz") == "#999999"

    def test_word_boundary_not_substring(self, p8):
        assert p8.color_for_emotion("intensified") == "#999999"  # 'intense' must not match
        assert p8.color_for_emotion("sadface") == "#999999"


class TestPhase7Join:
    def _write_inputs(self, proc: Path):
        shots = [{"shot_idx": 0, "start_sec": 0.0, "end_sec": 6.0, "duration_sec": 6.0,
                  "mid_sec": 3.0, "mid_frame_path": "d/f1.jpg"},
                 {"shot_idx": 1, "start_sec": 6.0, "end_sec": 8.0, "duration_sec": 2.0,
                  "mid_sec": 7.0, "mid_frame_path": "d/f2.jpg"}]
        (proc / "shots.json").write_text(json.dumps(shots))
        (proc / "shot_vision.csv").write_text(
            "shot_idx,caption,emotion,colors,entities,location,lighting,composition,camera,mid_frame\n"
            "0,cap0,sensual,red,x,indoor,soft,closeup,static,d/f1.jpg\n"
            "1,cap1,joyful,blue,y,indoor,soft,closeup,static,d/f2.jpg\n")
        (proc / "shot_camera.csv").write_text(
            "shot_idx,camera_motion\n0,pan-left\n1,static\n")
        (proc / "transcript.csv").write_text(
            "start_sec,end_sec,text\n1.0,2.0,hello world\n7.0,7.5,zwei\n")
        (proc / "audio_clap.csv").write_text(
            "start_sec,end_sec,happy and bright,sad and melancholic\n"
            "0.0,5.0,0.6,0.1\n5.0,10.0,0.4,0.3\n")
        (proc / "music_features.csv").write_text(
            "second,start_sec,end_sec,rms_energy,n_beats\n"
            "0,0.0,1.0,0.2,1\n1,1.0,2.0,0.3,0\n6,6.0,7.0,0.5,1\n")
        (proc / "music_summary.json").write_text(json.dumps({"beat_times": [0.02, 6.01, 12.0]}))
        (proc / "shot_vision_stats.json").write_text(json.dumps(
            {"provider": "openrouter", "model": "google/gemma-4-31b-it",
             "endpoint": "https://openrouter.ai/api/v1"}))

    def test_join_and_provenance(self, p7, tmp_processed):
        self._write_inputs(tmp_processed)
        import csv, os
        rows = list(csv.DictReader((tmp_processed / "sync_per_shot.csv").open("w")), ) if False else None
        # run the join directly
        from _clap_tags import MOOD_TAGS
        shots = json.loads((tmp_processed / "shots.json").read_text())
        vision = list(csv.DictReader((tmp_processed / "shot_vision.csv").open()))
        camera = list(csv.DictReader((tmp_processed / "shot_camera.csv").open()))
        transcript = list(csv.DictReader((tmp_processed / "transcript.csv").open()))
        clap = list(csv.DictReader((tmp_processed / "audio_clap.csv").open()))
        music = list(csv.DictReader((tmp_processed / "music_features.csv").open()))
        beats = [0.02, 6.01, 12.0]
        rows, stats = p7.join_data(shots, vision, camera, transcript, clap, music, beats, MOOD_TAGS)
        assert len(rows) == 2
        assert rows[0]["vision_caption"] == "cap0"
        assert rows[0]["camera_motion"] == "pan-left"
        assert rows[0]["cut_on_beat"] == True   # start 0.0 vs beat 0.02
        assert rows[1]["cut_on_beat"] == True   # start 6.0 vs beat 6.01
        assert rows[0]["n_lyric_segments"] == 1
        assert rows[1]["n_lyric_segments"] == 1  # 7.0-7.5 overlaps 6-8
        assert rows[0]["audio_top_mood"] == "happy and bright"
        assert stats["cuts_on_beat"] == 2
        assert stats["total_shots"] == 2

    def test_main_writes_provenance_and_empty_csv(self, p7, tmp_processed, monkeypatch):
        monkeypatch.setattr("sys.argv", ["phase7_sync.py"])
        self._write_inputs(tmp_processed)
        import csv, io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            p7.main()
        stats = json.loads((tmp_processed / "sync_stats.json").read_text())
        # F4: provenance propagated into sync_stats
        assert stats["vision_model"] == "google/gemma-4-31b-it"
        assert stats["vision_provider"] == "openrouter"
        # always-write fix: file exists even for a run with no rows
        (tmp_processed / "shots.json").write_text("[]")
        with contextlib.redirect_stdout(io.StringIO()):
            p7.main()
        content = (tmp_processed / "sync_per_shot.csv").read_text()
        assert "shot_idx" in content  # header written even with zero shots


class TestPhase8Guards:
    """AUDIT_R4 F6: partial pipelines must produce a dashboard, not crash."""

    def _minimal(self, proc: Path):
        (proc / "sync_per_shot.csv").write_text("shot_idx,start_sec,end_sec,duration_sec\n")
        (proc / "sync_stats.json").write_text("{}")

    def test_header_only_sync_csv(self, p8, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["phase8_dashboard.py"])
        proc = tmp_path / "data" / "x"
        proc.mkdir(parents=True)
        import os
        os.environ["PROCESSED_DIR"] = str(proc)
        p8b = load_phase("phase8_dashboard", proc)
        reports = tmp_path / "reports"
        p8b.REPORTS = reports
        os.environ["REPORTS_DIR"] = str(reports)
        self._minimal(proc)
        assert p8b.main() == 0
        html = (reports / "dashboard.html").read_text(encoding="utf-8")
        assert "dashboard" in html.lower()

    def test_missing_music_and_transcript_no_crash(self, p8, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["phase8_dashboard.py"])
        proc = tmp_path / "data" / "y"
        proc.mkdir(parents=True)
        (proc / "sync_per_shot.csv").write_text(
            "shot_idx,start_sec,end_sec,duration_sec,vision_emotion,vision_caption,"
            "camera_motion,audio_top_mood,lyric_text,mid_frame,vision_camera_from_vlm\n"
            "0,0,5,5,sensual,cap,static,happy,b,df.jpg,static\n")
        (proc / "sync_stats.json").write_text(json.dumps({"vision_model": "google/gemma-4-31b-it"}))
        import os
        os.environ["PROCESSED_DIR"] = str(proc)
        p8b = load_phase("phase8_dashboard", proc)
        reports = tmp_path / "reports"
        p8b.REPORTS = reports
        os.environ["REPORTS_DIR"] = str(reports)
        assert p8b.main() == 0
        html = (reports / "dashboard.html").read_text(encoding="utf-8")
        # F4: the model name that actually ran is rendered, and no hardcode
        assert "google/gemma-4-31b-it" in html

    def test_single_clap_window_does_not_crash(self, p8, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["phase8_dashboard.py"])
        proc = tmp_path / "data" / "z"
        proc.mkdir(parents=True)
        (proc / "sync_per_shot.csv").write_text(
            "shot_idx,start_sec,end_sec,duration_sec,vision_emotion,vision_caption,"
            "camera_motion,audio_top_mood,lyric_text,mid_frame,vision_camera_from_vlm\n"
            "0,0,5,5,sensual,cap,static,happy,b,df.jpg,static\n")
        (proc / "sync_stats.json").write_text("{}")
        (proc / "audio_clap.csv").write_text(
            "start_sec,end_sec,happy and bright,sad and melancholic\n0.0,5.0,0.9,0.1\n")
        import os
        os.environ["PROCESSED_DIR"] = str(proc)
        p8b = load_phase("phase8_dashboard", proc)
        reports = tmp_path / "reports"
        p8b.REPORTS = reports
        os.environ["REPORTS_DIR"] = str(reports)
        assert p8b.main() == 0