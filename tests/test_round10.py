"""Round 10: pipeline integration — lineage, early routing, validity masks,
coverage, cross-module QA, frame timing, calibration helpers."""
import csv
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from conftest import load_phase


# ---------------------------------------------------------------- lineage
class TestLineage:
    @pytest.fixture()
    def lin(self):
        return load_phase("_lineage")

    def test_clean_record_checks_ok(self, lin, tmp_path):
        (tmp_path / "in.txt").write_text("a")
        (tmp_path / "out.txt").write_text("b")
        lin.record(tmp_path, "stepA", [tmp_path / "in.txt"], [tmp_path / "out.txt"], {"x": 1})
        res = lin.check(tmp_path)
        assert res["has_lineage"] and res["problems"] == [] and res["ok_steps"] == ["stepA"]

    def test_changed_input_is_stale(self, lin, tmp_path):
        (tmp_path / "in.txt").write_text("a")
        (tmp_path / "out.txt").write_text("b")
        lin.record(tmp_path, "stepA", [tmp_path / "in.txt"], [tmp_path / "out.txt"])
        (tmp_path / "in.txt").write_text("a changed upstream")
        res = lin.check(tmp_path)
        assert res["problems"] == [{"step": "stepA", "kind": "stale", "file": "in.txt"}]
        assert lin.bad_steps(res) == {"stepA"}

    def test_replaced_output_is_overwritten(self, lin, tmp_path):
        (tmp_path / "in.txt").write_text("a")
        (tmp_path / "out.txt").write_text("b")
        lin.record(tmp_path, "stepA", [tmp_path / "in.txt"], [tmp_path / "out.txt"])
        (tmp_path / "out.txt").write_text("written by another run")
        assert lin.check(tmp_path)["problems"][0]["kind"] == "overwritten"

    def test_no_lineage_file(self, lin, tmp_path):
        assert lin.check(tmp_path) == {"has_lineage": False, "problems": [], "ok_steps": [], "missing": []}


# ---------------------------------------------------------------- routing
class TestRoute:
    @pytest.fixture()
    def rc(self, tmp_path):
        return load_phase("route_content", tmp_path)

    def test_music_detected(self, rc):
        r = rc.decide({"music": 0.8, "speech": 0.1, "silence": 0.1})
        assert r["has_music"] and r["audio_class"] == "music" and r["has_music_basis"] == "clap"
        assert r["modules"]["cut_on_beat_test"]

    def test_speech_turns_music_analyses_off(self, rc):
        r = rc.decide({"music": 0.05, "speech": 0.9, "silence": 0.05})
        assert not r["has_music"] and r["audio_class"] == "speech"
        assert not r["modules"]["beat_analysis"] and r["modules"]["transcription"]

    def test_override_wins_but_disagreement_is_warned(self, rc):
        r = rc.decide({"music": 0.05, "speech": 0.9, "silence": 0.05}, "song")
        assert r["has_music"] and r["has_music_basis"] == "override"
        assert any(x.startswith("WARNING") for x in r["reasons"])

    def test_unverified_without_clap(self, rc):
        r = rc.decide(None)
        assert r["has_music"] and r["has_music_basis"] == "unverified"

    def test_windows_spread_over_whole_file(self, rc, tmp_path):
        """Calibration found the sampler only listened to the first windows of
        short files (spacing used n-1 instead of m-1)."""
        import numpy as np
        import soundfile as sf
        sf.write(str(tmp_path / "a.wav"), np.zeros(48000 * 36, np.float32), 48000)
        wins, _ = rc.sample_windows(tmp_path / "a.wav", n=24, window_sec=5.0)
        starts = [s for s, _ in wins]
        assert starts[0] == 0.0 and starts[-1] == 30.0 and len(starts) == 7


class TestSilenceByLoudness:
    def test_quiet_window_is_silence_even_if_clap_says_speech(self):
        from _clap_tags import window_audio_type
        assert window_audio_type({"music": 0.0, "speech": 0.88, "silence": 0.12,
                                  "rms_dbfs": -70}) == "silence"

    def test_loud_window_is_never_silence(self):
        from _clap_tags import window_audio_type
        assert window_audio_type({"music": 0.2, "speech": 0.1, "silence": 0.7,
                                  "rms_dbfs": -12}) == "music"

    def test_rms_dbfs(self):
        import numpy as np
        from _clap_tags import rms_dbfs
        assert rms_dbfs(np.ones(100)) == pytest.approx(0.0)
        assert rms_dbfs(np.zeros(100)) < -150


# ---------------------------------------------------------------- phase 7
@pytest.fixture()
def p7(tmp_processed):
    return load_phase("phase7_sync")


SHOTS = [{"shot_idx": 0, "start_sec": 0.0, "end_sec": 2.0},
         {"shot_idx": 1, "start_sec": 2.0, "end_sec": 9.0}]


class TestJoinRound10:
    def test_overlap_weighted_audio(self, p7):
        clap = [{"start_sec": "0", "end_sec": "5", "happy and bright": "1.0"},
                {"start_sec": "5", "end_sec": "10", "happy and bright": "0.0"}]
        rows, _ = p7.join_data(SHOTS, [], [], [], clap, [], [], ["happy and bright"])
        # shot 1 = 3 s of window A (1.0) + 4 s of window B (0.0) -> 3/7, not 0.5
        assert rows[1]["audio_mood_happy and bright"] == pytest.approx(3 / 7, abs=1e-3)

    def test_no_music_skips_beat_test_and_masks_mood(self, p7):
        clap = [{"start_sec": "0", "end_sec": "5", "happy and bright": "0.9"}]
        rows, stats = p7.join_data(SHOTS, [], [], [], clap, [], [0.0, 2.0, 4.0],
                                   ["happy and bright"], music_present=False)
        assert stats["cut_beat_alignment"]["skipped"]
        assert rows[1]["cut_on_beat"] == "" and stats["total_cuts"] == 0
        assert not rows[0]["valid_audio_mood"] and rows[0]["audio_top_mood"] == ""

    def test_mood_invalid_over_speech_windows(self, p7):
        clap = [{"start_sec": "0", "end_sec": "10", "happy and bright": "0.9",
                 "music": "0.1", "speech": "0.9", "silence": "0", "rms_dbfs": "-20"}]
        rows, _ = p7.join_data(SHOTS, [], [], [], clap, [], [], ["happy and bright"],
                               music_present=True)
        assert not rows[0]["valid_audio_mood"]

    def test_coverage_and_validity(self, p7):
        vision = [{"shot_idx": "0", "start_sec": "0.0", "caption": "ok", "status": "ok"},
                  {"shot_idx": "1", "start_sec": "2.0", "caption": "[error: 500]", "status": "error"}]
        camera = [{"shot_idx": "0", "start_sec": "0.0", "camera_motion": "pan-left",
                   "camera_confidence": "high"},
                  {"shot_idx": "1", "start_sec": "2.0", "camera_motion": "unknown"}]
        rows, stats = p7.join_data(SHOTS, vision, camera, [], [], [], [], [])
        assert [r["valid_vision"] for r in rows] == [True, False]
        assert [r["valid_camera"] for r in rows] == [True, False]
        cov = stats["coverage"]
        assert cov["vision"] == {"valid": 1, "total": 2, "pct": 50.0}
        assert cov["camera_high_confidence"]["valid"] == 1
        assert cov["shots_shorter_than_audio_window"]["valid"] == 1

    def test_mock_vision_is_not_valid(self, p7):
        vision = [{"shot_idx": "0", "start_sec": "0.0", "caption": "MOCK caption", "status": "mock"}]
        rows, _ = p7.join_data(SHOTS[:1], vision, [], [], [], [], [], [])
        assert rows[0]["valid_vision"] is False

    def test_main_refuses_stale_stream(self, p7, tmp_processed, monkeypatch):
        """Lineage: captions made for an older shots.json are not joined."""
        lin = load_phase("_lineage")
        shots = [{"shot_idx": 0, "start_sec": 0.0, "end_sec": 2.0, "duration_sec": 2.0}]
        (tmp_processed / "shots.json").write_text(json.dumps(shots))
        (tmp_processed / "shot_vision.csv").write_text(
            "shot_idx,start_sec,caption,status\n0,0.0,old caption,ok\n")
        lin.record(tmp_processed, "phase2", [tmp_processed / "shots.json"],
                   [tmp_processed / "shot_vision.csv"])
        shots[0]["end_sec"] = 2.5                       # phase 1 re-ran afterwards
        (tmp_processed / "shots.json").write_text(json.dumps(shots))
        monkeypatch.setattr("sys.argv", ["phase7_sync.py"])
        with redirect_stdout(io.StringIO()):
            assert p7.main() == 0
        stats = json.loads((tmp_processed / "sync_stats.json").read_text())
        assert stats["streams_not_joined"] == ["vision"]
        row = next(csv.DictReader((tmp_processed / "sync_per_shot.csv").open()))
        assert row["vision_caption"] == ""
        qa = json.loads((tmp_processed / "qa_report.json").read_text())
        assert any(i["check"] == "lineage" and i["severity"] == "critical" for i in qa["issues"])


# ---------------------------------------------------------------- QA
class TestQA:
    @pytest.fixture()
    def qa(self):
        return load_phase("_qa")

    def _img(self, path, seed, dark=False, tint=None):
        cv2 = pytest.importorskip("cv2")
        import numpy as np
        rng = np.random.default_rng(seed)
        small = (rng.random((9, 16, 3)) * 255).astype(np.uint8)
        if tint is not None:                      # push colours to one channel
            small[..., tint] = np.maximum(small[..., tint], 200)
            small[..., (tint + 1) % 3] //= 3
        img = cv2.resize(small, (320, 180), interpolation=cv2.INTER_CUBIC)
        if dark:
            img = (img * 0.1).astype(np.uint8)
        cv2.imwrite(str(path), img)

    def test_identical_neighbours_flag_false_cut(self, qa, tmp_path):
        self._img(tmp_path / "a.jpg", 1)
        self._img(tmp_path / "b.jpg", 1)
        rows = [{"shot_idx": 0, "start_sec": 0, "end_sec": 1, "mid_frame": "a.jpg"},
                {"shot_idx": 1, "start_sec": 1, "end_sec": 2, "mid_frame": "b.jpg"}]
        out = qa.check_frames(rows, tmp_path, tmp_path / "none", 2)
        assert [i["check"] for i in out] == ["possible_false_cut"]

    def test_different_neighbours_not_flagged(self, qa, tmp_path):
        self._img(tmp_path / "a.jpg", 1)
        self._img(tmp_path / "b.jpg", 2)
        rows = [{"shot_idx": 0, "start_sec": 0, "end_sec": 1, "mid_frame": "a.jpg"},
                {"shot_idx": 1, "start_sec": 1, "end_sec": 2, "mid_frame": "b.jpg"}]
        assert qa.check_frames(rows, tmp_path, tmp_path / "none", 2) == []

    def test_dark_frames_are_not_judged(self, qa, tmp_path):
        """Dark footage produced 11/12 false alarms with colour histograms."""
        self._img(tmp_path / "a.jpg", 1, dark=True)
        self._img(tmp_path / "b.jpg", 1, dark=True)
        rows = [{"shot_idx": 0, "start_sec": 0, "end_sec": 1, "mid_frame": "a.jpg"},
                {"shot_idx": 1, "start_sec": 1, "end_sec": 2, "mid_frame": "b.jpg"}]
        assert qa.check_frames(rows, tmp_path, tmp_path / "none", 2) == []

    def test_missed_cut_inside_long_shot(self, qa, tmp_path):
        frames = tmp_path / "frames"
        frames.mkdir()
        for n in range(1, 9):                 # 4 s shot at 2 fps; picture changes at 2 s
            self._img(frames / f"frame_{n:05d}.jpg", 1 if n <= 4 else 2,
                      tint=2 if n <= 4 else 0)   # reddish scene -> bluish scene
        rows = [{"shot_idx": 0, "start_sec": 0.0, "end_sec": 4.0, "mid_frame": ""}]
        out = qa.check_frames(rows, tmp_path, frames, 2)
        assert [i["check"] for i in out] == ["possible_missed_cut"]
        assert out[0]["time_sec"] == 2.0

    def test_lyrics_over_silence(self, qa):
        clap = [{"start_sec": "0", "end_sec": "5", "music": "0.1", "speech": "0.8",
                 "silence": "0.1", "rms_dbfs": "-70"}]
        tr = [{"start_sec": "1", "end_sec": "3", "text": "thanks for watching", "suspect": "False"}]
        out = qa.check_transcript(tr, clap, {})
        assert out and out[0]["check"] == "lyrics_in_silence"

    def test_mock_vision_is_critical(self, qa):
        out = qa.check_vision([], {"provider": "mock", "is_mock": True})
        assert out[0]["severity"] == "critical"

    def test_unusual_tempo_warned(self, qa):
        out = qa.check_music({"tempo_bpm": 64.6, "beats_valid": True}, None)
        assert out and out[0]["check"] == "tempo"

    def test_report_ranks_shots(self, qa, tmp_path):
        rep = qa.build_report(rows=[], coverage={}, route=None, lineage={"has_lineage": True},
                              music_summary={}, transcript=[], clap=[], transcript_stats={},
                              vision_stats={}, repo_root=tmp_path, frames_dir=tmp_path, frame_fps=2)
        assert rep["summary"]["warning"] == 1        # no route file
        assert rep["review_first"] == []


# ---------------------------------------------------------------- frame timing
def test_frames_extracted_with_round_up(tmp_path, monkeypatch):
    """ffmpeg's default fps filter showed t+0.21 s in frame N (round-10
    calibration); round=up gives the frame at (N-1)/fps."""
    p0 = load_phase("phase0_input", tmp_path)
    seen = {}
    monkeypatch.setattr(p0.subprocess, "run", lambda cmd, **k: seen.setdefault("cmd", cmd))
    p0.extract_frames(tmp_path / "v.mp4", tmp_path / "frames", fps=2)
    assert "fps=2:round=up" in seen["cmd"]


# ---------------------------------------------------------------- calibration
class TestCalibrationHelpers:
    @pytest.fixture()
    def cal(self):
        return load_phase("calibration")

    def test_ground_truth_design(self, cal):
        gt = cal.ground_truth(speech=True)
        assert len(gt["cuts_on_beat"]) == 8 and len(gt["cuts_off_beat"]) == 2
        assert gt["beats"][:3] == [0.0, 0.5, 1.0] and len(gt["beats"]) == 40
        assert any(abs(c * 2 - round(c * 2)) > 1e-6 for c in gt["cuts"])   # an off-grid cut
        assert [s["type"] for s in gt["audio_segments"]] == ["music", "speech", "silence"]

    def test_match_events(self, cal):
        m = cal.match_events([1.0, 2.02, 5.0], [1.0, 2.0, 3.0], tol=0.05)
        assert m["tp"] == 2 and m["precision"] == pytest.approx(0.667, abs=1e-3)

    @pytest.mark.skipif(os.environ.get("RUN_CALIBRATION") != "1",
                        reason="end-to-end calibration takes ~4 min; set RUN_CALIBRATION=1")
    def test_end_to_end(self, cal):
        import subprocess, sys
        rc = subprocess.run([sys.executable, str(Path(cal.__file__)), "run"],
                            cwd=str(Path(cal.__file__).parent.parent)).returncode
        assert rc == 0


class TestLineagePortability:
    """Round 13: moving the project or deleting the raw video must not make
    every step 'stale'."""

    def test_missing_input_is_not_stale(self, tmp_path):
        lin = load_phase("_lineage")
        (tmp_path / "in.mp4").write_text("v")
        (tmp_path / "out.txt").write_text("o")
        lin.record(tmp_path, "s", [tmp_path / "in.mp4"], [tmp_path / "out.txt"])
        (tmp_path / "in.mp4").unlink()
        res = lin.check(tmp_path)
        assert res["problems"] == [] and res["missing"] == [{"step": "s", "file": "in.mp4"}]

    def test_repo_files_stored_relative_to_repo(self, tmp_path):
        lin = load_phase("_lineage")
        key = lin._key(lin.REPO_ROOT / "data" / "raw" / "x.mp4", tmp_path)
        assert key == "@repo/data/raw/x.mp4"
        assert lin._resolve(key, tmp_path) == lin.REPO_ROOT / "data" / "raw" / "x.mp4"


def test_speech_override_on_music_is_warned():
    rc = load_phase("route_content")
    r = rc.decide({"music": 0.57, "speech": 0.29, "silence": 0.14}, "speech")
    assert r["has_music"] is False and any(x.startswith("WARNING") for x in r["reasons"])
