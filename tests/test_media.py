"""Tests on synthetic video/audio with known ground truth (need ffmpeg, OpenCV, librosa)."""
from __future__ import annotations
import pytest

from conftest import MOTIONS, click_track, extract_frames, make_clip, needs_ffmpeg, textured_still

pytestmark = needs_ffmpeg


@pytest.fixture(scope="module")
def still(tmp_path_factory):
    return textured_still(tmp_path_factory.mktemp("still") / "still.png", seed=0)


@pytest.mark.parametrize("motion", list(MOTIONS))
def test_camera_motion_classification(motion, still, tmp_path):
    import phase3_camera as p3
    clip = make_clip(still, motion, 4, tmp_path / "clip.mp4")
    frames = extract_frames(clip, tmp_path / "frames", fps=2)
    row = p3.analyze_shots([{"start_sec": 0, "end_sec": 4, "duration_sec": 4}], frames, 2.0)[0]
    assert row["camera_motion"] == motion, row
    assert row["n_frames"] == 8


def test_camera_speeds_are_physical(still, tmp_path):
    """pan of 60 px/s on a 320 px wide frame = 0.1875 frame widths per second."""
    import phase3_camera as p3
    frames = extract_frames(make_clip(still, "pan-right", 4, tmp_path / "c.mp4"), tmp_path / "f", fps=2)
    row = p3.analyze_shots([{"start_sec": 0, "end_sec": 4, "duration_sec": 4}], frames, 2.0)[0]
    assert row["pan_score_mean"] == pytest.approx(60 / 320, rel=0.05)
    assert abs(row["tilt_score_mean"]) < 0.01


def test_flat_still_frames_are_static(tmp_path):
    """No texture → no features; identical frames must still read as static, not unknown."""
    import cv2
    import numpy as np
    import phase3_camera as p3
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(1, 5):
        cv2.imwrite(str(d / f"frame_{i:05d}.jpg"), np.full((240, 320, 3), 128, np.uint8))
    row = p3.analyze_shots([{"start_sec": 0, "end_sec": 2, "duration_sec": 2}], d, 2.0)[0]
    assert row["camera_motion"] == "static"


def test_tempo_and_beats(tmp_path):
    import numpy as np
    import phase6_music as p6
    rows, summary = p6.analyze_music(click_track(tmp_path / "a.wav", 20, bpm=120))
    assert summary["tempo_bpm"] == pytest.approx(120, rel=0.05)
    gaps = np.diff(summary["beat_times"])
    assert np.median(gaps) == pytest.approx(0.5, abs=0.02)
    assert len(rows) == 20 and {"rms_energy", "onset_strength", "n_beats"} <= rows[0].keys()
    assert sum(r["n_beats"] for r in rows) == len(summary["beat_times"])


def test_shot_detection_on_synthetic_video(smoke_video):
    import phase1_shots as p1
    from conftest import SMOKE_SHOT_SEC, SMOKE_SHOTS
    dets = p1.build_detectors("adaptive", 35, 3.0, 30, fades=True)
    shots, fps, total = p1.detect_shots(smoke_video, dets, frame_fps=2)
    assert len(shots) == len(SMOKE_SHOTS)
    for i, s in enumerate(shots):
        assert s["start_sec"] == pytest.approx(i * SMOKE_SHOT_SEC, abs=0.05)
        assert len(s["key_frame_paths"]) == 3
