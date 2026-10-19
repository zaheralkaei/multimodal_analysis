"""Round 8: phase-3 camera estimator — sign conventions, camera-vs-subject
separation, cut guard bands, end-to-end on a synthetic video."""
import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from conftest import load_phase


@pytest.fixture()
def p3():
    return load_phase("phase3_camera")


def _texture(h=180, w=320, seed=0):
    rng = np.random.default_rng(seed)
    img = (rng.random((h, w)) * 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _warp(img, M):
    h, w = img.shape[:2]
    return cv2.warpAffine(img, np.float32(M), (w, h), borderMode=cv2.BORDER_REFLECT)


def _label(p3, img1, img2, dt=0.125, n=3):
    est = p3.estimate_pair(img1, img2)
    assert est is not None
    est.update(p3.pair_rates(est, dt))
    return p3.classify_shot([est] * n), est


class TestSignConventions:
    """The pre-round-8 classifier reported every axis backwards."""

    def test_camera_pan_right_content_moves_left(self, p3):
        img = _texture()
        out, est = _label(p3, img, _warp(img, [[1, 0, -6], [0, 1, 0]]))
        assert out["camera_motion"] == "pan-right"
        assert est["pan_rate"] > 0

    def test_camera_pan_left(self, p3):
        img = _texture()
        out, _ = _label(p3, img, _warp(img, [[1, 0, 6], [0, 1, 0]]))
        assert out["camera_motion"] == "pan-left"

    def test_camera_tilt_down_content_moves_up(self, p3):
        img = _texture()
        out, _ = _label(p3, img, _warp(img, [[1, 0, 0], [0, 1, -6]]))
        assert out["camera_motion"] == "tilt-down"

    def test_camera_tilt_up(self, p3):
        img = _texture()
        out, _ = _label(p3, img, _warp(img, [[1, 0, 0], [0, 1, 6]]))
        assert out["camera_motion"] == "tilt-up"

    def test_zoom_in_content_expands(self, p3):
        img = _texture()
        out, est = _label(p3, img, _warp(img, cv2.getRotationMatrix2D((160, 90), 0, 1.05)))
        assert out["camera_motion"] == "zoom-in"
        assert abs(est["pan_rate"]) < 0.02 and abs(est["tilt_rate"]) < 0.02

    def test_zoom_out(self, p3):
        img = _texture()
        out, _ = _label(p3, img, _warp(img, cv2.getRotationMatrix2D((160, 90), 0, 0.95)))
        assert out["camera_motion"] == "zoom-out"

    def test_confidence_high_for_clear_motion_low_near_threshold(self, p3):
        img = _texture()
        clear, _ = _label(p3, img, _warp(img, [[1, 0, -8], [0, 1, 0]]))       # 0.2 /s = 5x
        weak, _ = _label(p3, img, _warp(img, [[1, 0, -2.0], [0, 1, 0]]))      # 0.05 /s = 1.25x
        assert clear["camera_confidence"] == "high"
        assert weak["camera_motion"] == "pan-right" and weak["camera_confidence"] == "low"

    def test_static(self, p3):
        img = _texture()
        out, _ = _label(p3, img, img.copy())
        assert out["camera_motion"] == "static"

    def test_rate_units_are_frames_per_second(self, p3):
        img = _texture()
        # 8 px of a 320 px frame in 0.125 s = 0.2 frame-widths/s
        _, est = _label(p3, img, _warp(img, [[1, 0, -8], [0, 1, 0]]))
        assert est["pan_rate"] == pytest.approx(0.2, abs=0.01)


class TestCameraVsSubject:
    def test_moving_subject_on_static_background_is_static(self, p3):
        bg = _texture(seed=1)
        subj = _texture(60, 80, seed=2)
        f1, f2 = bg.copy(), bg.copy()
        f1[60:120, 40:120] = subj
        f2[60:120, 52:132] = subj            # subject walks 12 px right
        out, est = _label(p3, f1, f2)
        assert out["camera_motion"] == "static"
        assert out["subject_motion_share"] > 0.02   # the subject's tracks are outliers

    def test_too_few_features_is_unknown(self, p3):
        flat = np.full((180, 320), 128, np.uint8)
        assert p3.estimate_pair(flat, flat) is None
        assert p3.classify_shot([None, None])["camera_motion"] == "unknown"

    def test_majority_bad_pairs_is_unknown(self, p3):
        img = _texture()
        est = p3.estimate_pair(img, img.copy())
        est.update(p3.pair_rates(est, 0.125))
        assert p3.classify_shot([est, None, None, None])["camera_motion"] == "unknown"


class TestGuardBands:
    def test_samples_stay_inside_shot(self, p3):
        ts = p3.sample_times(10.0, 12.0, 8.0, 0.06)
        assert ts and min(ts) >= 10.06 and max(ts) <= 11.94
        assert all(b - a == pytest.approx(0.125) for a, b in zip(ts, ts[1:]))

    def test_tiny_shot_has_no_samples(self, p3):
        assert p3.sample_times(5.0, 5.1, 8.0, 0.06) == []
        assert p3.sample_frame_indices(5.0, 5.1, 24.0, 8.0) == []

    def test_frame_indices_skip_guard_frames(self, p3):
        # shot [2.0, 4.0) at 24 fps = native frames 48..95; guard 2 each side
        idx = p3.sample_frame_indices(2.0, 4.0, 24.0, 8.0)
        assert idx[0] == 50 and idx[-1] <= 93
        assert all(b - a == 3 for a, b in zip(idx, idx[1:]))

    def test_frame_indices_ntsc_rate(self, p3):
        idx = p3.sample_frame_indices(10.0, 11.0, 30000 / 1001, 8.0)
        assert all(10.0 <= i / (30000 / 1001) < 11.0 for i in idx)

    def test_extracted_frame_range_excludes_previous_shot(self, p3):
        first, last = p3.shot_frame_range(26.86, 29.82, 2)
        times = [(i - 1) / 2 for i in range(first, last + 1)]
        assert min(times) >= 26.86 and max(times) < 29.82


class TestEndToEnd:
    def test_synthetic_video_two_shots(self, p3, tmp_path):
        """Shot A (0-2 s): camera pans right. Cut. Shot B (2-4 s): static.
        A frame pair across the cut would read as a huge motion — the guard
        band must keep shot B static."""
        fps, w, h = 24, 320, 180
        big = _texture(h, w * 3, seed=3)
        other = _texture(h, w, seed=4)
        path = tmp_path / "synth.avi"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
        assert vw.isOpened()
        for i in range(2 * fps):                 # pan right: window slides right
            x = i * 2
            vw.write(cv2.cvtColor(big[:, x:x + w], cv2.COLOR_GRAY2BGR))
        for _ in range(2 * fps):
            vw.write(cv2.cvtColor(other, cv2.COLOR_GRAY2BGR))
        vw.release()
        shots = [
            {"start_sec": 0.0, "end_sec": 2.0, "duration_sec": 2.0},
            {"start_sec": 2.0, "end_sec": 4.0, "duration_sec": 2.0},
        ]
        rows = p3.analyze_video(path, shots, flow_fps=8.0)
        assert rows[0]["camera_motion"] == "pan-right"
        # 2 px/frame * 24 fps / 320 px = 0.15 frame-widths per second
        assert rows[0]["pan_rate"] == pytest.approx(0.15, abs=0.02)
        assert rows[1]["camera_motion"] == "static"
        assert rows[0]["n_frames"] >= 10
