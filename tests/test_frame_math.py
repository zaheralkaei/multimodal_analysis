"""Round-4 audit fixes F1a/F1b: fps-aware frame-index math (phases 1 & 3)."""
import os
from pathlib import Path

import pytest

from conftest import load_phase


@pytest.fixture()
def phase1():
    return load_phase("phase1_shots")


@pytest.fixture()
def phase3():
    return load_phase("phase3_camera")


class TestMidFrameIndex:
    def test_two_fps_picks_second_inside_shot(self, phase1):
        # AUDIT_R4 regression: old code gave int(1.418)+1 = 2 (frame at t=0.5s);
        # correct frame for mid=1.418s at 2fps is idx 4 (t=1.5s).
        assert phase1.mid_frame_index(1.418, 2, 100) == 4

    def test_two_fps_late_shot(self, phase1):
        # Old code picked frame at half the timestamp — regression test.
        assert phase1.mid_frame_index(59.02, 2, 1000) == int(round(59.02 * 2)) + 1  # 119

    def test_one_fps_unchanged(self, phase1):
        # At 1 fps the old formula and the new one agree.
        assert phase1.mid_frame_index(10.4, 1, 100) == 11

    def test_clamped_high(self, phase1):
        assert phase1.mid_frame_index(999.0, 2, 100) == 100

    def test_clamped_low(self, phase1):
        assert phase1.mid_frame_index(0.0, 2, 100) == 1


class TestNearestExistingFrame:
    def test_exact_hit(self, phase1, tmp_path):
        (tmp_path / "frame_00004.jpg").write_bytes(b"x")
        assert phase1.nearest_existing_frame(4, 2, tmp_path) == tmp_path / "frame_00004.jpg"

    def test_steps_outward_when_missing(self, phase1, tmp_path):
        (tmp_path / "frame_00003.jpg").write_bytes(b"x")
        got = phase1.nearest_existing_frame(4, 2, tmp_path)
        assert got is not None and got.name == "frame_00003.jpg"

    def test_none_when_dir_empty(self, phase1, tmp_path):
        assert phase1.nearest_existing_frame(4, 2, tmp_path) is None


class TestShotFrameRange:
    def test_two_fps_covers_full_shot(self, phase3):
        # AUDIT_R4 regression: shot 10 of the Tyla video is 26.86–29.82s; the
        # old code read frame FILES 27..30 (t=13.0–14.5s at 2fps) — wrong
        # footage. New math must map seconds -> file indices.
        start_f, end_f = phase3.shot_frame_range(26.86, 29.82, 2)
        times = [(i - 1) / 2 for i in range(start_f, end_f + 1)]
        assert min(times) < 26.86          # starts at/before the cut
        assert max(times) < 29.82          # ends before the next cut
        assert max(times) >= 26.86         # inside the shot
        # At 2 fps a ~3s shot must yield ~6 frames, not 4 files at half-time
        assert end_f - start_f + 1 >= 5

    def test_one_fps_matches_seconds_convention(self, phase3):
        # At 1 fps frame N is at t=N-1: shot [10,15) must read files 11..15
        # (t=10..14). The OLD code read 11..16, which included the first
        # frame of the NEXT shot (t=15).
        start_f, end_f = phase3.shot_frame_range(10.0, 15.0, 1)
        assert (start_f, end_f) == (11, 15)

    def test_very_short_shot_yields_at_least_one(self, phase3):
        start_f, end_f = phase3.shot_frame_range(5.0, 5.2, 2)
        assert end_f >= start_f

    def test_first_shot_starts_at_frame_1(self, phase3):
        start_f, end_f = phase3.shot_frame_range(0.0, 2.8, 2)
        assert start_f == 1