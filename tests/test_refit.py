"""Round-4 audit F1a data repair: _refit_midframes recomputes mid-frames in place."""
import json
import sys
from pathlib import Path

import pytest

from conftest import load_phase


@pytest.fixture()
def refit(tmp_processed, fake_metadata):
    mod = load_phase("_refit_midframes", tmp_processed, module_name="_refit_midframes")
    # fake_metadata gives tmp_processed = data/vid1 with 100 frames at 2 fps.
    # buggy old math picked frame index 2 for shot 0 (drift error);
    # correct math: idx = round(1.25*2)+1 = 3 (t=1.0s, inside 0–2.5s shot)
    frames_dir = tmp_processed / "frames"
    shots = [
        {"shot_idx": 0, "start_sec": 0.0, "end_sec": 2.5, "mid_sec": 1.25,
         "mid_frame_path": "data/vid1/frames/frame_00002.jpg", "n_frames": 4},
        {"shot_idx": 1, "start_sec": 2.5, "end_sec": 7.5, "mid_sec": 5.0,
         "mid_frame_path": "data/vid1/frames/frame_00005.jpg", "n_frames": 10},
    ]
    (tmp_processed / "shots.json").write_text(json.dumps(shots), encoding="utf-8")
    return mod, tmp_processed, frames_dir


def test_refit_recomputes_mid_frames(refit):
    mod, tmp_processed, frames_dir = refit
    out = mod.refit_video(tmp_processed)
    assert out["changed_mid_frames"] == 2
    shots = json.loads((tmp_processed / "shots.json").read_text(encoding="utf-8"))
    # frame t=(N-1)/fps -> shot[0] mid 1.25s = frame 3; shot[1] mid 5.0s = frame 11
    # (tmp PROCESSED_DIR is outside the repo, so paths come back absolute)
    assert shots[0]["mid_frame_path"].endswith("frame_00003.jpg")
    assert shots[1]["mid_frame_path"].endswith("frame_00011.jpg")
    assert shots[0]["n_frames"] == 5  # round(2.5 * 2)


def test_refit_propagates_to_csvs(refit):
    mod, tmp_processed, _ = refit
    (tmp_processed / "shot_vision.csv").write_text(
        "shot_idx,mid_frame,caption\n"
        '0,data/vid1/frames/frame_00002.jpg,"old"\n'
        '1,data/vid1/frames/frame_00005.jpg,"old"\n',
        encoding="utf-8")
    mod.refit_video(tmp_processed)
    rows = (tmp_processed / "shot_vision.csv").read_text(encoding="utf-8").strip().splitlines()
    assert "frame_00003.jpg" in rows[1]
    assert "frame_00011.jpg" in rows[2]