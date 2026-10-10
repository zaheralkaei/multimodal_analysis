"""round-4 follow-up: shared _paths.disp survives relative + out-of-repo paths."""
import os
from pathlib import Path


def test_disp_relative_env_path():
    # `PROCESSED_DIR=data/vid` given as a relative path must not crash prints
    from _paths import disp
    out = disp("data/vid/shot_camera.csv")
    assert "shot_camera.csv" in out


def test_disp_outside_repo(tmp_path):
    from _paths import disp
    f = tmp_path / "x" / "y.csv"
    out = disp(f)
    assert out.endswith(str(Path("x") / "y.csv"))


def test_disp_in_repo_is_relative():
    from _paths import disp
    out = disp(Path(__file__))
    assert out == str(Path("tests") / Path(__file__).name)