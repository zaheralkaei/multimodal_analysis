"""End-to-end: run_pipeline.py on the synthetic video, without the model phases.

This is the test that would have caught the round-3 bugs: thumbnails from the
wrong shot (frame-rate mix-up), the wrong video being analysed, a crash in
phase 6 and a failure reported as success.
"""
from __future__ import annotations
import csv, json, os, shutil, subprocess, sys
from pathlib import Path

import pytest

from conftest import SMOKE_SHOTS, needs_ffmpeg

pytestmark = needs_ffmpeg
REPO = Path(__file__).resolve().parent.parent
VIDEO_ID = "pytest_smoke"


def run_pipeline(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts" / "run_pipeline.py"), *args],
                          cwd=REPO, capture_output=True, text=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})


@pytest.fixture(scope="module")
def pipeline_run(smoke_video):
    data, reports = REPO / "data" / VIDEO_ID, REPO / "reports" / VIDEO_ID
    for d in (data, reports):
        shutil.rmtree(d, ignore_errors=True)
    first = run_pipeline(str(smoke_video), "--id", VIDEO_ID, "--skip", "2,4,5")
    second = run_pipeline(str(smoke_video), "--id", VIDEO_ID, "--skip", "2,4,5")
    yield first, second, data, reports
    for d in (data, reports):
        shutil.rmtree(d, ignore_errors=True)


def test_pipeline_succeeds(pipeline_run):
    first, *_ = pipeline_run
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]


def test_shots_and_thumbnails_match(pipeline_run):
    import cv2
    _, _, data, _ = pipeline_run
    shots = json.loads((data / "shots.json").read_text())
    assert len(shots) == len(SMOKE_SHOTS)
    for shot, (channel, _) in zip(shots, SMOKE_SHOTS):
        img = cv2.imread(str(REPO / shot["mid_frame_path"]))
        assert img is not None
        means = img.reshape(-1, 3).mean(axis=0)
        assert means.argmax() == channel, (shot, means)  # thumbnail comes from its own shot


def test_camera_labels(pipeline_run):
    _, _, data, _ = pipeline_run
    with (data / "shot_camera.csv").open() as f:
        labels = [r["camera_motion"] for r in csv.DictReader(f)]
    assert labels == [m for _, m in SMOKE_SHOTS]


def test_cuts_on_beat_detected(pipeline_run):
    _, _, data, _ = pipeline_run
    stats = json.loads((data / "sync_stats.json").read_text())
    head = stats["beat_alignment"][0]
    assert stats["total_cuts"] == len(SMOKE_SHOTS) - 1
    assert head["observed_pct"] == 100.0
    assert head["p_value"] < 0.05


def test_dashboard_is_self_contained(pipeline_run):
    _, _, _, reports = pipeline_run
    html = (reports / "dashboard.html").read_text()
    assert html.count("data:image/jpeg;base64,") == len(SMOKE_SHOTS)
    assert "src='frames/" not in html
    assert "Cuts on a beat" in html and "Provenance" in html


def test_second_run_skips_everything(pipeline_run):
    _, second, _, _ = pipeline_run
    assert second.returncode == 0
    assert second.stdout.count("(up to date)") == 6  # phases 0,1,3,6,7,8


def test_missing_file_fails_loudly():
    res = run_pipeline("/definitely/not/here.mp4")
    assert res.returncode != 0


def test_labeling_and_validation_flow(pipeline_run):
    """label_shots → (simulated human) → validate_labels → dashboard shows the result."""
    _, _, data, reports = pipeline_run
    env = {**os.environ, "PROCESSED_DIR": str(data), "REPORTS_DIR": str(reports)}
    scripts = REPO / "scripts"
    res = subprocess.run([sys.executable, str(scripts / "label_shots.py"), "--n", "5"], env=env,
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    page = (reports / "labeling.html").read_text()
    assert page.count("class='card'") == 5 and "Download CSV" in page
    # A "human" who agrees with the true motions (pipeline output for camera is exact on this video)
    with (data / "human_labels.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shot_idx", "emotion", "camera", "notes"])
        for i, (_, motion) in enumerate(SMOKE_SHOTS):
            w.writerow([i, "neutral", motion if i != 1 else "unsure", ""])
    res = subprocess.run([sys.executable, str(scripts / "validate_labels.py")], env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    result = {r["comparison"]: r for r in json.loads((data / "validation.json").read_text())["results"]}
    exact = result["camera: optical flow vs human (exact)"]
    assert exact["n"] == len(SMOKE_SHOTS) - 1 and exact["accuracy"] == 1.0
    subprocess.run([sys.executable, str(scripts / "phase8_dashboard.py")], env=env, check=True, capture_output=True)
    assert "How accurate are these labels?" in (reports / "dashboard.html").read_text()
    assert "100% (" in (reports / "dashboard.html").read_text()


def test_compare_videos(pipeline_run, smoke_video, tmp_path):
    _, _, data, _ = pipeline_run
    other = REPO / "data" / f"{VIDEO_ID}_b"
    try:
        res = run_pipeline(str(smoke_video), "--id", f"{VIDEO_ID}_b", "--skip", "2,4,5", "--fps", "1")
        assert res.returncode == 0, res.stdout[-2000:]
        out = tmp_path / "comparison.html"
        res = subprocess.run([sys.executable, str(REPO / "scripts" / "compare_videos.py"), VIDEO_ID, f"{VIDEO_ID}_b",
                              "--out", str(out)], cwd=REPO, capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        assert "frame_fps" in res.stdout  # different fps is flagged as a comparability problem
        rows = list(csv.DictReader(out.with_suffix(".csv").open()))
        assert [r["n_shots"] for r in rows] == [str(len(SMOKE_SHOTS))] * 2
        assert "Comparison of 2 videos" in out.read_text()
    finally:
        shutil.rmtree(other, ignore_errors=True)
        shutil.rmtree(REPO / "reports" / f"{VIDEO_ID}_b", ignore_errors=True)
