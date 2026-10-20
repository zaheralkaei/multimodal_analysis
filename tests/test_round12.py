"""Round 12: pre-spend audit — edge-case inputs found by running odd videos."""
import json
from pathlib import Path

import pytest

from conftest import load_phase


def _static_video(path: Path, seconds=3, fps=24, size=(160, 90)):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    rng = np.random.default_rng(0)
    img = (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
    for _ in range(seconds * fps):
        vw.write(img)
    vw.release()
    return path


def test_video_without_cuts_is_one_shot(tmp_path):
    """PySceneDetect returns [] when there is no cut: a single-take speech got
    ZERO shots and no per-shot analysis."""
    p1 = load_phase("phase1_shots", tmp_path)
    video = _static_video(tmp_path / "v.avi")
    (tmp_path / "frames").mkdir()
    for n in range(1, 7):
        (tmp_path / "frames" / f"frame_{n:05d}.jpg").write_bytes(b"x")
    shots, fps, _ = p1.detect_shots(video, 35.0, 15, 2, tmp_path / "frames", 6)
    assert len(shots) == 1
    assert shots[0]["start_sec"] == 0.0 and shots[0]["end_sec"] == pytest.approx(3.0, abs=0.05)


def test_has_audio_stream_and_silence(tmp_path):
    p0 = load_phase("phase0_input", tmp_path)
    import soundfile as sf
    video = _static_video(tmp_path / "v.avi")
    assert p0.has_audio_stream(video) is False
    p0.write_silence(tmp_path / "a.wav", 2.5, 16000)
    info = sf.info(str(tmp_path / "a.wav"))
    assert info.samplerate == 16000 and info.frames == 40000


def test_route_flags_missing_audio(tmp_path, monkeypatch):
    rc = load_phase("route_content", tmp_path)
    import numpy as np
    import soundfile as sf
    sf.write(str(tmp_path / "audio_music.wav"), np.zeros(48000 * 6, np.float32), 48000)
    (tmp_path / "metadata.json").write_text(json.dumps({"has_audio": False}))
    monkeypatch.setattr("sys.argv", ["route_content.py", "--no-clap"])
    import io
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):
        assert rc.main() == 0
    route = json.loads((tmp_path / "content_route.json").read_text())
    assert route["has_music"] is False and "no audio track" in route["reasons"][0]


def test_language_warning_only_when_words_exist():
    qa = load_phase("_qa")
    assert qa.check_transcript([], [], {"language": "nn", "language_probability": 0.2}) == []
    tr = [{"start_sec": "0", "end_sec": "1", "text": "hej", "suspect": "False"}]
    out = qa.check_transcript(tr, [], {"language": "nn", "language_probability": 0.2})
    assert [i["check"] for i in out] == ["language"]


def test_interrupted_rerun_keeps_previous_paid_answers(tmp_path, monkeypatch):
    """A re-run that is interrupted (crash, Ctrl-C) must not lose answers for
    shots it had not reached yet — they were paid for in the earlier run."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    import csv
    p2 = load_phase("phase2_vision", tmp_path)
    monkeypatch.setattr(p2, "REPO_ROOT", tmp_path)
    frames = tmp_path / "data" / "vid" / "frames"
    frames.mkdir(parents=True)
    shots = []
    for i in range(6):
        cv2.imwrite(str(frames / f"frame_{i + 1:05d}.jpg"), np.full((40, 60, 3), 30 * i, np.uint8))
        shots.append({"shot_idx": i, "start_sec": float(i), "end_sec": float(i + 1), "duration_sec": 1.0,
                      "mid_frame_path": f"data/vid/frames/frame_{i + 1:05d}.jpg"})
    out = tmp_path / "v.csv"
    monkeypatch.setattr(p2, "call_llm", lambda *a, **k: ('{"caption": "paid A"}', 0.0))
    p2.analyze_shots("openrouter", "model-A", shots, tmp_path, out)
    calls = []

    def interrupted(*a, **k):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return '{"caption": "paid B"}', 0.0
    monkeypatch.setattr(p2, "call_llm", interrupted)
    with pytest.raises(KeyboardInterrupt):
        p2.analyze_shots("openrouter", "model-B", shots, tmp_path, out)   # model change -> re-run
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 6                                          # nothing lost
    assert [r["caption"] for r in rows] == ["paid B", "paid B"] + ["paid A"] * 4


def test_repeated_unparseable_answers_stop_paid_calls(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    p2 = load_phase("phase2_vision", tmp_path)
    monkeypatch.setattr(p2, "REPO_ROOT", tmp_path)
    frames = tmp_path / "data" / "vid" / "frames"
    frames.mkdir(parents=True)
    shots = []
    for i in range(9):
        cv2.imwrite(str(frames / f"frame_{i + 1:05d}.jpg"), np.full((40, 60, 3), 20 * i, np.uint8))
        shots.append({"shot_idx": i, "start_sec": float(i), "end_sec": float(i + 1), "duration_sec": 1.0,
                      "mid_frame_path": f"data/vid/frames/frame_{i + 1:05d}.jpg"})
    calls = []
    monkeypatch.setattr(p2, "call_llm", lambda *a, **k: calls.append(1) or ("", 0.0))
    rows, stats = p2.analyze_shots("openrouter", "m", shots, tmp_path, tmp_path / "v.csv")
    assert len(calls) == p2.MAX_CONSECUTIVE_ERRORS and stats["stopped_after_errors"]
