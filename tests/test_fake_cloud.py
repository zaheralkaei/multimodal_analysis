"""Round 12: the real client code against the local fake cloud (no network).

Positive tests prove each request passes the provider's documented format
checks (the fake server answers 400 otherwise). Fault tests prove a failing
service never yields silent wrong data, never poisons the cache, and never
logs a paid result that did not happen.
"""
import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from conftest import load_phase

SPEECH = {"start": 2.0, "end": 5.0, "text": "hello calibration world"}


def _wav(path: Path, seconds=6.0, sr=48000):
    import numpy as np
    import soundfile as sf
    t = np.arange(int(seconds * sr)) / sr
    sf.write(str(path), (0.2 * np.sin(2 * np.pi * 330 * t)).astype("float32"), sr)
    return path


@pytest.fixture()
def cloud_env(tmp_path, monkeypatch):
    """Start a fake cloud with the faults given via request.param-like helper."""
    import fake_cloud
    servers = []

    def start(faults=()):
        srv, base, state, env = fake_cloud.start(0, set(faults), vocals_silence_before=1.0, speech=SPEECH)
        servers.append(srv)
        env["CLOUD_CACHE_DIR"] = str(tmp_path / "cache")
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        import _cloud
        monkeypatch.setattr(_cloud, "CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr(_cloud.time, "sleep", lambda s: None)
        return state
    yield start
    for s in servers:
        s.shutdown()


def _usage(tmp_path):
    p = tmp_path / "cloud_usage.json"
    return json.loads(p.read_text())["calls"] if p.exists() else []


# ---------------------------------------------------------------- fal
class TestFal:
    def _sep(self, tmp_path, monkeypatch):
        sep = load_phase("separate_vocals", tmp_path)
        monkeypatch.setattr(sep.time, "sleep", lambda s: None)
        _wav(tmp_path / "audio_music.wav")
        return sep

    def test_request_accepted_and_stem_downloaded(self, tmp_path, monkeypatch, cloud_env):
        state = cloud_env()
        sep = self._sep(tmp_path, monkeypatch)
        info = sep.separate_fal(tmp_path / "audio_music.wav", tmp_path / "stem.wav", tmp_path)
        assert state.log == [] and (tmp_path / "stem.wav").stat().st_size > 1000
        assert not info["cached"] and _usage(tmp_path)[0]["est_usd"] > 0

    def test_completed_with_error_raises_and_caches_nothing(self, tmp_path, monkeypatch, cloud_env):
        cloud_env({"fal_error"})
        sep = self._sep(tmp_path, monkeypatch)
        with pytest.raises(sep._cloud.CloudError, match="failed"):
            sep.separate_fal(tmp_path / "audio_music.wav", tmp_path / "stem.wav", tmp_path)
        assert not (tmp_path / "cache" / "fal_demucs").exists() or \
            not any((tmp_path / "cache" / "fal_demucs").iterdir())
        assert _usage(tmp_path) == []

    def test_main_failure_stops_with_nonzero_exit(self, tmp_path, monkeypatch, cloud_env):
        cloud_env({"fal_error"})
        sep = self._sep(tmp_path, monkeypatch)
        monkeypatch.setattr("sys.argv", ["separate_vocals.py", "--provider", "fal"])
        with redirect_stdout(io.StringIO()):
            assert sep.main() == 1
        assert not (tmp_path / "vocals.wav").exists()


# ---------------------------------------------------------------- Groq
class TestGroq:
    def test_request_accepted_words_and_cache(self, tmp_path, cloud_env):
        state = cloud_env()
        p4 = load_phase("phase4_transcribe", tmp_path)
        audio = _wav(tmp_path / "vocals.wav", sr=16000)
        rows = p4.transcribe_groq(audio, None, tmp_path)
        assert state.log == []
        assert rows[0]["text"] == "hello calibration world" and len(rows[0]["words"]) == 3
        n = state.counts.get("groq_500", 0)
        p4.transcribe_groq(audio, None, tmp_path)          # cached: the server is not called
        assert [c["cached"] for c in _usage(tmp_path)] == [False, True]

    def test_server_error_is_retried(self, tmp_path, cloud_env):
        state = cloud_env({"groq_500_once"})
        p4 = load_phase("phase4_transcribe", tmp_path)
        rows = p4.transcribe_groq(_wav(tmp_path / "v.wav", sr=16000), None, tmp_path)
        assert rows and state.counts["groq_500"] == 2


# ---------------------------------------------------------------- Replicate
class TestReplicate:
    def test_flow_accepted(self, tmp_path, cloud_env):
        state = cloud_env()
        ms = load_phase("music_structure", tmp_path)
        out = ms.replicate_allin1(_wav(tmp_path / "audio_music.wav"), tmp_path)
        assert state.log == [] and out["segments"][1]["label"] == "chorus"

    def test_failed_prediction_falls_back_and_is_reported(self, tmp_path, monkeypatch, cloud_env):
        cloud_env({"replicate_failed"})
        ms = load_phase("music_structure", tmp_path)
        _wav(tmp_path / "audio_music.wav")
        monkeypatch.setattr(ms, "beats_beat_this", lambda a: ([i * 0.5 for i in range(12)], [0.0, 2.0, 4.0]))
        monkeypatch.setattr(ms, "sections_local", lambda *a: [{"start": 0.0, "end": 5.5, "label": "A"}])
        monkeypatch.setattr("sys.argv", ["music_structure.py", "--beats", "beat_this", "--sections", "replicate"])
        with redirect_stdout(io.StringIO()):
            assert ms.main() == 0
        st = json.loads((tmp_path / "music_structure.json").read_text())
        assert st["sections_source"] == "local" and "failed" in st["fallback_reason"]
        qa = load_phase("_qa")
        assert any(i["check"] == "structure_fallback" for i in qa.check_audio_stack({}, st, None))
        assert _usage(tmp_path) == []


# ---------------------------------------------------------------- OpenRouter
class TestOpenRouter:
    def _shots(self, base: Path):
        cv2 = pytest.importorskip("cv2")
        import numpy as np
        frames = base / "data" / "vid" / "frames"
        frames.mkdir(parents=True)
        shots = []
        for i in range(2):
            cv2.imwrite(str(frames / f"frame_{i + 1:05d}.jpg"), np.full((90, 160, 3), 40 * (i + 1), np.uint8))
            shots.append({"shot_idx": i, "start_sec": 2.0 * i, "end_sec": 2.0 * i + 2, "duration_sec": 2.0,
                          "mid_frame_path": f"data/vid/frames/frame_{i + 1:05d}.jpg"})
        return shots

    def test_analyze_and_rate_limit_retry(self, tmp_path, monkeypatch, cloud_env):
        state = cloud_env({"openrouter_429_once"})
        p2 = load_phase("phase2_vision", tmp_path)
        monkeypatch.setattr(p2, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(p2.time, "sleep", lambda s: None)
        rows, stats = p2.analyze_shots("openrouter", "google/gemma-4-31b-it", self._shots(tmp_path),
                                       tmp_path, tmp_path / "v.csv")
        assert state.log == [] and stats["errors"] == 0 and stats["parse_errors"] == 0
        assert all(r["status"] == "ok" and r["caption"].startswith("FAKE-CLOUD") for r in rows)
        assert p2.USAGE["retries"] >= 1

    def test_garbage_answer_is_parse_error_and_retried_next_run(self, tmp_path, monkeypatch, cloud_env):
        cloud_env({"openrouter_garbage"})
        p2 = load_phase("phase2_vision", tmp_path)
        monkeypatch.setattr(p2, "REPO_ROOT", tmp_path)
        shots = self._shots(tmp_path)
        rows, stats = p2.analyze_shots("openrouter", "google/gemma-4-31b-it", shots, tmp_path, tmp_path / "v.csv")
        assert stats["parse_errors"] == 2 and all(r["status"] == "parse_error" for r in rows)
        assert p2._count_pending(shots, tmp_path / "v.csv", "google/gemma-4-31b-it") == 2

    def test_rows_never_reused_across_endpoints(self, tmp_path, monkeypatch, cloud_env):
        cloud_env()
        p2 = load_phase("phase2_vision", tmp_path)
        monkeypatch.setattr(p2, "REPO_ROOT", tmp_path)
        shots = self._shots(tmp_path)
        p2.analyze_shots("openrouter", "google/gemma-4-31b-it", shots, tmp_path, tmp_path / "v.csv")
        row = next(csv.DictReader((tmp_path / "v.csv").open()))
        assert row["endpoint"].startswith("http://127.0.0.1")
        assert not p2._reusable(row, row["mid_frame"], "google/gemma-4-31b-it", row["mid_frame_sha"],
                                "https://openrouter.ai/api/v1")


# ---------------------------------------------------------------- round-12 safeguards
def test_phase2_stops_after_consecutive_errors(tmp_path, monkeypatch):
    p2 = load_phase("phase2_vision", tmp_path)
    monkeypatch.setattr(p2, "REPO_ROOT", tmp_path)
    shots = TestOpenRouter()._shots(tmp_path) * 4                     # 8 shots
    shots = [dict(s, shot_idx=i) for i, s in enumerate(shots)]
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise RuntimeError("HTTP 401: invalid key")
    monkeypatch.setattr(p2, "call_llm", boom)
    rows, stats = p2.analyze_shots("openrouter", "m", shots, tmp_path, tmp_path / "v.csv")
    assert len(calls) == p2.MAX_CONSECUTIVE_ERRORS and stats["stopped_after_errors"]
    assert len(rows) == 8 and all(r["status"] == "error" for r in rows)


def test_phase2_no_health_check_when_everything_answered(tmp_path, monkeypatch):
    p2 = load_phase("phase2_vision", tmp_path)
    (tmp_path / "shots.json").write_text("[]")
    monkeypatch.setattr(p2, "PROCESSED", tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(p2, "fetch_openrouter_pricing", lambda m: {"prompt": "0", "completion": "0"})
    monkeypatch.setattr(p2, "call_llm", lambda *a, **k: pytest.fail("no call expected"))
    monkeypatch.setattr("sys.argv", ["phase2_vision.py", "--provider", "openrouter"])
    with redirect_stdout(io.StringIO()):
        assert p2.main() == 0


def test_groq_bad_response_is_not_cached(tmp_path, monkeypatch):
    p4 = load_phase("phase4_transcribe", tmp_path)
    import _cloud
    monkeypatch.setattr(_cloud, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(_cloud, "get_key", lambda n: "gsk")
    monkeypatch.setattr(_cloud, "request", lambda *a, **k: {"error": "model overloaded"})
    with pytest.raises(_cloud.CloudError, match="no segments"):
        p4.transcribe_groq(_wav(tmp_path / "v.wav", sr=16000), None, tmp_path)
    assert not (tmp_path / "cache" / "groq").exists() or not any((tmp_path / "cache" / "groq").iterdir())


def test_groq_cached_rerun_allowed_at_budget_cap(tmp_path, monkeypatch, cloud_env):
    cloud_env()
    p4 = load_phase("phase4_transcribe", tmp_path)
    audio = _wav(tmp_path / "v.wav", sr=16000)
    p4.transcribe_groq(audio, None, tmp_path)
    monkeypatch.setenv("AUDIO_MAX_USD", "0.0000001")                  # cap already exceeded
    assert p4.transcribe_groq(audio, None, tmp_path)                   # cached: allowed, free
