"""Round 11: cloud audio stack — shared cloud layer, fal Demucs separation,
Groq transcription, Beat This!/All-In-One structure, downbeat + section
statistics, runner wiring. All HTTP is mocked: no network, no cost."""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from conftest import load_phase


def _wav(path: Path, seconds: float = 3.0, sr: int = 48000, amp: float = 0.1, freq: float = 440.0):
    import numpy as np
    import soundfile as sf
    t = np.arange(int(seconds * sr)) / sr
    sf.write(str(path), (amp * np.sin(2 * np.pi * freq * t)).astype("float32"), sr)
    return path


@pytest.fixture()
def cloud(tmp_path, monkeypatch):
    mod = load_phase("_cloud")
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.delenv("AUDIO_MAX_USD", raising=False)
    return mod


# ---------------------------------------------------------------- _cloud
class TestCloudLayer:
    def test_cache_key_depends_on_content_and_params(self, cloud):
        a = cloud.cache_key("svc", "sha1", "model-x")
        assert a == cloud.cache_key("svc", "sha1", "model-x")
        assert a != cloud.cache_key("svc", "sha2", "model-x")
        assert a != cloud.cache_key("svc", "sha1", "model-y")

    def test_usage_log_and_budget(self, cloud, tmp_path, monkeypatch):
        cloud.log_usage(tmp_path, "fal", "demucs", 0.6, cached=False)
        cloud.log_usage(tmp_path, "fal", "demucs", 0.6, cached=True)     # cache hit is free
        assert cloud.spent(tmp_path) == pytest.approx(0.6)
        ok, _ = cloud.budget_ok(tmp_path, 0.3)
        assert ok
        ok, why = cloud.budget_ok(tmp_path, 0.5)
        assert not ok and "AUDIO_MAX_USD" in why
        monkeypatch.setenv("AUDIO_MAX_USD", "5")
        assert cloud.budget_ok(tmp_path, 0.5)[0]

    def test_request_retries_then_succeeds(self, cloud, monkeypatch):
        import requests
        calls = []

        class R:
            def __init__(self, code):
                self.status_code, self.headers, self.text = code, {}, "x"

            def json(self):
                return {"ok": True}
        monkeypatch.setattr(cloud.time, "sleep", lambda s: None)

        def fake(*a, **k):
            calls.append(1)
            return R(503 if len(calls) < 3 else 200)
        monkeypatch.setattr(requests, "request", fake)
        assert cloud.request("GET", "https://x") == {"ok": True} and len(calls) == 3

    def test_request_error_has_no_secrets(self, cloud, monkeypatch):
        import requests

        class R:
            status_code, headers, text = 401, {}, "bad key"
        monkeypatch.setattr(requests, "request", lambda *a, **k: R())
        with pytest.raises(cloud.CloudError) as e:
            cloud.request("POST", "https://api.x/y?token=SECRET", headers={"Authorization": "Bearer SECRET"})
        assert "SECRET" not in str(e.value) and "HTTP 401" in str(e.value)


# ---------------------------------------------------------------- separation
@pytest.fixture()
def sep(tmp_path, cloud, monkeypatch):
    mod = load_phase("separate_vocals", tmp_path)
    monkeypatch.setattr(mod, "_cloud", cloud)
    _wav(tmp_path / "audio_music.wav")
    _wav(tmp_path / "audio.wav", sr=16000)
    return mod


class TestSeparation:
    def test_auto_provider(self, sep, monkeypatch):
        monkeypatch.delenv("FAL_KEY", raising=False)
        monkeypatch.setattr(sep._cloud, "get_key", lambda n: None)
        assert sep.resolve_provider("auto", {"has_music": True})[0] == "off"
        monkeypatch.setattr(sep._cloud, "get_key", lambda n: "k")
        assert sep.resolve_provider("auto", {"has_music": True})[0] == "fal"
        assert sep.resolve_provider("auto", {"has_music": False})[0] == "off"
        assert sep.resolve_provider("passthrough", None)[0] == "passthrough"

    def test_fal_flow_cache_and_cost(self, sep, tmp_path, monkeypatch):
        monkeypatch.setattr(sep._cloud, "get_key", lambda n: "fal-test")
        monkeypatch.setattr(sep._cloud, "encode_audio",
                            lambda src, dest, fmt, **k: dest.write_bytes(b"mp3bytes") or dest)
        monkeypatch.setattr(sep.time, "sleep", lambda s: None)
        seen = []

        def fake_request(method, url, **k):
            seen.append((method, url, k.get("json_body")))
            if method == "POST":
                return {"request_id": "r1", "status_url": "https://q/status", "response_url": "https://q/res"}
            if url.endswith("status"):
                return {"status": "COMPLETED"}
            return {"vocals": {"url": "https://cdn/vocals.wav"}}
        monkeypatch.setattr(sep._cloud, "request", fake_request)
        monkeypatch.setattr(sep._cloud, "download",
                            lambda url, dest, **k: _wav(dest, seconds=3.0, sr=44100))
        info = sep.separate_fal(tmp_path / "audio_music.wav", tmp_path / "stem.wav", tmp_path)
        body = seen[0][2]
        assert seen[0][1] == "https://queue.fal.run/fal-ai/demucs"
        assert body["audio_url"].startswith("data:audio/mpeg;base64,")
        assert body["stems"] == ["vocals"] and body["model"] == "htdemucs_ft"
        assert info["est_usd"] == pytest.approx(3.0 * 0.0007, rel=1e-3) and not info["cached"]
        # second call: served from the cache, no HTTP
        seen.clear()
        info2 = sep.separate_fal(tmp_path / "audio_music.wav", tmp_path / "stem2.wav", tmp_path)
        assert info2["cached"] and seen == []

    def test_fal_respects_budget(self, sep, tmp_path, monkeypatch):
        monkeypatch.setattr(sep._cloud, "get_key", lambda n: "fal-test")
        monkeypatch.setenv("AUDIO_MAX_USD", "0.0001")
        monkeypatch.setattr(sep._cloud, "request", lambda *a, **k: pytest.fail("must not call"))
        with pytest.raises(sep._cloud.CloudError, match="budget"):
            sep.separate_fal(tmp_path / "audio_music.wav", tmp_path / "stem.wav", tmp_path)

    def test_passthrough_main_writes_aligned_outputs(self, sep, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["separate_vocals.py", "--provider", "passthrough"])
        with redirect_stdout(io.StringIO()):
            assert sep.main() == 0
        import soundfile as sf
        v = sf.info(str(tmp_path / "vocals.wav"))
        assert v.samplerate == 16000 and abs(v.frames / 16000 - 3.0) < 0.01
        st = json.loads((tmp_path / "separation_stats.json").read_text())
        assert st["is_passthrough"] and st["active_share"] == 1.0
        assert (tmp_path / "vocal_activity.csv").exists()

    def test_off_removes_stale_stem(self, sep, tmp_path, monkeypatch):
        (tmp_path / "vocals.wav").write_bytes(b"old")
        monkeypatch.setattr("sys.argv", ["separate_vocals.py", "--provider", "off"])
        with redirect_stdout(io.StringIO()):
            assert sep.main() == 0
        assert not (tmp_path / "vocals.wav").exists()


# ---------------------------------------------------------------- Groq ASR
@pytest.fixture()
def p4(tmp_path, cloud, monkeypatch):
    mod = load_phase("phase4_transcribe", tmp_path)
    import _cloud as real
    monkeypatch.setattr(real, "CACHE_DIR", tmp_path / "cache")
    return mod


class TestGroq:
    def test_chunk_plan_owns_every_second_once(self, p4):
        plan = p4.chunk_plan(1500.0, chunk=600.0, overlap=5.0)
        assert plan[0][0] == 0.0 and plan[-1][1] == 1500.0
        owned = [(o0, o1) for _, _, o0, o1 in plan]
        assert owned[0][0] == 0.0 and owned[-1][1] == 1500.0
        assert all(abs(a[1] - b[0]) < 1e-9 for a, b in zip(owned, owned[1:]))   # contiguous
        assert all(c1 - c0 <= 600.0 + 1e-9 for c0, c1, _, _ in plan)
        assert p4.chunk_plan(100.0) == [(0.0, 100.0, 0.0, 100.0)]

    def test_parse_groq_offsets_and_ownership(self, p4):
        resp = {"segments": [
            {"start": 1.0, "end": 3.0, "text": " hello ", "avg_logprob": -0.2, "no_speech_prob": 0.1,
             "compression_ratio": 1.2},
            {"start": 598.0, "end": 599.0, "text": "edge", "avg_logprob": -0.2, "no_speech_prob": 0.1,
             "compression_ratio": 1.2}],
            "words": [{"word": "hello", "start": 1.0, "end": 1.5}]}
        # chunk 2 starts at 595 s and owns [597.5, 1192.5): "hello" (mid 597 s) belongs
        # to chunk 1, "edge" (mid 1193.5 s) to chunk 3 -> neither is kept here
        assert p4.parse_groq(resp, offset=595.0, own=(597.5, 1192.5)) == []
        rows = p4.parse_groq(resp, offset=595.0, own=(595.0, 1200.0))
        assert [r["start_sec"] for r in rows] == [596.0, 1193.0]
        assert rows[0]["words"][0] == {"start": 596.0, "end": 596.5, "word": "hello", "p": None}

    def test_transcribe_groq_request_and_cache(self, p4, tmp_path, monkeypatch):
        import _cloud
        monkeypatch.setattr(_cloud, "get_key", lambda n: "gsk-test")
        seen = []

        def fake(method, url, **k):
            seen.append((url, k.get("data"), k.get("files")))
            return {"language": "english", "segments": [
                {"start": 0.2, "end": 1.5, "text": "la la", "avg_logprob": -0.3,
                 "no_speech_prob": 0.05, "compression_ratio": 1.1}], "words": []}
        monkeypatch.setattr(_cloud, "request", fake)
        audio = _wav(tmp_path / "vocals.wav", seconds=2.0, sr=16000)
        rows = p4.transcribe_groq(audio, "en", tmp_path)
        url, data, files = seen[0]
        assert url == "https://api.groq.com/openai/v1/audio/transcriptions"
        assert ("response_format", "verbose_json") in data and ("temperature", "0") in data
        assert ("timestamp_granularities[]", "word") in data and ("language", "en") in data
        assert files["file"][2] == "audio/flac"
        assert rows[0]["text"] == "la la" and rows[0]["suspect"] is False
        p4.transcribe_groq(audio, "en", tmp_path)
        assert len(seen) == 1                       # second run served from the cache
        usage = json.loads((tmp_path / "cloud_usage.json").read_text())
        assert usage["calls"][0]["est_usd"] == pytest.approx(10 / 3600 * 0.04, abs=1e-5)  # 10 s minimum

    def test_vocal_gate(self, p4, tmp_path):
        (tmp_path / "act.csv").write_text("second,start_sec,end_sec,rms_dbfs,active\n"
                                          "0,0,1,-60,False\n1,1,2,-20,True\n")
        segs = [{"start_sec": 0.1, "end_sec": 0.9, "suspect": False},
                {"start_sec": 0.5, "end_sec": 1.5, "suspect": False}]
        assert p4.apply_vocal_gate(segs, tmp_path / "act.csv") == 1
        assert segs[0]["suspect"] and segs[0]["suspect_reason"] == "vocal stem silent"
        assert not segs[1]["suspect"]


# ---------------------------------------------------------------- structure
@pytest.fixture()
def ms(tmp_path, cloud, monkeypatch):
    mod = load_phase("music_structure", tmp_path)
    monkeypatch.setattr(mod, "_cloud", cloud)
    return mod


class TestStructure:
    def test_music_regions_and_masking(self, ms):
        rows = [{"start_sec": "0", "end_sec": "5", "music": ".9", "speech": ".1", "silence": "0", "rms_dbfs": "-10"},
                {"start_sec": "5", "end_sec": "10", "music": ".9", "speech": ".1", "silence": "0", "rms_dbfs": "-10"},
                {"start_sec": "10", "end_sec": "15", "music": ".1", "speech": ".9", "silence": "0", "rms_dbfs": "-10"}]
        reg = ms.music_regions(rows)
        assert reg == [[0.0, 10.0]]
        assert ms.mask_to_regions([1.0, 9.9, 10.5, 14.0], reg) == [1.0, 9.9]
        assert ms.mask_to_regions([1.0], None) == [1.0]

    def test_bpm_from_beats(self, ms):
        assert ms.bpm_from_beats([i * 0.5 for i in range(20)]) == pytest.approx(120.0)
        assert ms.bpm_from_beats([0.0, 0.5]) is None

    def test_parse_allin1_variants(self, ms, monkeypatch):
        d = {"bpm": 120, "beats": [0.5], "downbeats": [0.5], "segments": [{"start": 0, "end": 9, "label": "verse"}]}
        assert ms.parse_allin1_output(d) == d
        assert ms.parse_allin1_output({"analysis": d}) == d
        monkeypatch.setattr(ms._cloud, "request", lambda *a, **k: d)
        assert ms.parse_allin1_output(["https://x/viz.png", "https://x/out.json"]) == d
        with pytest.raises(ms._cloud.CloudError):
            ms.parse_allin1_output(["https://x/viz.png"])

    def test_replicate_flow_uses_schema_field_and_cache(self, ms, tmp_path, monkeypatch):
        monkeypatch.setattr(ms._cloud, "get_key", lambda n: "r8-test")
        monkeypatch.setattr(ms._cloud, "encode_audio",
                            lambda src, dest, fmt, **k: dest.write_bytes(b"mp3") or dest)
        monkeypatch.setattr(ms.time, "sleep", lambda s: None)
        audio = _wav(tmp_path / "audio_music.wav", seconds=1.0)
        calls = []

        def fake(method, url, **k):
            calls.append((method, url, k.get("json_body")))
            if url.endswith("/models/sakemin/all-in-one-music-structure-analyzer"):
                return {"latest_version": {"id": "v123", "openapi_schema": {"components": {"schemas": {
                    "Input": {"properties": {"music_input": {"type": "string", "format": "uri"},
                                             "visualize": {"type": "boolean"}}}}}}}}
            if url.endswith("/files"):
                return {"urls": {"get": "https://api.replicate.com/v1/files/f1"}}
            if url.endswith("/predictions"):
                return {"id": "p1", "status": "starting", "urls": {"get": "https://api/p1"}}
            return {"status": "succeeded", "output": {"bpm": 128, "beats": [0.1], "downbeats": [0.1],
                                                      "segments": [{"start": 0, "end": 1, "label": "chorus"}]}}
        monkeypatch.setattr(ms._cloud, "request", fake)
        out = ms.replicate_allin1(audio, tmp_path)
        pred = next(c for c in calls if c[1].endswith("/predictions"))
        assert pred[2] == {"version": "v123", "input": {"music_input": "https://api.replicate.com/v1/files/f1"}}
        assert out["segments"][0]["label"] == "chorus" and out["replicate_version"] == "v123"
        n = len(calls)
        ms.replicate_allin1(audio, tmp_path)
        assert len(calls) == n + 1          # only the version lookup; the analysis is cached


# ---------------------------------------------------------------- statistics
def test_boundary_alignment():
    st = load_phase("_stats")
    res = st.boundary_alignment([10.0, 30.0, 50.0], [10.1, 29.9, 50.05, 5.0, 70.0], 100.0)
    assert res["hits"] == 3 and res["chance_rate"] == pytest.approx(0.025)
    assert res["significant"]
    res = st.boundary_alignment([10.0], [10.0], 100.0)
    assert res["p_value"] is None


class TestJoinRound11:
    @pytest.fixture()
    def p7(self, tmp_processed):
        return load_phase("phase7_sync")

    def test_downbeat_section_vocals(self, p7):
        shots = [{"shot_idx": 0, "start_sec": 0.0, "end_sec": 2.0},
                 {"shot_idx": 1, "start_sec": 2.0, "end_sec": 3.0},
                 {"shot_idx": 2, "start_sec": 3.0, "end_sec": 6.0}]
        beats = [i * 0.5 for i in range(12)]
        downbeats = [0.0, 2.0, 4.0]
        segments = [{"start": 0.0, "end": 3.0, "label": "A"}, {"start": 3.0, "end": 6.0, "label": "B"}]
        act = [{"start_sec": str(s), "end_sec": str(s + 1), "active": str(s >= 3)} for s in range(6)]
        rows, stats = p7.join_data(shots, [], [], [], [], [], beats, [], music_present=True,
                                   downbeats=downbeats, segments=segments, vocal_activity=act)
        assert [r["section"] for r in rows] == ["A", "A", "B"]
        assert rows[1]["cut_on_downbeat"] is True and rows[2]["cut_on_downbeat"] is False
        assert rows[0]["vocals_active_share"] == 0.0 and rows[2]["vocals_active_share"] == 1.0
        table = p7.section_table(rows, segments, [])
        assert [t["n_shots"] for t in table] == [2, 1] and table[1]["vocals_active_share"] == 1.0


def test_qa_audio_stack():
    qa = load_phase("_qa")
    out = qa.check_audio_stack({"separation_passthrough": True}, {"downbeats": [], "segments": [{"x": 1}],
                                                                   "sections_labelled": False}, None)
    checks = {i["check"] for i in out}
    assert {"separation_passthrough", "downbeats", "sections"} <= checks


def test_runner_offline_wires_audio_steps(tmp_path, monkeypatch):
    import run_pipeline as rp
    seen = []

    class R:
        returncode = 0
    monkeypatch.setattr(rp.subprocess, "run", lambda argv, **k: seen.append(argv) or R())
    monkeypatch.setattr(rp, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rp, "_finish", lambda *a, **k: tmp_path / "m.json")
    monkeypatch.setattr("sys.argv", ["run_pipeline.py", "https://youtu.be/abcdefghijk", "--offline",
                                     "--provider", "mock"])
    with redirect_stdout(io.StringIO()):
        assert rp.main() == 0
    by_script = {Path(a[1]).name: a for a in seen if len(a) > 1}
    assert by_script["separate_vocals.py"][-1] == "off"
    asr = by_script["phase4_transcribe.py"]
    assert asr[asr.index("--backend") + 1] == "local"
    st = by_script["music_structure.py"]
    assert st[st.index("--sections") + 1] == "local"
    order = [Path(a[1]).name for a in seen if len(a) > 1]
    assert order.index("separate_vocals.py") < order.index("phase4_transcribe.py")
    assert order.index("phase6_music.py") < order.index("music_structure.py") < order.index("phase7_sync.py")


def test_tempo_disagreement_flagged_but_octave_tolerated():
    qa = load_phase("_qa")
    ok = qa.check_tempo_agreement({"tempo_bpm": 120.0}, {"bpm": 60.5, "beats_source": "beat_this"})
    assert ok == []                                   # half tempo: the usual ambiguity
    bad = qa.check_tempo_agreement({"tempo_bpm": 129.2}, {"bpm": 96.77, "beats_source": "beat_this"})
    assert bad and bad[0]["check"] == "tempo_disagreement"   # 4:3 — one tracker is wrong
