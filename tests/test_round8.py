"""Round 8: analysis-validity fixes and spend-safety features.

Covers: chance-corrected beat statistics, phase-2 offline mock / budget /
retry / resume rules, Whisper hallucination flags, per-group CLAP
normalization, stale-row guards in the join, provenance manifest, ffmpeg
discovery, ffprobe-free metadata, runner exit codes.
"""
import io
import json
import math
import random
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from conftest import load_phase


# ---------------------------------------------------------------- _stats
class TestBeatStatistics:
    @pytest.fixture()
    def st(self):
        return load_phase("_stats")

    def test_hit_rate(self, st):
        beats = [0.0, 0.5, 1.0, 1.5]
        assert st.hit_rate([0.05, 0.75], beats, 0.1) == 0.5

    def test_beat_phase(self, st):
        assert st.beat_phases([0.25, 0.5], [0.0, 1.0]) == [0.25, 0.5]
        assert st.beat_phases([5.0], [0.0, 1.0]) == []      # outside the grid

    def test_rayleigh_concentrated_vs_uniform(self, st):
        conc = st.rayleigh([0.01, 0.99, 0.0, 0.02, 0.98] * 6)
        unif = st.rayleigh([i / 30 for i in range(30)])
        assert conc["R"] > 0.9 and conc["p"] < 1e-6
        assert unif["R"] < 0.05 and unif["p"] > 0.5

    def test_cuts_on_beat_are_significant(self, st):
        beats = [i * 0.5 for i in range(400)]                # 120 BPM, 200 s
        cuts = [b + 0.01 for b in beats[3::7]]
        res = st.beat_alignment(cuts, beats, 200.0, tol=0.1, n_perm=500)
        assert res["observed_rate"] == 1.0
        assert 0.3 < res["chance_rate"] < 0.5                # ±0.1 of 0.5 s grid = 40 %
        assert res["significant"] and res["p_value"] < 0.01

    def test_random_cuts_are_not_significant(self, st):
        rng = random.Random(1)
        beats = [i * 0.5 for i in range(400)]
        cuts = sorted(rng.uniform(0, 200) for _ in range(60))
        res = st.beat_alignment(cuts, beats, 200.0, tol=0.1, n_perm=500)
        assert not res["significant"]
        assert abs(res["observed_rate"] - res["chance_rate"]) < 3 * res["chance_sd"]

    def test_beat_locked_periodic_editing_is_detected(self, st):
        # cuts every 7 beats exactly: the circular-shift null cannot see this
        # (every shift moves all cuts together) — the per-cut model can
        beats = [i * 0.5 for i in range(400)]
        cuts = [b + 0.01 for b in beats[3::7]]
        res = st.beat_alignment(cuts, beats, 200.0, n_perm=200)
        assert res["p_value"] < 1e-10
        assert res["circular_shift_check"]["p_value"] > 0.05

    def test_poisson_binomial_exact(self, st):
        assert st.poisson_binomial_sf([0.5, 0.5], 2) == pytest.approx(0.25)
        assert st.poisson_binomial_sf([0.5, 0.5], 1) == pytest.approx(0.75)
        assert st.poisson_binomial_sf([0.2] * 5, 0) == pytest.approx(1.0)

    def test_tempo_change_uses_local_period(self, st):
        beats = [i * 0.5 for i in range(100)] + [50 + i * 0.25 for i in range(1, 200)]
        res = st.beat_alignment([10.3, 20.2, 60.1, 70.13], beats, 100.0, n_perm=0)
        # 2 cuts at 0.5 s period (p=.4), 2 at 0.25 s (p=.8)
        assert res["chance_rate"] == pytest.approx(0.6, abs=1e-6)

    def test_deterministic(self, st):
        beats = [i * 0.5 for i in range(100)]
        cuts = [1.0, 7.3, 12.1, 20.0, 33.3]
        a = st.beat_alignment(cuts, beats, 50.0, n_perm=200)
        b = st.beat_alignment(cuts, beats, 50.0, n_perm=200)
        assert a == b

    def test_too_few_events(self, st):
        res = st.beat_alignment([1.0], [0.0, 0.5, 1.0], 2.0)
        assert res["significant"] is False and "note" in res


# ---------------------------------------------------------------- phase 2
@pytest.fixture()
def p2(tmp_path):
    mod = load_phase("phase2_vision", tmp_path)
    for k in mod.USAGE:
        mod.USAGE[k] = 0 if k != "cost_usd" else 0.0
    return mod


def _shots(base: Path, n=2):
    frames = base / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    shots = []
    for i in range(n):
        (frames / f"frame_{i + 1:05d}.jpg").write_bytes(f"jpg{i}".encode())
        shots.append({"shot_idx": i, "start_sec": 2.0 * i, "end_sec": 2.0 * i + 2,
                      "duration_sec": 2.0,
                      "mid_frame_path": f"data/vid/frames/frame_{i + 1:05d}.jpg"})
    return shots


class TestPhase2Safety:
    def test_no_blank_line_stop_sequence(self, p2):
        # "\n\n" truncated pretty-printed JSON mid-object
        assert "\n\n" not in p2.STOP_SEQUENCES

    def test_mock_provider_runs_offline_and_is_labelled(self, p2, tmp_path, monkeypatch):
        monkeypatch.setattr(p2, "REPO_ROOT", tmp_path)
        monkeypatch.setattr("urllib.request.urlopen",
                            lambda *a, **k: pytest.fail("mock must not touch the network"))
        shots = _shots(tmp_path / "data" / "vid")
        rows, stats = p2.analyze_shots("mock", "m", shots, tmp_path, tmp_path / "v.csv")
        assert stats["calls"] == 2 and stats["errors"] == 0
        assert all(r["status"] == "mock" and "MOCK" in r["caption"] for r in rows)

    def test_mock_rows_are_never_reused_for_real_runs(self, p2):
        row = {"mid_frame": "f", "model": "m", "status": "mock", "caption": "MOCK"}
        assert not p2._reusable(row, "f", "m")

    def test_error_rows_are_retried(self, p2):
        assert not p2._reusable({"mid_frame": "f", "caption": "[error: 429]"}, "f", "m")
        assert not p2._reusable({"mid_frame": "f", "status": "parse_error", "caption": "x"}, "f", "m")

    def test_model_change_reanalyzes(self, p2):
        assert not p2._reusable({"mid_frame": "f", "model": "a", "caption": "ok"}, "f", "b")
        assert p2._reusable({"mid_frame": "f", "model": "b", "caption": "ok"}, "f", "b")
        assert p2._reusable({"mid_frame": "f", "caption": "legacy row"}, "f", "b")

    def test_estimate_cost(self, p2):
        pricing = {"prompt": "0.0000002", "completion": "0.0000004"}
        est = p2.estimate_cost(100, pricing)
        assert 0.01 < est < 0.1
        assert p2.estimate_cost(10, None) is None

    def test_budget_stops_calls(self, p2, tmp_path, monkeypatch):
        monkeypatch.setattr(p2, "REPO_ROOT", tmp_path)
        shots = _shots(tmp_path / "data" / "vid", n=3)
        calls = []

        def fake(*a, **k):
            calls.append(1)
            p2.USAGE["cost_usd"] += 0.6
            return '{"caption": "x"}', 0.0
        monkeypatch.setattr(p2, "call_llm", fake)
        rows, stats = p2.analyze_shots("openrouter", "m", shots, tmp_path,
                                       tmp_path / "v.csv", max_usd=1.0)
        assert len(calls) == 2                       # 0.6, 1.2 -> stop
        assert rows[2]["status"] == "budget"

    def test_retry_then_success(self, p2, monkeypatch):
        import urllib.error
        monkeypatch.setattr(p2.time, "sleep", lambda s: None)
        attempts = []

        class Resp:
            def __enter__(self):
                return io.BytesIO(json.dumps({"choices": [{"message": {"content": "ok"}}],
                                              "usage": {"prompt_tokens": 10,
                                                        "completion_tokens": 2,
                                                        "cost": 0.001}}).encode())

            def __exit__(self, *a):
                return False

        def fake(req, timeout=None):
            attempts.append(1)
            if len(attempts) < 3:
                raise urllib.error.HTTPError(req.full_url, 429, "rate", {}, io.BytesIO(b""))
            return Resp()
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        monkeypatch.setattr("urllib.request.urlopen", fake)
        text, _ = p2.call_openrouter("m", "p", "AAAB")
        assert text == "ok" and len(attempts) == 3
        assert p2.USAGE["retries"] == 2 and p2.USAGE["cost_usd"] == pytest.approx(0.001)

    def test_non_retryable_error_raises(self, p2, monkeypatch):
        import urllib.error

        def fake(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 401, "auth", {}, io.BytesIO(b"bad key"))
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        monkeypatch.setattr("urllib.request.urlopen", fake)
        with pytest.raises(RuntimeError, match="HTTP 401"):
            p2.call_openrouter("m", "p", "AAAB")

    def test_dry_run_makes_no_model_calls(self, p2, tmp_path, monkeypatch):
        proc = tmp_path
        (proc / "shots.json").write_text(json.dumps(_shots(tmp_path / "data" / "vid")))
        monkeypatch.setattr(p2, "PROCESSED", proc)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        monkeypatch.setattr(p2, "fetch_openrouter_pricing",
                            lambda m: {"prompt": "0.0000002", "completion": "0.0000004"})
        monkeypatch.setattr(p2, "call_llm", lambda *a, **k: pytest.fail("dry run called the model"))
        monkeypatch.setattr("sys.argv", ["phase2_vision.py", "--provider", "openrouter", "--dry-run"])
        with redirect_stdout(io.StringIO()) as out:
            assert p2.main() == 0
        assert "[cost] estimated upper bound" in out.getvalue()

    def test_estimate_over_budget_aborts(self, p2, tmp_path, monkeypatch):
        (tmp_path / "shots.json").write_text(json.dumps(_shots(tmp_path / "data" / "vid")))
        monkeypatch.setattr(p2, "PROCESSED", tmp_path)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        monkeypatch.setattr(p2, "fetch_openrouter_pricing",
                            lambda m: {"prompt": "0.01", "completion": "0.01"})
        monkeypatch.setattr(p2, "call_llm", lambda *a, **k: pytest.fail("must abort first"))
        monkeypatch.setattr("sys.argv", ["phase2_vision.py", "--provider", "openrouter",
                                         "--max-usd", "0.5"])
        with redirect_stdout(io.StringIO()):
            assert p2.main() == 1

    def test_unknown_model_slug_aborts(self, p2, tmp_path, monkeypatch):
        (tmp_path / "shots.json").write_text(json.dumps(_shots(tmp_path / "data" / "vid")))
        monkeypatch.setattr(p2, "PROCESSED", tmp_path)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        monkeypatch.setattr(p2, "fetch_openrouter_pricing", lambda m: {})
        monkeypatch.setattr("sys.argv", ["phase2_vision.py", "--provider", "openrouter"])
        with redirect_stdout(io.StringIO()):
            assert p2.main() == 1


# ---------------------------------------------------------------- phase 4
class TestWhisperSuspect:
    @pytest.fixture()
    def p4(self, tmp_path):
        return load_phase("phase4_transcribe", tmp_path)

    def test_repetition_loop_flagged(self, p4):
        assert p4.is_suspect({"compression_ratio": 3.1, "no_speech_prob": 0.0, "avg_logprob": -0.2})

    def test_text_over_silence_flagged(self, p4):
        assert p4.is_suspect({"compression_ratio": 1.2, "no_speech_prob": 0.8, "avg_logprob": -1.3})

    def test_normal_segment_kept(self, p4):
        assert not p4.is_suspect({"compression_ratio": 1.5, "no_speech_prob": 0.7, "avg_logprob": -0.4})


# ---------------------------------------------------------------- phase 5
class TestClapNormalization:
    @pytest.fixture()
    def p5(self, tmp_path):
        return load_phase("phase5_audio", tmp_path)

    def test_softmax_is_per_group(self, p5):
        import numpy as np
        from _clap_tags import ALL_TAGS, TAG_GROUPS
        logits = np.random.default_rng(0).normal(size=(4, len(ALL_TAGS))) * 10
        probs = p5.group_softmax(logits, TAG_GROUPS, ALL_TAGS)
        for names in TAG_GROUPS.values():
            idx = [ALL_TAGS.index(t) for t in names]
            assert np.allclose(probs[:, idx].sum(axis=1), 1.0)

    def test_instrument_logits_do_not_change_mood_probs(self, p5):
        import numpy as np
        from _clap_tags import ALL_TAGS, TAG_GROUPS, INSTRUMENT_TAGS, MOOD_TAGS
        a = np.zeros((1, len(ALL_TAGS)))
        b = a.copy()
        b[0, ALL_TAGS.index(INSTRUMENT_TAGS[0])] = 50.0   # very "piano"
        mood = [ALL_TAGS.index(t) for t in MOOD_TAGS]
        pa = p5.group_softmax(a, TAG_GROUPS, ALL_TAGS)[0, mood]
        pb = p5.group_softmax(b, TAG_GROUPS, ALL_TAGS)[0, mood]
        assert np.allclose(pa, pb)

    def test_prefers_full_band_audio(self, p5, tmp_path):
        (tmp_path / "audio.wav").write_bytes(b"x")
        assert p5.pick_audio(tmp_path) == (tmp_path / "audio.wav", True)
        (tmp_path / "audio_music.wav").write_bytes(b"x")
        assert p5.pick_audio(tmp_path) == (tmp_path / "audio_music.wav", False)


# ---------------------------------------------------------------- phase 7
class TestJoinGuards:
    @pytest.fixture()
    def p7(self, tmp_processed):
        return load_phase("phase7_sync")

    def _shots(self):
        return [{"shot_idx": 0, "start_sec": 0.0, "end_sec": 4.0},
                {"shot_idx": 1, "start_sec": 4.0, "end_sec": 8.0}]

    def test_stale_vision_row_is_dropped(self, p7):
        vision = [{"shot_idx": "0", "start_sec": "0.0", "caption": "good"},
                  {"shot_idx": "1", "start_sec": "5.5", "caption": "from an old shot list"}]
        rows, stats = p7.join_data(self._shots(), vision, [], [], [], [], [], [])
        assert rows[0]["vision_caption"] == "good"
        assert rows[1]["vision_caption"] == ""
        assert stats["misaligned_rows_dropped"]["vision"] == 1

    def test_suspect_transcript_excluded(self, p7):
        tr = [{"start_sec": "1", "end_sec": "2", "text": "real words", "suspect": "False"},
              {"start_sec": "5", "end_sec": "6", "text": "thanks for watching", "suspect": "True"}]
        rows, stats = p7.join_data(self._shots(), [], [], tr, [], [], [], [])
        assert rows[0]["n_lyric_segments"] == 1
        assert rows[1]["n_lyric_segments"] == 0
        assert stats["suspect_transcript_segments_excluded"] == 1
        assert stats["total_lyric_segments"] == 1

    def test_first_shot_is_not_a_cut(self, p7):
        rows, stats = p7.join_data(self._shots(), [], [], [], [], [], [0.0, 4.0], [])
        assert rows[0]["cut_on_beat"] == ""
        assert rows[1]["cut_on_beat"] is True
        assert stats["total_cuts"] == 1


# ---------------------------------------------------------------- provenance
class TestProvenance:
    def test_manifest_appends_runs(self, tmp_path):
        pv = load_phase("_provenance")
        video = tmp_path / "v.mp4"
        video.write_bytes(b"abc")
        (tmp_path / "sync_stats.json").write_text("{}")
        for _ in range(2):
            run = pv.new_run(["run_pipeline.py", "x"])
            run["phases"].append({"phase": 1, "returncode": 0})
            path = pv.finish_run(run, tmp_path, video)
        data = json.loads(path.read_text())
        assert len(data["runs"]) == 2
        r = data["runs"][-1]
        assert r["input_video"]["sha256"] == \
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        assert "sync_stats.json" in r["stats_files"]
        assert "numpy" in r["packages"] and "commit" in r["git"]


# ---------------------------------------------------------------- tools / phase 0
class TestTools:
    def test_env_override(self, tmp_path, monkeypatch):
        tools = load_phase("_tools")
        exe = tmp_path / "ffmpeg.exe"
        exe.write_bytes(b"")
        monkeypatch.setenv("FFMPEG_BINARY", str(exe))
        assert tools.ffmpeg_exe() == str(exe)

    def test_ffprobe_sibling(self, tmp_path, monkeypatch):
        tools = load_phase("_tools")
        (tmp_path / "ffmpeg.exe").write_bytes(b"")
        (tmp_path / "ffprobe.exe").write_bytes(b"")
        monkeypatch.setenv("FFMPEG_BINARY", str(tmp_path / "ffmpeg.exe"))
        monkeypatch.delenv("FFPROBE_BINARY", raising=False)
        monkeypatch.setattr(tools.shutil, "which", lambda name: None)
        assert tools.ffprobe_exe() == str(tmp_path / "ffprobe.exe")

    def test_metadata_without_ffprobe(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        import numpy as np
        p0 = load_phase("phase0_input", tmp_path)
        path = tmp_path / "v.avi"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
        for _ in range(50):
            vw.write(np.zeros((48, 64, 3), np.uint8))
        vw.release()
        meta = p0.extract_metadata_cv2(path)
        assert meta["video"]["width"] == 64 and meta["video"]["height"] == 48
        assert meta["video"]["fps"] == pytest.approx(25)
        assert meta["duration_sec"] == pytest.approx(2.0)


class TestRunnerExitCode:
    def test_phase0_failure_is_nonzero(self, tmp_path, monkeypatch):
        import run_pipeline as rp

        class R:
            returncode = 1
        monkeypatch.setattr(rp.subprocess, "run", lambda *a, **k: R())
        monkeypatch.setattr(rp, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(rp, "_finish", lambda *a, **k: tmp_path / "m.json")
        monkeypatch.setattr("sys.argv", ["run_pipeline.py", "https://youtu.be/abcdefghijk"])
        with redirect_stdout(io.StringIO()):
            rc = rp.main()
        assert rc == 10        # was 0: a failed download looked like success


class TestWebUiMockOption:
    def test_mock_checkbox_maps_to_provider_mock(self):
        import app as app_mod
        args, err = app_mod.parse_options({"mock_vision": "on"})
        assert err is None and args[args.index("--provider") + 1] == "mock"

    def test_skip_wins_over_mock(self):
        import app as app_mod
        args, err = app_mod.parse_options({"mock_vision": "on", "skip_phase2": "on"})
        assert "--skip-phase2" in args and "--provider" not in args


class TestPreflight:
    def _models(self, monkeypatch, models):
        class Resp:
            def __enter__(self):
                return io.BytesIO(json.dumps({"data": models}).encode())

            def __exit__(self, *a):
                return False
        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Resp())

    def test_unknown_model_is_blocking(self, monkeypatch):
        pf = load_phase("preflight")
        self._models(monkeypatch, [{"id": "other/model"}])
        r = pf.Report()
        with redirect_stdout(io.StringIO()):
            pf.check_online(r, "google/gemma-4-31b-it", 10, 1.0)
        assert r.blocking

    def test_text_only_model_is_blocking(self, monkeypatch):
        pf = load_phase("preflight")
        self._models(monkeypatch, [{"id": "a/b", "architecture": {"input_modalities": ["text"]},
                                    "pricing": {"prompt": "0", "completion": "0"}}])
        r = pf.Report()
        with redirect_stdout(io.StringIO()):
            pf.check_online(r, "a/b", 10, 1.0)
        assert r.blocking

    def test_good_model_reports_cost(self, monkeypatch):
        pf = load_phase("preflight")
        self._models(monkeypatch, [{"id": "a/b",
                                    "architecture": {"input_modalities": ["image", "text"]},
                                    "supported_parameters": ["temperature", "seed",
                                                             "response_format", "max_tokens"],
                                    "pricing": {"prompt": "0.0000002", "completion": "0.0000004"}}])
        r = pf.Report()
        with redirect_stdout(io.StringIO()):
            pf.check_online(r, "a/b", 72, 1.0)
        assert not r.blocking
        assert any("estimated phase-2 cost" in w for _, w, _ in r.items)


class TestWebUiSecurity:
    """Audit R9: command injection via the URL + cross-site requests."""

    @pytest.fixture()
    def client(self):
        import app as app_mod
        app_mod.app.config["TESTING"] = True
        return app_mod.app.test_client()

    def test_cross_origin_post_refused(self, client):
        r = client.post("/start", data={"url": "https://youtu.be/abcdefghijk"},
                        headers={"Origin": "https://evil.example"})
        assert r.status_code == 403

    def test_cross_site_fetch_metadata_refused(self, client):
        r = client.post("/job/x/cancel", headers={"Sec-Fetch-Site": "cross-site"})
        assert r.status_code == 403

    def test_rebound_host_refused(self, client):
        r = client.get("/", headers={"Host": "attacker.example:5000"})
        assert r.status_code == 403

    def test_same_origin_allowed(self, client):
        r = client.get("/", headers={"Host": "127.0.0.1:5000"})
        assert r.status_code == 200

    def test_quote_in_url_rejected(self, client, monkeypatch):
        import app as app_mod
        monkeypatch.setattr(app_mod, "start_job", lambda *a, **k: pytest.fail("must not start"))
        r = client.post("/start", data={"url": 'https://x.com/a" & calc & "'})
        assert r.status_code == 400

    def test_runner_never_uses_a_shell(self, tmp_path, monkeypatch):
        import run_pipeline as rp
        seen = []

        class R:
            returncode = 1

        def fake(cmd, *a, **k):
            if k.get("shell"):
                pytest.fail("shell=True used")
            seen.append(cmd)
            return R()
        monkeypatch.setattr(rp.subprocess, "run", fake)
        monkeypatch.setattr(rp, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(rp, "_finish", lambda *a, **k: tmp_path / "m.json")
        hostile = 'https://example.com/a" & echo INJECTED & "'
        monkeypatch.setattr("sys.argv", ["run_pipeline.py", hostile, "--id", "x"])
        with redirect_stdout(io.StringIO()):
            rp.main()
        phase0 = [c for c in seen if isinstance(c, list) and any("phase0" in str(x) for x in c)][0]
        assert hostile in phase0          # passed as ONE argv element, never parsed


class TestMusicPresence:
    """Audit R9: librosa reports a tempo for speech, so tempo alone made
    every speech a 'song'."""

    def _args(self, kind):
        from _clap_tags import AUDIO_TYPE_TAGS
        row = {t: 0.0 for t in AUDIO_TYPE_TAGS}
        row[kind] = 0.9
        dur = 130.0
        return dict(shots=[{"duration_sec": dur, "end_sec": dur, "camera_motion": "static"}],
                    music_summary={"tempo_bpm": 112.3, "n_beats": 250},
                    transcript=[{"start_sec": 0, "end_sec": dur, "text": "x" * 1800}],
                    video_duration=dur, clap_rows=[dict(row) for _ in range(26)])

    def test_speech_with_spurious_tempo_is_speech(self):
        from _content_profile import detect_content_type
        prof = detect_content_type(**self._args("speech"))
        assert prof["type"] == "speech"
        assert any("tempo ignored" in r for r in prof["reasons"])

    def test_music_is_song(self):
        from _content_profile import detect_content_type
        a = self._args("music")
        a["transcript"] = [{"start_sec": 10, "end_sec": 20, "text": "la la"}]
        a["shots"] = [{"duration_sec": 2.0, "end_sec": 2.0 * (i + 1), "camera_motion": "pan-left"}
                      for i in range(65)]
        assert detect_content_type(**a)["type"] == "song"

    def test_without_clap_music_is_flagged_unverified(self):
        from _content_profile import detect_content_type
        a = self._args("speech")
        a["clap_rows"] = []
        prof = detect_content_type(**a)
        assert any("UNVERIFIED" in r for r in prof["reasons"])

    def test_music_share(self):
        from _clap_tags import music_share
        rows = [{"music": 0.8, "speech": 0.1, "silence": 0.1},
                {"music": 0.2, "speech": 0.7, "silence": 0.1}]
        assert music_share(rows) == 0.5
        assert music_share([{"happy and bright": 0.3}]) is None


class TestEmotionNormalizer:
    @pytest.mark.parametrize("raw,expected", [
        ("courageous", "other"), ("dangerous", "other"), ("stranger", "other"),
        ("funeral mood", "other"), ("dangerous / tense", "anxious"),
        ("none (no people)", "no people"), ("No people visible.", "no people"),
        ("sultry/confident", "sensual"), ("intense or fierce", "intense"),
        ("Based on the body language and downcast expressions of the", "other"),
    ])
    def test_whole_word_matching(self, raw, expected):
        from _normalize_emotion import normalize_emotion
        assert normalize_emotion(raw) == expected

    def test_no_people_is_not_neutral(self):
        from _normalize_emotion import normalize_emotion, list_canonical
        assert normalize_emotion("none") == "no people"
        assert "no people" not in list_canonical()

    def test_raw_emotion_kept(self, p2, tmp_path, monkeypatch):
        monkeypatch.setattr(p2, "REPO_ROOT", tmp_path)
        shots = _shots(tmp_path / "data" / "vid", n=1)
        monkeypatch.setattr(p2, "call_llm", lambda *a, **k: ('{"emotion": "Sultry"}', 0.0))
        rows, _ = p2.analyze_shots("openrouter", "m", shots, tmp_path, tmp_path / "v.csv")
        assert rows[0]["emotion"] == "sensual" and rows[0]["emotion_raw"] == "Sultry"


def test_beats_counted_inside_shot_only(tmp_processed):
    p7 = load_phase("phase7_sync")
    shots = [{"shot_idx": 0, "start_sec": 0.0, "end_sec": 2.5},
             {"shot_idx": 1, "start_sec": 2.5, "end_sec": 3.2}]
    music = [{"start_sec": str(i), "end_sec": str(i + 1), "rms_energy": "0.1", "n_beats": "2"}
             for i in range(4)]
    beats = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75]
    rows, _ = p7.join_data(shots, [], [], [], [], music, beats, [])
    assert rows[1]["music_n_beats"] == 1     # only 2.75 lies in [2.5, 3.2)


class TestPhase0Download:
    def test_no_playlist_and_long_timeout(self, tmp_path, monkeypatch):
        p0 = load_phase("phase0_input", tmp_path)
        seen = {}

        def fake(cmd, check=False, timeout=None, **k):
            seen["cmd"], seen["timeout"] = cmd, timeout
            (p0.RAW / "vid.mp4").write_bytes(b"x")
        monkeypatch.setattr(p0, "RAW", tmp_path / "raw")
        monkeypatch.setattr(p0.subprocess, "run", fake)
        p0.get_video("https://www.youtube.com/watch?v=abcdefghijk&list=PL123", "vid")
        assert "--no-playlist" in seen["cmd"]
        assert seen["timeout"] >= 3600

    def test_media_timeout_scales(self, tmp_path):
        p0 = load_phase("phase0_input", tmp_path)
        assert p0.media_timeout(60) == 300
        assert p0.media_timeout(2 * 3600) == 4 * 3600


def test_clap_never_marks_short_windows_as_longer(tmp_path):
    """transformers randomly flags one clip per batch as 'longer' for the fused
    CLAP model; phase 5 must override that (run-to-run determinism)."""
    torch = pytest.importorskip("torch")
    p5 = load_phase("phase5_audio", tmp_path)
    seen = {}

    def processor(**kw):
        return {"input_features": torch.zeros(3, 1), "is_longer": torch.tensor([[False], [True], [False]])}

    class Model:
        logit_scale_a = torch.tensor(0.0)

        def __call__(self, **inputs):
            seen["is_longer"] = inputs["is_longer"]

            class O:
                logits_per_audio = torch.zeros(3, 2)
            return O()
    p5.compute_batch_logits(Model(), processor, [0, 0, 0], 48000)
    assert not seen["is_longer"].any()
