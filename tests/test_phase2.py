"""Round-4 audit: phase 2 OpenRouter backend, JSON parsing, resume semantics."""
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from conftest import load_phase


@pytest.fixture()
def p2(tmp_path):
    return load_phase("phase2_vision", tmp_path)


class TestProviderResolution:
    def test_auto_uses_openrouter_when_key_set(self, p2, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        assert p2.resolve_provider("auto") == "openrouter"

    def test_auto_falls_back_to_ollama(self, p2, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert p2.resolve_provider("auto") == "ollama"

    def test_explicit_provider_wins(self, p2, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        assert p2.resolve_provider("ollama") == "ollama"

    def test_openrouter_endpoint(self, p2):
        assert p2.endpoint_for("openrouter").startswith("https://openrouter.ai/api/v1")


class TestOpenRouterCall:
    def _fake_urlopen(self, captured, response_body):
        class FakeResp:
            def __enter__(self):
                return io.BytesIO(json.dumps(response_body).encode())

            def __exit__(self, *a):
                return False

        def fake(req, timeout=None):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            captured["body"] = json.loads(req.data.decode())
            return FakeResp()

        return fake

    def test_call_openrouter_sends_chat_completion(self, p2, monkeypatch):
        captured = {}
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        monkeypatch.setattr("urllib.request.urlopen",
                            self._fake_urlopen(captured, {
                                "choices": [{"message": {"content": "READY OK"}}]}))
        text, secs = p2.call_openrouter("google/gemma-4-31b-it", "PROMPT", image_b64="AAAB", timeout=5)
        assert text == "READY OK"
        assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
        assert captured["headers"]["Authorization"] == "Bearer sk-test"
        body = captured["body"]
        assert body["model"] == "google/gemma-4-31b-it"
        assert body["temperature"] == 0 and body["seed"] == 42
        assert body["messages"][0]["content"][0]["type"] == "text"
        assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        # Ollama-style 'images' key must NOT leak into the OpenRouter payload
        assert "images" not in body

    def test_call_openrouter_requires_key(self, p2, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            p2.call_openrouter("m", "p", "AAAB")

    def test_call_openrouter_bad_shape_raises(self, p2, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        monkeypatch.setattr("urllib.request.urlopen",
                            self._fake_urlopen({}, {"error": "quota"}))
        with pytest.raises(RuntimeError, match="unexpected OpenRouter response"):
            p2.call_openrouter("m", "p", "AAAB")

    def test_dispatch_openrouter(self, p2, monkeypatch):
        monkeypatch.setattr(p2, "call_openrouter", lambda *a, **k: ("via-or", 0.1))
        assert p2.call_llm("openrouter", "m", "p", "AAAB")[0] == "via-or"

    def test_dispatch_ollama(self, p2, monkeypatch):
        monkeypatch.setattr(p2, "call_ollama", lambda *a, **k: ("via-ollama", 0.1))
        assert p2.call_llm("ollama", "m", "p", "AAAB")[0] == "via-ollama"


class TestParseJsonResponse:
    def test_clean_json(self, p2):
        got = p2.parse_json_response('{"caption": "a", "camera": "pan"}')
        assert got["caption"] == "a"
        assert got["camera"] == "pan"
        assert got["location"] == ""  # missing keys are empty, not absent

    def test_markdown_fence(self, p2):
        got = p2.parse_json_response('```json\n{"caption": "b"}\n```')
        assert got["caption"] == "b"

    def test_junk_around_json(self, p2):
        got = p2.parse_json_response('Sure! Here you go:\n{"caption": "c"}\nDone.')
        assert got["caption"] == "c"

    def test_garbage_reports_parse_error(self, p2):
        got = p2.parse_json_response("I cannot see any image at all sorry")
        assert "_parse_error" in got

    def test_list_value_joined(self, p2):
        got = p2.parse_json_response('{"colors": ["red", "blue"]}')
        assert got["colors"] == "red, blue"


class TestResume:
    def _shots(self, base: Path):
        """base corresponds to data/vid/ in the phase-2 world (paths in
        shots.json are REPO_ROOT-relative, hence 'data/vid/frames/...')."""
        frames = base / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        for i in (3, 4):
            (frames / f"frame_{i:05d}.jpg").write_bytes(b"jpg")
        return [
            {"shot_idx": 0, "start_sec": 0.0, "end_sec": 2.0, "duration_sec": 2.0,
             "mid_frame_path": "data/vid/frames/frame_00003.jpg"},
            {"shot_idx": 1, "start_sec": 2.0, "end_sec": 4.0, "duration_sec": 2.0,
             "mid_frame_path": "data/vid/frames/frame_00004.jpg"},
        ]

    def test_row_reanalyzed_when_mid_frame_changes(self, p2, tmp_path, monkeypatch):
        monkeypatch.setattr(p2, "REPO_ROOT", tmp_path)
        shots = self._shots(tmp_path / "data" / "vid")
        out_csv = tmp_path / "shot_vision.csv"
        # Simulate a stale run: row's mid_frame points to the OLD (wrong) frame
        old_rel = "data/vid/frames/frame_00002.jpg"  # no longer what shots.json says
        out_csv.write_text(
            "shot_idx,start_sec,end_sec,duration_sec,mid_frame,caption\n"
            f"0,0.0,2.0,2.0,{old_rel},stale caption\n"
            f"1,2.0,4.0,2.0,{old_rel},stale caption\n",
            encoding="utf-8")

        monkeypatch.setattr(p2, "call_llm",
                            lambda *a, **k: ('{"caption": "fresh"}', 0.01))
        rows, stats = p2.analyze_shots("ollama", "m", shots,
                                      tmp_path / "data" / "vid" / "frames", out_csv)
        # Both rows must have been re-analyzed (mid_frame changed)
        assert stats["calls"] == 2
        assert all(r["caption"] == "fresh" for r in rows)

    def test_row_reused_when_mid_frame_matches(self, p2, tmp_path, monkeypatch):
        monkeypatch.setattr(p2, "REPO_ROOT", tmp_path)
        shots = self._shots(tmp_path / "data" / "vid")
        out_csv = tmp_path / "shot_vision.csv"
        cur = shots[0]["mid_frame_path"]
        out_csv.write_text(
            "shot_idx,start_sec,end_sec,duration_sec,mid_frame,caption\n"
            f"0,0.0,2.0,2.0,{cur},keep me\n",
            encoding="utf-8")
        monkeypatch.setattr(p2, "call_llm",
                            lambda *a, **k: ('{"caption": "fresh"}', 0.01))
        rows, stats = p2.analyze_shots("ollama", "m", shots,
                                      tmp_path / "data" / "vid" / "frames", out_csv)
        # shot 0 reused from CSV; shot 1 (no prior row) analyzed
        assert stats["calls"] == 1
        assert rows[0]["caption"] == "keep me"
        assert rows[1]["caption"] == "fresh"

    def test_missing_mid_frame_recorded_as_skipped(self, p2, tmp_path):
        shots = [{"shot_idx": 0, "start_sec": 0.0, "end_sec": 2.0, "duration_sec": 2.0,
                  "mid_frame_path": "data/vid/frames/frame_99999.jpg"}]
        rows, stats = p2.analyze_shots("ollama", "m", shots, tmp_path / "frames",
                                       tmp_path / "shot_vision.csv")
        assert stats["skipped"] == [0]
        assert rows == []
        # CSV stays in sync with shots.json (no phantom rows)
        content = (tmp_path / "shot_vision.csv").read_text(encoding="utf-8")
        assert len(content.strip().splitlines()) == 1  # header only