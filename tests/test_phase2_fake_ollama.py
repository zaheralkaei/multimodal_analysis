"""Phase 2 against a fake Ollama server: no API key, no network."""
from __future__ import annotations
import csv, json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


class FakeOllama:
    """Records requests; answers like Ollama /api/generate."""

    def __init__(self, reject_schema=False, fail_first=0, reply=None):
        self.requests, self.reject_schema, self.fail_first = [], reject_schema, fail_first
        self.reply = reply or {"caption": "A test shot.", "camera": "pan-left", "emotion": "Sultry",
                               "colors": ["red"], "entities": "box", "location": "indoor studio",
                               "lighting": "flat", "composition": "wide"}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                if outer.reject_schema and isinstance(body.get("format"), dict):
                    self.send_response(400)
                    self.end_headers()
                    return
                if outer.fail_first > 0:
                    outer.fail_first -= 1
                    self.send_response(500)
                    self.end_headers()
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"response": json.dumps(outer.reply)}).encode())

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()


@pytest.fixture
def shots(tmp_path):
    from PIL import Image
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(1, 13):
        Image.new("RGB", (64, 48), (i * 20, 0, 0)).save(frames / f"frame_{i:05d}.jpg")
    return [{"shot_idx": k, "start_sec": k * 2.0, "end_sec": k * 2.0 + 2, "duration_sec": 2.0,
             "mid_frame_path": str(frames / f"frame_{k * 4 + 2:05d}.jpg"),
             "key_frame_paths": [str(frames / f"frame_{k * 4 + j:05d}.jpg") for j in (1, 2, 3)]}
            for k in range(3)]


@pytest.fixture
def phase2(monkeypatch):
    import phase2_vision
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    return phase2_vision


def read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def test_schema_mode_three_images_and_normalization(phase2, shots, tmp_path, monkeypatch):
    fake = FakeOllama()
    monkeypatch.setenv("OLLAMA_BASE_URL", fake.url)
    out = tmp_path / "shot_vision.csv"
    rows, stats = phase2.analyze_shots("m", shots, out, workers=3)
    fake.close()
    assert stats["schema_mode"] and stats["errors"] == 0
    assert all(len(r["images"]) == 3 for r in fake.requests)
    assert fake.requests[0]["format"]["properties"]["emotion"]["enum"]
    saved = read_rows(out)
    assert [int(r["shot_idx"]) for r in saved] == [0, 1, 2]
    assert {r["emotion"] for r in saved} == {"sensual"} and {r["camera"] for r in saved} == {"pan"}


def test_falls_back_when_schema_rejected(phase2, shots, tmp_path, monkeypatch):
    fake = FakeOllama(reject_schema=True)
    monkeypatch.setenv("OLLAMA_BASE_URL", fake.url)
    rows, stats = phase2.analyze_shots("m", shots, tmp_path / "v.csv", workers=1)
    fake.close()
    assert not stats["schema_mode"] and stats["errors"] == 0
    assert fake.requests[-1]["format"] == "json"
    assert all(r["emotion"] == "sensual" for r in rows)


def test_resume_retries_only_failures(phase2, shots, tmp_path, monkeypatch):
    out = tmp_path / "v.csv"
    fake = FakeOllama(fail_first=1)
    monkeypatch.setenv("OLLAMA_BASE_URL", fake.url)
    _, stats = phase2.analyze_shots("m", shots, out, workers=1)
    assert stats["errors"] == 1
    n_before = len(fake.requests)
    _, stats2 = phase2.analyze_shots("m", shots, out, workers=1)
    assert len(fake.requests) - n_before == 1 and stats2["errors"] == 0
    # Changing the shot list (phase 1 re-run) invalidates the stale row
    shots[0]["start_sec"] = 0.5
    n_before = len(fake.requests)
    phase2.analyze_shots("m", shots, out, workers=1)
    fake.close()
    assert len(fake.requests) - n_before == 1
    assert len(read_rows(out)) == 3
