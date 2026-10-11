"""Tests for the local Flask web UI wrapper (scripts/app.py).

The app never runs pipeline code itself — jobs are subprocesses of
run_pipeline.py, so every test stubs Popen with a FakeProc that plays back
scripted stdout (same stubbing style as test_staging's subprocess.run fake).
"""
import io
import time
import zipfile
from collections import deque
from pathlib import Path

import pytest

import app as app_mod  # conftest puts scripts/ on sys.path


# ---------------------------------------------------------------- stubs

class FakeProc:
    def __init__(self, argv, kwargs, lines, rc):
        self.argv, self.kwargs, self.pid = argv, kwargs, 4321
        self.stdout = (line + "\n" for line in lines)
        self._rc = rc
        self.terminated = False

    def wait(self):
        return self._rc

    def terminate(self):
        self.terminated = True


class Recorder:
    """Popen stand-in: records every call, replays scripted stdout."""

    def __init__(self, lines=(), rc=0):
        self.calls: list[FakeProc] = []
        self.lines, self.rc = list(lines), rc

    def __call__(self, argv, **kwargs):
        proc = FakeProc(argv, kwargs, self.lines, self.rc)
        self.calls.append(proc)
        return proc


@pytest.fixture
def clean(tmp_path, monkeypatch):
    """Redirect all module-level dirs into tmp_path and clear job state."""
    for attr in ("DATA_DIR", "REPORTS_DIR"):
        monkeypatch.setattr(app_mod, attr, tmp_path / attr.lower())
    monkeypatch.setattr(app_mod, "LOG_DIR", tmp_path / "logs" / "ui")
    app_mod.JOBS.clear()
    yield tmp_path
    app_mod.JOBS.clear()


@pytest.fixture
def client(clean):
    return app_mod.app.test_client()


def wait_finished(video_id, timeout=5.0) -> dict:
    """The reader thread is async; poll the registry until it settles."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with app_mod.LOCK:
            j = app_mod.JOBS.get(video_id)
            if j and j["status"] != "running":
                return j
        time.sleep(0.02)
    raise AssertionError("job did not settle")


def fake_running_job(video_id):
    proc = FakeProc([], {}, [], 0)
    job = {"video_id": video_id, "source": "x", "args": [], "status": "running",
           "phase": 2, "exit_code": None, "lines": deque(), "started": "",
           "finished": None, "log_path": "unused", "proc": proc}
    app_mod.JOBS[video_id] = job
    return job


# ------------------------------------------------------------ home + listing

class TestHome:
    def test_lists_past_analyses(self, client, clean):
        d = app_mod.REPORTS_DIR / "myvid"
        (d / "frames").mkdir(parents=True)
        (d / "dashboard.html").write_text("<html>dashboard</html>", encoding="utf-8")
        (app_mod.REPORTS_DIR / "not_a_report").mkdir()  # no dashboard -> hidden
        r = client.get("/")
        assert r.status_code == 200
        html = r.get_data(as_text=True)
        assert "myvid" in html
        assert "/download/myvid" in html
        assert "/report/myvid/dashboard.html" in html
        assert "not_a_report" not in html

    def test_empty_listing(self, client, clean):
        html = client.get("/").get_data(as_text=True)
        assert "No analyses yet" in html


# ------------------------------------------------------------------- /start

class TestStart:
    def test_url_mode_spawns_pipeline_with_options(self, client, clean, monkeypatch):
        rec = Recorder(lines=["[info] video_id: rtwpk9rb1Dc",
                              "========== phase 0 =========="])
        monkeypatch.setattr(app_mod, "Popen", rec)
        r = client.post("/start", data={
            "url": "https://www.youtube.com/watch?v=rtwpk9rb1Dc",
            "content_type": "speech",
            "fps": "2", "skip_phase2": "on", "skip_phase5": "on",
            "whisper_model": "small",
        }, content_type="multipart/form-data")
        assert r.status_code == 202
        assert r.get_json()["video_id"] == "rtwpk9rb1Dc"
        job = wait_finished("rtwpk9rb1Dc")
        assert job["status"] == "done" and job["exit_code"] == 0
        argv = rec.calls[0].argv
        assert argv[1].endswith("run_pipeline.py")
        assert "https://www.youtube.com/watch?v=rtwpk9rb1Dc" in argv
        assert argv[argv.index("--content-type") + 1] == "speech"
        assert "--skip-phase2" in argv and "--skip-phase5" in argv
        assert "--fps" not in argv  # default 2 is not passed

    def test_rejects_when_busy(self, client, clean):
        fake_running_job("busyvid")
        r = client.post("/start", data={"url": "https://youtu.be/abc12345678"},
                        content_type="multipart/form-data")
        assert r.status_code == 409
        assert "already running" in r.get_json()["error"]

    def test_url_without_protocol_400(self, client, clean):
        r = client.post("/start", data={"url": "youtube.com/watch?v=x"},
                        content_type="multipart/form-data")
        assert r.status_code == 400

    def test_upload_stages_slugified_copy(self, client, clean, monkeypatch):
        rec = Recorder()
        monkeypatch.setattr(app_mod, "Popen", rec)
        r = client.post("/start", data={
            "video_file": (io.BytesIO(b"0" * 128), "My Clip.MP4"),
        }, content_type="multipart/form-data")
        assert r.status_code == 202
        assert r.get_json()["video_id"] == "my_clip"
        staged = app_mod.DATA_DIR / "raw" / "my_clip.mp4"
        assert staged.is_file() and staged.read_bytes() == b"0" * 128
        source = [a for a in rec.calls[0].argv if a.endswith("my_clip.mp4")]
        assert source and Path(source[0]).is_absolute()
        assert wait_finished("my_clip")["status"] == "done"

    def test_upload_collision_gets_suffix(self, client, clean, monkeypatch):
        monkeypatch.setattr(app_mod, "Popen", Recorder())
        raw = app_mod.DATA_DIR / "raw"
        raw.mkdir(parents=True)
        (raw / "my_clip.mp4").write_bytes(b"old")
        r = client.post("/start", data={
            "video_file": (io.BytesIO(b"new"), "My Clip.mp4"),
        }, content_type="multipart/form-data")
        assert r.get_json()["video_id"] == "my_clip_2"
        assert (raw / "my_clip_2.mp4").is_file()
        assert (raw / "my_clip.mp4").read_bytes() == b"old"  # untouched

    def test_upload_bad_extension_400(self, client, clean):
        r = client.post("/start", data={
            "video_file": (io.BytesIO(b"x"), "notes.txt"),
        }, content_type="multipart/form-data")
        assert r.status_code == 400
        assert ".mp4" in r.get_json()["error"]

    def test_bad_options_400(self, client, clean):
        for form in ({"url": "https://youtu.be/abc12345678", "content_type": "bogus"},
                     {"url": "https://youtu.be/abc12345678", "fps": "0"},
                     {"url": "https://youtu.be/abc12345678", "whisper_model": "giant"}):
            r = client.post("/start", data=form, content_type="multipart/form-data")
            assert r.status_code == 400, form

    def test_custom_id_flows_to_pipeline(self, client, clean, monkeypatch):
        rec = Recorder()
        monkeypatch.setattr(app_mod, "Popen", rec)
        r = client.post("/start", data={
            "url": "https://youtu.be/abc12345678", "custom_id": "demoset",
        }, content_type="multipart/form-data")
        assert r.get_json()["video_id"] == "demoset"
        argv = rec.calls[0].argv
        assert argv[argv.index("--id") + 1] == "demoset"

    def test_bad_custom_id_400(self, client, clean):
        r = client.post("/start", data={
            "url": "https://youtu.be/abc12345678", "custom_id": "../evil",
        }, content_type="multipart/form-data")
        assert r.status_code == 400


# --------------------------------------------------------------- /job + reader

class TestJobStatus:
    def test_streams_phase_and_lines_then_done(self, client, clean, monkeypatch):
        monkeypatch.setattr(app_mod, "Popen", Recorder(lines=[
            "[info] video_id: vid9", "hello from phase 0",
            "========== phase 3 ==========", "[ok] phase 3 finished.",
            "[ok] all phases done.",
        ]))
        client.post("/start", data={"url": "https://youtu.be/abcdefghijk"},
                    content_type="multipart/form-data")
        wait_finished("abcdefghijk")  # id comes from the URL, not the fake log line
        r = client.get("/job/abcdefghijk")  # assert through the API, not the registry
        j = r.get_json()
        assert j["status"] == "done" and j["phase"] == 3
        assert any("hello from phase 0" in ln for ln in j["lines"])

    def test_failed_job_reports_exit_code(self, client, clean, monkeypatch):
        monkeypatch.setattr(app_mod, "Popen", Recorder(
            lines=["========== phase 4 =========="], rc=4))
        client.post("/start", data={"url": "https://youtu.be/abcdefghijk"},
                    content_type="multipart/form-data")
        wait_finished("abcdefghijk")  # reader thread runs async
        r = client.get("/job/abcdefghijk")
        j = r.get_json()
        assert j["status"] == "failed" and j["exit_code"] == 4

    def test_unknown_id_404(self, client, clean):
        r = client.get("/job/never_seen")
        assert r.status_code == 404

    def test_invalid_id_400(self, client, clean):
        assert client.get("/job/bad..id").status_code == 400

    def test_survives_restart_via_dashboard(self, client, clean):
        d = app_mod.REPORTS_DIR / "oldvid"
        d.mkdir(parents=True)
        (d / "dashboard.html").write_text("<html></html>", encoding="utf-8")
        r = client.get("/job/oldvid")
        j = r.get_json()
        assert j["status"] == "done" and j["dashboard_ready"] is True


# ------------------------------------------------------------------ /report

class TestReportServing:
    def _report(self, vid):
        d = app_mod.REPORTS_DIR / vid
        (d / "frames").mkdir(parents=True)
        (d / "dashboard.html").write_text("<html>dashboard</html>", encoding="utf-8")
        (d / "frames" / "f_001.jpg").write_bytes(b"\xff\xd8fakejpg")

    def test_serves_dashboard_and_frames(self, client, clean):
        self._report("vid5")
        r1 = client.get("/report/vid5/dashboard.html")
        r2 = client.get("/report/vid5/frames/f_001.jpg")
        assert r1.status_code == 200 and "dashboard" in r1.get_data(as_text=True)
        assert r2.status_code == 200 and r2.data == b"\xff\xd8fakejpg"

    def test_blocks_traversal(self, client, clean):
        self._report("vid5")
        r = client.get("/report/vid5/frames/../../scripts/app.py")
        assert r.status_code == 404

    def test_invalid_id_400(self, client, clean):
        assert client.get("/report/seg.men/dashboard.html").status_code == 400


# ---------------------------------------------------------------- /download

class TestDownloadZip:
    def _analysis(self, vid):
        rep = app_mod.REPORTS_DIR / vid
        (rep / "frames").mkdir(parents=True)
        (rep / "dashboard.html").write_text("<html>d</html>", encoding="utf-8")
        (rep / "frames" / "f_001.jpg").write_bytes(b"jpg")
        dat = app_mod.DATA_DIR / vid
        (dat / "frames").mkdir(parents=True)  # raw frames must NOT be zipped
        (dat / "sync_per_shot.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        (dat / "audio.wav").write_bytes(b"\0" * 16)  # excluded
        (dat / "metadata.json").write_text("{}", encoding="utf-8")

    def test_zip_layout_keeps_dashboard_offline(self, client, clean):
        self._analysis("zipvid")
        r = client.get("/download/zipvid")
        assert r.status_code == 200
        assert r.mimetype == "application/zip"
        names = zipfile.ZipFile(io.BytesIO(r.data)).namelist()
        assert "zipvid/dashboard.html" in names
        assert "zipvid/frames/f_001.jpg" in names
        assert "zipvid/data/sync_per_shot.csv" in names
        assert "zipvid/data/metadata.json" in names
        assert not any(n.endswith("audio.wav") for n in names)
        assert not any(n.endswith("frames/frame") for n in names)

    def test_404_when_no_dashboard(self, client, clean):
        assert client.get("/download/zipvid").status_code == 404

    def test_409_while_running(self, client, clean):
        self._analysis("zipvid")
        fake_running_job("zipvid")
        assert client.get("/download/zipvid").status_code == 409


# ------------------------------------------------------------------ /cancel

class TestCancel:
    def test_cancels_running_job(self, client, clean, monkeypatch):
        killed = []
        monkeypatch.setattr("subprocess.run",
                            lambda *a, **k: killed.append(a[0]) if a else None)
        job = fake_running_job("runvid")
        r = client.post("/job/runvid/cancel")
        assert r.status_code == 200
        with app_mod.LOCK:
            assert app_mod.JOBS["runvid"]["status"] == "cancelled"
        # Windows tree-kill via taskkill; POSIX would call proc.terminate()
        assert killed and killed[0][:4] == ["taskkill", "/F", "/T", "/PID"]
        assert killed[0][4] == str(job["proc"].pid)

    def test_no_running_job_404(self, client, clean):
        assert client.post("/job/missing/cancel").status_code == 404