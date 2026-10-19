"""
Local web UI for the multimodal analysis pipeline.

Runs entirely on the user's machine (binds 127.0.0.1) and wraps the existing
CLI pipeline (`run_pipeline.py`) as a background job per analysis:

  python scripts/app.py            # open http://127.0.0.1:5000
  python scripts/app.py --port 8080

Features:
  - start an analysis from a YouTube URL *or* an uploaded video file
  - live progress (phase banner + pipeline log) while the job runs
  - open the rendered dashboard in the browser
  - "save the analysis" = download a ZIP (dashboard + frames + data CSVs/JSONs)
  - home page lists past analyses found under reports/

The app never runs model code itself: each job is a subprocess of
run_pipeline.py, so torch/CLAP/whisper stay out of this process and the UI
survives pipeline crashes. Job state is in-memory (single job at a time —
CLAP/Whisper are heavyweight); a finished dashboard on disk still shows up
after an app restart.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import subprocess
import sys
import threading
import time
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from flask import Flask, Response, jsonify, render_template_string, request, send_file
from werkzeug.utils import safe_join

SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
from run_pipeline import derive_video_id  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
LOG_DIR = REPO_ROOT / "logs" / "ui"

VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov", ".avi")
CONTENT_TYPES = ("auto", "song", "speech", "film", "series", "vlog", "other")
WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3")
MAX_UPLOAD = 2 * 1024 ** 3  # 2 GB upload cap (Flask auto-413s beyond this)

VID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
PHASE_RE = re.compile(r"========== phase (\d+) ==========")

Popen = subprocess.Popen  # module alias so tests can monkeypatch the launcher

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD

JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


# --------------------------------------------------------------------------- jobs

def any_running() -> bool:
    with LOCK:
        return any(j["status"] == "running" for j in JOBS.values())


def phase_n(job) -> int | None:
    with LOCK:
        return job.get("phase")


def _kill_tree(proc) -> None:
    """Kill the pipeline and its children (yt-dlp/ffmpeg spawn underneath)."""
    if os.name == "nt":
        # TerminateProcess on the cmd wrapper would orphan grandchildren.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    else:
        proc.terminate()


def _reader(job: dict, proc) -> None:
    """Daemon thread: tail subprocess stdout into the job's ring buffer + log."""
    log_path = Path(job["log_path"])
    try:
        with log_path.open("w", encoding="utf-8") as fh:
            for raw in proc.stdout:
                line = raw.rstrip("\r\n")
                with LOCK:
                    job["lines"].append(line)
                    fh.write(line + "\n")
                    m = PHASE_RE.search(line)
                    if m:
                        job["phase"] = int(m.group(1))
    finally:
        rc = proc.wait()
        with LOCK:
            if job["status"] == "running":  # a cancel keeps "cancelled"
                job["status"] = "done" if rc == 0 else "failed"
            job["exit_code"] = rc
            job["finished"] = datetime.now(timezone.utc).isoformat()


def start_job(video_id: str, source: str, pipeline_args: list[str]) -> dict:
    """Launch run_pipeline.py as a subprocess; returns the job dict."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = Path(LOG_DIR) / f"{video_id}.log"
    job = {
        "video_id": video_id,
        "source": source,
        "args": pipeline_args,
        "status": "running",
        "phase": None,
        "exit_code": None,
        "lines": deque(maxlen=400),
        "started": datetime.now(timezone.utc).isoformat(),
        "finished": None,
        "log_path": str(log_path),
        "proc": None,
    }
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    argv = [sys.executable, str(SCRIPTS / "run_pipeline.py"), *pipeline_args]
    kwargs = dict(cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                  stderr=subprocess.STDOUT, text=True,
                  encoding="utf-8", errors="replace", env=env)
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = Popen(argv, **kwargs)
    job["proc"] = proc
    with LOCK:
        JOBS[video_id] = job
    threading.Thread(target=_reader, args=(job, proc), daemon=True).start()
    return job


# ------------------------------------------------------------------- validation

def bad(msg: str, status: int = 400):
    return jsonify({"error": msg}), status


def valid_id_or_none(video_id):
    return video_id if (video_id and VID_RE.fullmatch(video_id)) else None


def parse_options(form) -> tuple[list[str], str | None]:
    """Validate UI options -> pipeline args. Returns (args, error)."""
    args: list[str] = []

    ctype = form.get("content_type", "auto")
    if ctype not in CONTENT_TYPES:
        return [], f"unknown content type '{ctype}'"
    if ctype != "auto":
        args += ["--content-type", ctype]

    fps = form.get("fps", "2")
    try:
        fps_n = int(fps)
        assert 1 <= fps_n <= 10
    except (ValueError, AssertionError):
        return [], f"fps must be an integer 1-10 (got '{fps}')"
    if fps_n != 2:
        args += ["--fps", str(fps_n)]

    if form.get("skip_phase2"):
        args.append("--skip-phase2")
    elif form.get("mock_vision"):
        args += ["--provider", "mock"]  # offline rehearsal: no API calls, labelled MOCK
    if form.get("skip_phase5"):
        args.append("--skip-phase5")

    wmodel = form.get("whisper_model", "small")
    if wmodel not in WHISPER_MODELS:
        return [], f"unknown whisper model '{wmodel}'"
    if wmodel != "small":
        args += ["--whisper-model", wmodel]

    lang = (form.get("whisper_language") or "").strip()
    if lang:
        if not re.fullmatch(r"[A-Za-z]{2,5}", lang):
            return [], f"whisper language must be a short code like 'en' (got '{lang}')"
        args += ["--whisper-language", lang]

    return args, None


def stage_upload(f) -> tuple[str, str] | None:
    """Save an uploaded video into data/raw/<id><ext> (slug + collision suffix).

    Returns the staged absolute path to pass to run_pipeline (its own
    derive_video_id turns the filename's stem into the same id).
    """
    filename = Path(f.filename or "")
    ext = filename.suffix.lower()
    if ext not in VIDEO_EXTS:
        return None
    stem = derive_video_id(f.filename, None)
    if not stem:
        return None
    raw = DATA_DIR / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    video_id, n = stem, 1
    while (raw / f"{video_id}{ext}").exists():
        n += 1
        video_id = f"{stem}_{n}"
    dest = raw / f"{video_id}{ext}"
    f.save(str(dest))
    return str(dest.resolve()), video_id


# ------------------------------------------------------------------------ routes

def _dashboard_ready(video_id: str) -> bool:
    return (REPORTS_DIR / video_id / "dashboard.html").is_file()


def list_past_analyses() -> list[dict]:
    out = []
    if REPORTS_DIR.is_dir():
        for d in REPORTS_DIR.iterdir():
            if d.is_dir() and VID_RE.fullmatch(d.name) and (d / "dashboard.html").is_file():
                out.append({
                    "id": d.name,
                    "mtime": datetime.fromtimestamp((d / "dashboard.html").stat().st_mtime,
                                                    tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                })
    return sorted(out, key=lambda a: a["mtime"], reverse=True)


@app.get("/")
def index():
    return render_template_string(PAGE_HTML, analyses=list_past_analyses())


@app.post("/start")
def start():
    if any_running():
        return bad("a job is already running — cancel it first", 409)

    opts, err = parse_options(request.form)
    if err:
        return bad(err)

    upload = request.files.get("video_file")
    if upload and upload.filename:
        staged = stage_upload(upload)
        if staged is None:
            return bad(f"upload must be one of {', '.join(VIDEO_EXTS)}")
        source, video_id = staged
    else:
        source = (request.form.get("url") or "").strip()
        if not source:
            return bad("provide a YouTube URL or an uploaded video file")
        if not source.startswith(("http://", "https://")):
            return bad("URL must start with http:// or https://")
        video_id = derive_video_id(source, None)

    custom = (request.form.get("custom_id") or "").strip()
    if custom:
        if not VID_RE.fullmatch(custom):
            return bad("id may contain only letters, digits, '_' and '-' (max 64)")
        video_id = custom

    with LOCK:  # re-check under lock (two submits could race past any_running)
        if any(j["status"] == "running" for j in JOBS.values()):
            return bad("a job is already running — cancel it first", 409)

    argv = [source, *opts]
    if custom:
        argv += ["--id", custom]  # run_pipeline's own derive must agree with ours
    start_job(video_id, source, argv)
    return jsonify({"video_id": video_id, "status": "running"}), 202


@app.get("/job/<video_id>")
def job(video_id: str):
    if not VID_RE.fullmatch(video_id):
        return bad("invalid id")
    snap = None
    with LOCK:
        j = JOBS.get(video_id)
        if j:
            snap = {k: j[k] for k in ("video_id", "status", "phase", "exit_code")}
            snap["lines"] = list(j["lines"])[-30:]
    if snap is None:
        # survive an app restart: a dashboard on disk means the analysis is done
        if _dashboard_ready(video_id):
            snap = {"video_id": video_id, "status": "done", "phase": 8,
                    "exit_code": 0, "lines": []}
        else:
            return bad("no such analysis", 404)
    snap["dashboard_ready"] = _dashboard_ready(video_id)
    return jsonify(snap)


@app.post("/job/<video_id>/cancel")
def cancel(video_id: str):
    if not VID_RE.fullmatch(video_id):
        return bad("invalid id")
    with LOCK:
        j = JOBS.get(video_id)
        if not j or j["status"] != "running":
            return bad("no running job for this id", 404)
        j["status"] = "cancelled"
        proc = j["proc"]
    _kill_tree(proc)
    return jsonify({"video_id": video_id, "status": "cancelled"})


@app.get("/report/<video_id>/<path:rel>")
def report(video_id: str, rel: str):
    if not VID_RE.fullmatch(video_id):
        return bad("invalid id")
    target = safe_join(str(REPORTS_DIR), video_id, rel)
    if target is None or not Path(target).is_file():
        return Response("not found", status=404)
    return send_file(target)


@app.get("/download/<video_id>")
def download(video_id: str):
    if not VID_RE.fullmatch(video_id):
        return bad("invalid id")
    if not _dashboard_ready(video_id):
        return bad("no finished analysis for this id", 404)
    with LOCK:
        j = JOBS.get(video_id)
        if j and j["status"] == "running":
            return bad("job still running — download when it finishes", 409)

    rep = REPORTS_DIR / video_id
    dat = DATA_DIR / video_id
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(rep / "dashboard.html", f"{video_id}/dashboard.html")
        fdir = rep / "frames"
        if fdir.is_dir():
            for p in fdir.rglob("*"):
                if p.is_file():
                    zf.write(p, f"{video_id}/frames/{p.relative_to(fdir).as_posix()}")
        if dat.is_dir():
            for p in sorted(dat.glob("*")):
                if p.is_file() and p.suffix.lower() in (".csv", ".json"):
                    zf.write(p, f"{video_id}/data/{p.name}")
    buf.seek(0)
    fname = quote(f"{video_id}_analysis.zip")
    return Response(buf.getvalue(), mimetype="application/zip", headers={
        "Content-Disposition": f"attachment; filename*=UTF-8''{fname}"})


# ----------------------------------------------------------------------- page

PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Multimodal Video Analysis</title>
<style>
  :root { --ink:#1c2230; --muted:#6b7385; --line:#dfe3ec; --bg:#f5f6fa;
          --card:#ffffff; --accent:#2f6fed; --ok:#1d8a4e; --bad:#c23b3b; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.5 system-ui, "Segoe UI", sans-serif; }
  main { max-width: 860px; margin: 0 auto; padding: 24px 16px 64px; }
  h1 { font-size: 1.45rem; margin: 0 0 4px; }
  .sub { color: var(--muted); margin: 0 0 20px; font-size: .92rem; }
  .card { background: var(--card); border: 1px solid var(--line);
          border-radius: 10px; padding: 18px 20px; margin-bottom: 18px; }
  .tabs { display:flex; gap:6px; margin-bottom:14px; }
  .tab { padding:7px 14px; border:1px solid var(--line); border-radius:8px;
         background:#fff; cursor:pointer; font-size:.92rem; }
  .tab.on { background: var(--accent); border-color: var(--accent); color:#fff; }
  label { display:block; font-size:.85rem; color:var(--muted);
          margin:10px 0 3px; }
  input[type=text], input[type=url], select {
      width:100%; padding:8px 10px; border:1px solid var(--line);
      border-radius:7px; font:inherit; }
  .opts { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
          gap:0 16px; margin-top:6px; }
  .checks { display:flex; gap:18px; margin-top:10px; font-size:.92rem; }
  .checks label { display:flex; align-items:center; gap:6px; margin:0;
                  color:var(--ink); }
  button.go { margin-top:16px; background:var(--accent); color:#fff;
              border:0; border-radius:8px; padding:10px 22px; font:inherit;
              font-weight:600; cursor:pointer; }
  button.go:disabled { opacity:.5; cursor:not-allowed; }
  button.cancel { background:#fff; color:var(--bad); border:1px solid var(--bad);
                  border-radius:8px; padding:8px 16px; font:inherit;
                  cursor:pointer; margin-left:8px; }
  #prog { display:none; }
  #prog .bar { height:8px; border-radius:6px; background:var(--line);
               overflow:hidden; margin:10px 0 6px; }
  #prog .fill { height:100%; width:0%; background:var(--accent);
                transition: width .3s; }
  .phase { font-weight:700; }
  .stat-running { color:var(--accent); }
  .stat-done { color:var(--ok); }
  .stat-failed, .stat-cancelled { color:var(--bad); }
  #log { background:#0e1420; color:#c8d2e4; font:12px/1.45 Consolas, monospace;
         border-radius:8px; padding:12px; max-height:280px; overflow-y:auto;
         white-space:pre-wrap; word-break:break-word; margin-top:8px; }
  .btns a { display:inline-block; margin-right:10px; margin-top:10px;
            background:#fff; border:1px solid var(--accent); color:var(--accent);
            padding:8px 14px; border-radius:8px; text-decoration:none;
            font-size:.92rem; cursor:pointer; }
  .btns a.solid { background:var(--accent); color:#fff; }
  ul.an { margin:0; padding-left:4px; list-style:none; }
  ul.an li { padding:9px 0; border-bottom:1px solid var(--line);
             display:flex; gap:12px; align-items:baseline; }
  ul.an li:last-child { border-bottom:0; }
  ul.an .when { color:var(--muted); font-size:.85rem; }
  ul.an .dl { margin-left:auto; font-size:.88rem; }
  .none { color:var(--muted); margin:4px 0; }
  .msg { margin-top:12px; font-size:.92rem; min-height:1.2em; }
  .msg.err { color: var(--bad); }
</style>
</head>
<body>
<main>
  <h1>Multimodal Video Analysis</h1>
  <p class="sub">Runs the full 8-phase pipeline locally — nothing leaves your machine
     except the vision-model API calls you configured in <code>.env</code>.</p>

  <div class="card">
    <div class="tabs">
      <button type="button" class="tab on" id="tab-url">YouTube URL</button>
      <button type="button" class="tab" id="tab-upload">Upload file</button>
    </div>
    <form id="form">
      <div id="row-url">
        <label for="url">Video URL</label>
        <input type="url" id="url" name="url"
               placeholder="https://www.youtube.com/watch?v=...">
      </div>
      <div id="row-upload" style="display:none">
        <label for="video_file">Video file (mp4 / webm / mkv / mov / avi)</label>
        <input type="file" id="video_file" name="video_file"
               accept=".mp4,.webm,.mkv,.mov,.avi,video/*">
        <div class="bar" style="margin-top:8px;height:6px">
          <div class="fill" id="upbar" style="background:var(--ok)"></div>
        </div>
      </div>
      <div class="opts">
        <div>
          <label for="content_type">Content type</label>
          <select id="content_type" name="content_type">
            <option value="auto" selected>auto-detect</option>
            <option value="song">song</option>
            <option value="speech">speech</option>
            <option value="film">film</option>
            <option value="series">series</option>
            <option value="vlog">vlog</option>
            <option value="other">other</option>
          </select>
        </div>
        <div>
          <label for="fps">Frame extraction fps</label>
          <input type="text" id="fps" name="fps" value="2">
        </div>
        <div>
          <label for="whisper_model">Whisper model</label>
          <select id="whisper_model" name="whisper_model">
            <option value="tiny">tiny</option>
            <option value="base">base</option>
            <option value="small" selected>small</option>
            <option value="medium">medium</option>
            <option value="large-v3">large-v3</option>
          </select>
        </div>
        <div>
          <label for="whisper_language">Transcription language (optional)</label>
          <input type="text" id="whisper_language" name="whisper_language"
                 placeholder="e.g. en, de, ar">
        </div>
        <div>
          <label for="custom_id">Custom analysis ID (optional)</label>
          <input type="text" id="custom_id" name="custom_id"
                 placeholder="auto from URL / filename">
        </div>
      </div>
      <div class="checks">
        <label><input type="checkbox" name="skip_phase2"> Skip vision captions (phase 2)</label>
        <label><input type="checkbox" name="mock_vision"> Test run: fake vision answers, no API cost</label>
        <label><input type="checkbox" name="skip_phase5"> Skip audio tagging (phase 5)</label>
      </div>
      <button type="submit" class="go" id="go">Analyze</button>
      <span class="msg" id="msg"></span>
    </form>
  </div>

  <div class="card" id="prog">
    <div><span class="phase" id="phase">starting…</span>
         <span id="status" class="stat-running"></span>
         <button type="button" class="cancel" id="cancel" style="float:right">
             Cancel</button></div>
    <div class="bar"><div class="fill" id="fill"></div></div>
    <div id="log"></div>
    <div class="msg" id="donebox"></div>
    <div class="btns" id="btns" style="display:none">
      <a class="solid" id="openDash" target="_blank">Open dashboard</a>
      <a id="zipLink">Download analysis (ZIP)</a>
      <a href="/">New analysis</a>
    </div>
  </div>

  <div class="card">
    <h2 style="font-size:1.05rem;margin:0 0 10px">Saved analyses</h2>
    {% if analyses %}
    <ul class="an">
      {% for a in analyses %}
      <li>
        <b>{{ a.id }}</b>
        <span class="when">{{ a.mtime }} UTC</span>
        <span class="dl">
          <a href="/report/{{ a.id }}/dashboard.html" target="_blank">dashboard</a> ·
          <a href="/download/{{ a.id }}">ZIP</a>
        </span>
      </li>
      {% endfor %}
    </ul>
    {% else %}
    <p class="none">No analyses yet — start one above.</p>
    {% endif %}
  </div>
</main>
<script>
const $ = (id) => document.getElementById(id);
const form = $("form");
let mode = "url", pollTimer = null, currentJob = null, finishedStatus = null;

function anyRunning() {
  return currentJob !== null && finishedStatus === null;
}

function setMode(m) {
  mode = m;
  $("tab-url").classList.toggle("on", m === "url");
  $("tab-upload").classList.toggle("on", m === "upload");
  $("row-url").style.display = m === "url" ? "" : "none";
  $("row-upload").style.display = m === "upload" ? "" : "none";
}
$("tab-url").onclick = () => setMode("url");
$("tab-upload").onclick = () => setMode("upload");

function msg(t, err) { const el = $("msg"); el.textContent = t || "";
  el.className = "msg" + (err ? " err" : ""); }

function started(j) {
  currentJob = j.video_id;
  finishedStatus = null;
  $("prog").style.display = "";
  $("go").disabled = true;
  $("btns").style.display = "none";
  $("donebox").textContent = "";
  $("upbar").style.width = "100%";
  $("phase").textContent = "starting…";
  poll();
  pollTimer = setInterval(poll, 2000);
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  msg("");
  if (anyRunning()) { msg("A job is already running — cancel it first.", true); return; }
  const fd = new FormData(form);
  if (mode === "upload" && !$("video_file").files.length) {
    msg("Choose a video file first.", true); return;
  }
  $("go").disabled = true;
  $("prog").style.display = mode === "upload" ? "" : "none";
  if (mode === "upload") {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/start");
    xhr.upload.onprogress = (ev) => {
      if (ev.lengthComputable)
        $("upbar").style.width = Math.round(ev.loaded / ev.total * 100) + "%";
    };
    xhr.onload = () => {
      if (xhr.status === 202) started(JSON.parse(xhr.responseText));
      else { $("go").disabled = false; msg(uploadErrMsg(xhr), true); }
    };
    xhr.onerror = () => { $("go").disabled = false; msg("Upload failed.", true); };
    xhr.send(fd);
    return;
  }
  fetch("/start", { method: "POST", body: fd }).then(async (r) => {
    const body = await r.json();
    if (r.status === 202) started(body);
    else { $("go").disabled = false; msg(body.error || "Could not start.", true); }
  });
});

function uploadErrMsg(xhr) {
  try { return JSON.parse(xhr.responseText).error || "Start failed."; }
  catch (e) { return xhr.status === 413 ? "File too large (2 GB limit)." : "Start failed."; }
}

function poll() {
  if (!currentJob) return;
  fetch("/job/" + encodeURIComponent(currentJob)).then((r) => {
    if (!r.ok) throw new Error("poll");
    return r.json();
  }).then((j) => {
    $("phase").textContent = j.phase ? ("Phase " + j.phase + " / 8") : "starting…";
    $("status").textContent = j.status;
    $("status").className = "stat-" + j.status;
    $("log").textContent = (j.lines || []).join("\n");
    $("log").scrollTop = $("log").scrollHeight;
    if (j.status !== "running") finished(j);
  }).catch(() => {});
}

function finished(j) {
  finishedStatus = j.status;
  clearInterval(pollTimer);
  $("go").disabled = false;
  if (j.status === "done") {
    $("phase").textContent = "Analysis complete";
    $("fill").style.width = "100%";
    $("btns").style.display = "";
    $("openDash").href = "/report/" + encodeURIComponent(currentJob) + "/dashboard.html";
    $("zipLink").href = "/download/" + encodeURIComponent(currentJob);
    refreshSaved();
  } else {
    $("phase").textContent = j.status === "cancelled" ? "Cancelled" : "Failed";
    $("donebox").textContent = j.status === "failed"
      ? ("Pipeline exited with code " + j.exit_code + " (that phase number failed — see log above).")
      : "Run cancelled.";
    refreshSaved();
  }
}

$("cancel").onclick = () => {
  if (!currentJob) return;
  fetch("/job/" + encodeURIComponent(currentJob) + "/cancel", { method: "POST" })
    .then(() => poll());
};

function refreshSaved() { fetch("/").then(r => r.text()).then((html) => {
  const doc = new DOMParser().parseFromString(html, "text/html");
  const cards = doc.querySelectorAll(".card");
  document.querySelectorAll(".card")[2].replaceWith(cards[2]);
}); }
</script>
</body>
</html>"""


# ------------------------------------------------------------------------ main

def create_app() -> Flask:  # test hook: a fresh app sharing the same module state
    return app


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Local web UI for the analysis pipeline")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="port (default 5000)")
    args = parser.parse_args(argv)
    print(f"[info] web UI: http://{args.host}:{args.port}  (Ctrl+C to stop)")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())