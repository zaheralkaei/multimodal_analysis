"""
Fake cloud — a local stand-in for fal, Groq, Replicate and OpenRouter (round 12).

Purpose: run the REAL client code paths (uploads, multipart forms, queue
polling, downloads, caching, cost logs, every phase that consumes the
results) end to end without spending anything, before the first paid run.
It implements only what the pipeline uses, following each provider's
documented request/response format, and it REJECTS malformed requests
(wrong auth header, missing form fields, wrong JSON keys) with HTTP 400 —
so a formatting bug fails here instead of on a paid call.

    python scripts/fake_cloud.py --port 8765          # standalone
    python scripts/calibration.py run --fake-cloud     # whole pipeline against it

Endpoints (base http://127.0.0.1:<port>):
  /fal/...          fal queue API for fal-ai/demucs
  /groq/...         Groq OpenAI-compatible transcription
  /replicate/...    Replicate models / files / predictions
  /openrouter/...   OpenRouter models + chat/completions

Faults (comma list in FAKE_CLOUD_FAULTS or --faults), for robustness tests:
  fal_error          fal job ends COMPLETED with an error field
  groq_500_once      first Groq request returns HTTP 500
  replicate_failed   Replicate prediction ends 'failed'
  openrouter_429_once  first chat completion returns HTTP 429
  openrouter_garbage   chat completions return non-JSON text
"""
from __future__ import annotations
import argparse, base64, io, json, os, re, sys, threading
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

FAKE_KEYS = {"fal": "fake-fal-key", "groq": "fake-groq-key", "replicate": "r8_fake",
             "openrouter": "sk-or-fake"}


class State:
    def __init__(self, faults: set, vocals_silence_before: float, speech: dict | None):
        self.faults = faults
        self.silence_before = vocals_silence_before
        self.speech = speech          # {"start","end","text"} for canned transcription
        self.counts: dict[str, int] = {}
        self.uploads: dict[str, bytes] = {}
        self.lock = threading.Lock()
        self.log: list[str] = []

    def hit(self, key: str) -> int:
        with self.lock:
            self.counts[key] = self.counts.get(key, 0) + 1
            return self.counts[key]


def _audio_bytes_from(src: str, state: State) -> bytes:
    """The uploaded mix (data URI or an earlier upload URL) as audio bytes."""
    if src.startswith("data:"):
        return base64.b64decode(src.split(",", 1)[1])
    m = re.search(r"/files/([\w.-]+)$", src)
    if m and m.group(1) in state.uploads:
        return state.uploads[m.group(1)]
    raise ValueError("unknown audio url")


def _fake_vocals(audio: bytes, silence_before: float) -> bytes:
    """'Separated vocals' = the mix with everything before `silence_before`
    seconds removed (the calibration song has no voice there)."""
    import numpy as np
    import soundfile as sf
    y, sr = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    y[: int(silence_before * sr)] = 0.0
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def make_handler(state: State, base: str):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        # ---------------- helpers
        def _send(self, code: int, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def _bad(self, msg):
            state.log.append(f"400 {self.path}: {msg}")
            self._send(400, {"error": msg})

        def _auth(self, expected: str) -> bool:
            if self.headers.get("Authorization") != expected:
                self._bad(f"bad Authorization header (expected '{expected.split()[0]} ...')")
                return False
            return True

        def _multipart(self, body: bytes) -> dict:
            ctype = self.headers.get("Content-Type", "")
            if not ctype.startswith("multipart/form-data"):
                return {}
            msg = BytesParser(policy=email_policy).parsebytes(
                b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + body)
            fields: dict = {}
            for part in msg.iter_parts():
                name = part.get_param("name", header="content-disposition")
                payload = part.get_payload(decode=True)
                entry = {"filename": part.get_filename(), "type": part.get_content_type(),
                         "data": payload}
                fields.setdefault(name, []).append(entry)
            return fields

        # ---------------- routing
        def do_GET(self):
            p = self.path.split("?")[0]
            if p.startswith("/fal/"):
                return self.fal_get(p)
            if p.startswith("/replicate/"):
                return self.replicate_get(p)
            if p.startswith("/openrouter/models"):
                return self.openrouter_models()
            if p.startswith("/files/"):
                name = p.rsplit("/", 1)[-1]
                if name in state.uploads:
                    return self._send(200, state.uploads[name], "application/octet-stream")
            self._send(404, {"error": "not found"})

        def do_POST(self):
            p = self.path.split("?")[0]
            body = self._body()
            if p == "/fal/fal-ai/demucs":
                return self.fal_submit(body)
            if p == "/groq/audio/transcriptions":
                return self.groq(body)
            if p.startswith("/replicate/"):
                return self.replicate_post(p, body)
            if p == "/openrouter/chat/completions":
                return self.openrouter_chat(body)
            self._send(404, {"error": "not found"})

        # ---------------- fal (queue API)
        def fal_submit(self, body):
            if not self._auth(f"Key {FAKE_KEYS['fal']}"):
                return
            try:
                req = json.loads(body)
            except ValueError:
                return self._bad("body is not JSON")
            for k in ("audio_url", "model", "stems"):
                if k not in req:
                    return self._bad(f"missing {k}")
            if req["model"] not in ("htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi",
                                    "mdx", "mdx_extra", "mdx_q", "mdx_extra_q"):
                return self._bad("unknown model")
            try:
                audio = _audio_bytes_from(req["audio_url"], state)
            except ValueError as e:
                return self._bad(str(e))
            rid = f"req{state.hit('fal_submit')}"
            state.uploads[f"{rid}_vocals.wav"] = _fake_vocals(audio, state.silence_before)
            app = f"{base}/fal/fal-ai/demucs"          # status URLs are owner/app scoped
            self._send(200, {"request_id": rid, "status_url": f"{app}/requests/{rid}/status",
                             "response_url": f"{app}/requests/{rid}",
                             "cancel_url": f"{app}/requests/{rid}/cancel", "queue_position": 0})

        def fal_get(self, p):
            if not self._auth(f"Key {FAKE_KEYS['fal']}"):
                return
            m = re.match(r"/fal/fal-ai/demucs/requests/(\w+)(/status)?$", p)
            if not m:
                return self._send(405, {"detail": "Method Not Allowed"})
            rid = m.group(1)
            if m.group(2):
                n = state.hit(f"fal_status_{rid}")
                if n == 1:
                    return self._send(200, {"status": "IN_PROGRESS"})
                if "fal_error" in state.faults:
                    return self._send(200, {"status": "COMPLETED", "error": "audio could not be decoded",
                                            "error_type": "input_error"})
                return self._send(200, {"status": "COMPLETED", "metrics": {"inference_time": 1.0}})
            self._send(200, {"vocals": {"url": f"{base}/files/{rid}_vocals.wav",
                                        "content_type": "audio/wav", "file_name": "vocals.wav"}})

        # ---------------- Groq
        def groq(self, body):
            if not self._auth(f"Bearer {FAKE_KEYS['groq']}"):
                return
            if "groq_500_once" in state.faults and state.hit("groq_500") == 1:
                return self._send(500, {"error": {"message": "internal"}})
            f = self._multipart(body)
            for k in ("file", "model", "response_format"):
                if k not in f:
                    return self._bad(f"missing form field {k}")
            if f["model"][0]["data"].decode() not in ("whisper-large-v3", "whisper-large-v3-turbo"):
                return self._bad("unknown model")
            if f["response_format"][0]["data"].decode() != "verbose_json":
                return self._bad("timestamps need response_format=verbose_json")
            grans = {g["data"].decode() for g in f.get("timestamp_granularities[]", [])}
            if not grans <= {"word", "segment"}:
                return self._bad("bad timestamp_granularities[]")
            file = f["file"][0]
            if not file["filename"]:
                return self._bad("file part needs a filename")
            import soundfile as sf
            try:
                info = sf.info(io.BytesIO(file["data"]))
            except Exception:
                return self._bad("could not decode audio")
            dur = info.frames / info.samplerate
            segs, words = [], []
            sp = state.speech
            if sp and dur >= sp["start"] + 1:
                s, e = sp["start"] + 0.2, min(sp["end"], dur) - 0.3
                segs.append({"id": 0, "seek": 0, "start": s, "end": e, "text": " " + sp["text"],
                             "tokens": [], "temperature": 0.0, "avg_logprob": -0.15,
                             "compression_ratio": 1.3, "no_speech_prob": 0.01})
                ws = sp["text"].split()
                step = (e - s) / max(1, len(ws))
                words = [{"word": w, "start": round(s + i * step, 2), "end": round(s + (i + 1) * step, 2)}
                         for i, w in enumerate(ws)]
            self._send(200, {"task": "transcribe", "language": "english", "duration": dur,
                             "text": " ".join(x["text"] for x in segs), "segments": segs,
                             "words": words if "word" in grans else None})

        # ---------------- Replicate
        def replicate_get(self, p):
            if not self._auth(f"Bearer {FAKE_KEYS['replicate']}"):
                return
            if p == "/replicate/models/sakemin/all-in-one-music-structure-analyzer":
                return self._send(200, {"latest_version": {"id": "f" * 64, "openapi_schema": {
                    "components": {"schemas": {"Input": {"properties": {
                        "music_input": {"type": "string", "format": "uri"},
                        "visualize": {"type": "boolean", "default": False}}}}}}}})
            m = re.match(r"/replicate/predictions/(\w+)$", p)
            if m:
                n = state.hit(f"pred_{m.group(1)}")
                if n == 1:
                    return self._send(200, {"id": m.group(1), "status": "processing",
                                            "urls": {"get": f"{base}{p}"}})
                if "replicate_failed" in state.faults:
                    return self._send(200, {"id": m.group(1), "status": "failed",
                                            "error": "CUDA out of memory"})
                return self._send(200, {"id": m.group(1), "status": "succeeded",
                                        "output": [f"{base}/replicate/files/viz.png",
                                                   f"{base}/replicate/files/analysis.json"]})
            if p == "/replicate/files/analysis.json":
                return self._send(200, state.uploads.get("allin1", {}))
            self._send(404, {"detail": "not found"})

        def replicate_post(self, p, body):
            if not self._auth(f"Bearer {FAKE_KEYS['replicate']}"):
                return
            if p == "/replicate/files":
                f = self._multipart(body)
                if "content" not in f or not f["content"][0]["filename"]:
                    return self._bad("multipart needs a 'content' part with a filename")
                name = f"up{state.hit('rep_file')}.mp3"
                state.uploads[name] = f["content"][0]["data"]
                return self._send(201, {"id": name, "urls": {"get": f"{base}/files/{name}"}})
            if p == "/replicate/predictions":
                try:
                    req = json.loads(body)
                except ValueError:
                    return self._bad("body is not JSON")
                if req.get("version") != "f" * 64:
                    return self._bad("unknown version")
                url = (req.get("input") or {}).get("music_input", "")
                if not url.startswith(f"{base}/files/"):
                    return self._bad("music_input must be an uploaded file URL")
                state.uploads["allin1"] = json.loads(json.dumps(_fake_allin1()))
                pid = f"p{state.hit('pred')}"
                return self._send(201, {"id": pid, "status": "starting",
                                        "urls": {"get": f"{base}/replicate/predictions/{pid}"}})
            self._send(404, {"detail": "not found"})

        # ---------------- OpenRouter
        def openrouter_models(self):
            self._send(200, {"data": [{
                "id": "google/gemma-4-31b-it",
                "architecture": {"input_modalities": ["image", "text", "video"]},
                "pricing": {"prompt": "0.0000002", "completion": "0.0000004"},
                "supported_parameters": ["temperature", "seed", "response_format", "max_tokens", "stop"]}]})

        def openrouter_chat(self, body):
            if not self._auth(f"Bearer {FAKE_KEYS['openrouter']}"):
                return
            if "openrouter_429_once" in state.faults and state.hit("or429") == 1:
                return self._send(429, {"error": {"message": "rate limited"}})
            try:
                req = json.loads(body)
                content = req["messages"][0]["content"]
                img = next(c for c in content if c.get("type") == "image_url")
                assert img["image_url"]["url"].startswith("data:image/jpeg;base64,")
            except Exception:
                return self._bad("expected one user message with text + image_url")
            if "openrouter_garbage" in state.faults:
                text = "Sorry, I cannot help with that."
            elif len(content[0].get("text", "")) < 80:
                text = "ready"
            else:
                h = state.hit("or_chat")
                text = json.dumps({"caption": f"FAKE-CLOUD caption {h}", "camera": "static",
                                   "emotion": "joyful", "colors": "red, blue, white",
                                   "entities": "person", "location": "indoor studio",
                                   "lighting": "bright", "composition": "medium shot"})
            self._send(200, {"choices": [{"message": {"content": text}}],
                             "usage": {"prompt_tokens": 900, "completion_tokens": 80, "cost": 0.0002}})
    return H


def _fake_allin1() -> dict:
    """What All-In-One returns for the calibration song (120 BPM, 2 sections)."""
    beats = [round(k * 0.5, 3) for k in range(40)]
    return {"path": "mix.mp3", "bpm": 120, "beats": beats, "downbeats": beats[::4],
            "beat_positions": [k % 4 + 1 for k in range(40)],
            "segments": [{"start": 0.0, "end": 10.0, "label": "verse"},
                         {"start": 10.0, "end": 20.0, "label": "chorus"},
                         {"start": 20.0, "end": 36.0, "label": "end"}]}


def start(port: int = 0, faults: set | None = None, vocals_silence_before: float = 0.0,
          speech: dict | None = None):
    """Start in a daemon thread. Returns (server, base_url, state, env_overrides)."""
    state = State(set(faults or ()), vocals_silence_before, speech)
    srv = ThreadingHTTPServer(("127.0.0.1", port), None)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    srv.RequestHandlerClass = make_handler(state, base)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    env = {
        "FAL_KEY": FAKE_KEYS["fal"], "GROQ_API_KEY": FAKE_KEYS["groq"],
        "REPLICATE_API_TOKEN": FAKE_KEYS["replicate"], "OPENROUTER_API_KEY": FAKE_KEYS["openrouter"],
        "FAL_QUEUE_BASE": f"{base}/fal", "GROQ_BASE_URL": f"{base}/groq",
        "REPLICATE_API_BASE": f"{base}/replicate", "OPENROUTER_BASE_URL": f"{base}/openrouter",
        "FAL_UPLOAD": "datauri",
        "CLOUD_CACHE_DIR": str(REPO_ROOT / "data" / "cache_fake"),
    }
    return srv, base, state, env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--faults", default=os.environ.get("FAKE_CLOUD_FAULTS", ""))
    args = ap.parse_args()
    srv, base, state, env = start(args.port, {f for f in args.faults.split(",") if f})
    print(f"fake cloud on {base}; set these to use it:")
    for k, v in env.items():
        print(f"  {k}={v}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
