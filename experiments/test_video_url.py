"""Try Gemini with a YouTube URL as video input (Google's recommended way)."""
import urllib.request, json, time
from pathlib import Path

import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from _env import load_env  # noqa: E402
from common import PROCESSED  # noqa: E402  (set PROCESSED_DIR=data/<video_id>)
load_env()
key = os.environ.get("OLLAMA_API_KEY")

if not key:
    print("[error] no OLLAMA_API_KEY in .env")
    exit(1)

# Test 1: try 'videos' field with YouTube URL (Ollama cloud may not support)
payload = {
    "model": "gemini-3-flash-preview",
    "prompt": "What is the camera doing at 30-second intervals? Respond with JSON.",
    "videos": ["https://www.youtube.com/watch?v=rtwpk9rb1Dc"],
    "stream": False,
    "options": {"temperature": 0, "num_predict": 500},
}
url = "https://ollama.com/api/generate"
t0 = time.time()
req = urllib.request.Request(
    url, data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
)
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        result = json.loads(r.read())
    elapsed = time.time() - t0
    print(f"Test 'videos' field ({elapsed:.1f}s):")
    print(f"  {result.get('response', 'NO RESPONSE')[:200]!r}")
except Exception as e:
    print(f"Test 'videos' field err: {e}")

# Try with larger num_predict and see the full response
payload["options"]["num_predict"] = 3000
t0 = time.time()
req = urllib.request.Request(
    url, data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
)
try:
    with urllib.request.urlopen(req, timeout=180) as r:
        result = json.loads(r.read())
    elapsed = time.time() - t0
    print(f"\nFull video test ({elapsed:.1f}s):")
    print(result.get('response', ''))
except Exception as e:
    print(f"err: {e}")