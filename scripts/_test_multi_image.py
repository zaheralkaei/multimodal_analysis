"""Test sending multiple frames per shot. Does Gemini see motion?"""
import urllib.request, json, base64, csv, time
from PIL import Image
from pathlib import Path

# Load API key
env = open(r"C:\github_projects\multimodal_analysis\.env").read()
key = env.split("OLLAMA_API_KEY=")[1].split("\n")[0].strip()

# Pick one shot
shots = json.loads(Path(r"C:\github_projects\multimodal_analysis\data\processed\shots.json").read_text(encoding="utf-8"))
s = shots[0]  # shot 0
print(f"Testing shot 0: {s['start_sec']:.1f}-{s['end_sec']:.1f}s ({s['duration_sec']:.1f}s)")
print(f"  Optical flow says: {s.get('camera_motion', '?')}")

# Find 5 frames within this shot (including start, mid, end)
fps_extract = 2  # we extracted at 2 fps
shot_start = int(s['start_sec'] * fps_extract)
shot_end = int(s['end_sec'] * fps_extract)
frame_indices = [shot_start + i * (shot_end - shot_start) // 4 for i in range(5)]
print(f"  Frame indices: {frame_indices}")

# Load and base64-encode 5 frames
frames_dir = Path(r"C:\github_projects\multimodal_analysis\data\processed\frames")
images_b64 = []
for fi in frame_indices:
    fp = frames_dir / f"frame_{fi:05d}.jpg"
    if fp.exists():
        with fp.open("rb") as f:
            images_b64.append(base64.b64encode(f.read()).decode())
    else:
        print(f"  Missing: {fp}")

print(f"  Loaded {len(images_b64)} frames")

# Ask: what camera motion do you see?
payload = {
    "model": "gemini-3-flash-preview",
    "prompt": f"""I'm showing you {len(images_b64)} frames sampled evenly across one shot of a music video ({s['start_sec']:.1f}-{s['end_sec']:.1f}s).

What is the camera doing across these frames? Answer in JSON:
{{"camera_motion": "<one of: static, pan-left, pan-right, tilt-up, tilt-down, zoom-in, zoom-out, handheld, dolly, tracking, unknown>", "evidence": "<1 sentence explaining what you saw>"}}""",
    "images": images_b64,
    "stream": False,
    "options": {"temperature": 0, "seed": 42, "num_predict": 300},
}
url = "https://ollama.com/api/generate"
t0 = time.time()
req = urllib.request.Request(
    url, data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
)
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        result = json.loads(r.read())
    elapsed = time.time() - t0
    print(f"\n  Response in {elapsed:.1f}s:")
    print(f"  {result.get('response', '')!r}")
except Exception as e:
    print(f"  err: {e}")