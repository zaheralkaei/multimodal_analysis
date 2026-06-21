"""Test multi-image VLM camera detection on a shot with known motion."""
import urllib.request, json, base64, csv, time
from pathlib import Path

# Load API key from .env
env_lines = open(r"C:\github_projects\multimodal_analysis\.env").read().splitlines()
key = None
for line in env_lines:
    if line.startswith("OLLAMA_API_KEY="):
        key = line.split("=", 1)[1].strip()
        break

# Read shot camera to find a motion shot
cams = list(csv.DictReader(open(r"C:\github_projects\multimodal_analysis\data\processed\shot_camera.csv", encoding="utf-8")))
shots = json.loads(Path(r"C:\github_projects\multimodal_analysis\data\processed\shots.json").read_text(encoding="utf-8"))

# Pick a clear motion shot — find one with strong vertical motion
best = None
for c in cams:
    if abs(float(c["tilt_score_mean"])) > 0.8 and float(c["duration_sec"]) > 2.0:
        idx = int(c["shot_idx"])
        s = shots[idx]
        if best is None or abs(float(c["tilt_score_mean"])) > abs(float(best[2]["tilt_score_mean"])):
            best = (idx, s, c)

if best is None:
    # Fall back to any non-static
    for c in cams:
        if c["camera_motion"] != "static" and float(c["duration_sec"]) > 2.0:
            idx = int(c["shot_idx"])
            s = shots[idx]
            best = (idx, s, c)
            break

idx, s, c = best
print(f"Testing shot {idx}: {s['start_sec']:.1f}-{s['end_sec']:.1f}s")
print(f"  Optical flow says: {c['camera_motion']}")
print(f"  Pan score: {c['pan_score_mean']}, Tilt score: {c['tilt_score_mean']}, Zoom score: {c['zoom_score_mean']}")

fps_extract = 2
shot_start = max(0, int(s['start_sec'] * fps_extract))
shot_end = int(s['end_sec'] * fps_extract)
n_frames = min(5, shot_end - shot_start + 1)
step = max(1, (shot_end - shot_start) // n_frames)
frame_indices = [shot_start + i * step for i in range(n_frames)]
frame_indices = [min(fi, shot_end) for fi in frame_indices]
print(f"  Frame indices: {frame_indices}")

frames_dir = Path(r"C:\github_projects\multimodal_analysis\data\processed\frames")
images_b64 = []
for fi in frame_indices:
    fp = frames_dir / f"frame_{fi:05d}.jpg"
    if fp.exists():
        with fp.open("rb") as f:
            images_b64.append(base64.b64encode(f.read()).decode())
        print(f"    loaded frame_{fi:05d}.jpg ({fp.stat().st_size} bytes)")
    else:
        print(f"    MISSING frame_{fi:05d}.jpg")

print(f"  Total: {len(images_b64)} images, payload: {sum(len(b) for b in images_b64)} bytes")

payload = {
    "model": "gemini-3-flash-preview",
    "prompt": f"""I will show you {len(images_b64)} frames sampled across a shot of a music video ({s['start_sec']:.1f}-{s['end_sec']:.1f}s).

What is the camera doing across these frames? The same subject may be in different positions because the camera moved.

Respond with ONLY this JSON: {{"camera_motion": "<one word from: static, pan-left, pan-right, tilt-up, tilt-down, zoom-in, zoom-out, handheld, dolly, tracking, unknown>", "evidence": "<1 sentence>"}}""",
    "images": images_b64,
    "stream": False,
    "options": {"temperature": 0, "seed": 42, "num_predict": 500},
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