"""Test which cloud models support image input."""
import urllib.request, json, base64
from PIL import Image

key = open(r"C:\github_projects\multimodal_analysis\.env").read().split("OLLAMA_API_KEY=")[1].split("\n")[0].strip()
print(f"key: {key[:8]}...{key[-4:]}")

# Make test image
img = Image.new("RGB", (224, 224), color=(70, 130, 180))
img.save(r"C:\github_projects\multimodal_analysis\test_img.jpg")
with open(r"C:\github_projects\multimodal_analysis\test_img.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

# Vision-capable candidates
candidates = ["gemma3:27b", "gemma4:31b", "gemini-3-flash-preview"]
for m in candidates:
    payload = {
        "model": m, "prompt": "Describe this image in one short sentence.",
        "images": [b64], "stream": False,
    }
    req = urllib.request.Request(
        "https://ollama.com/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            result = json.loads(r.read())
        resp = result.get('response', '')[:120]
        dur = result.get('total_duration', 0)/1e9
        print(f"\n✓ {m} ({dur:.1f}s)")
        print(f"  → {resp!r}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        print(f"\n✗ {m} → {body}")
    except Exception as e:
        print(f"\n✗ {m} → {type(e).__name__}: {e}")
