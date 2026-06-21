"""Test gemma3:4b for image understanding."""
import urllib.request, json, base64
from PIL import Image

img = Image.new("RGB", (640, 480), color=(70, 130, 180))
img.save("test_gemma.png")
print("saved")

with open("test_gemma.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

payload = {
    "model": "gemma3:4b",
    "prompt": "Describe this image in one sentence.",
    "images": [b64],
    "stream": False,
}
req = urllib.request.Request(
    "http://localhost:11434/api/generate",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"}
)
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        result = json.loads(r.read())
    print(f"Response: {result.get('response')!r}")
    print(f"Duration: {result.get('total_duration')/1e9:.1f}s")
except Exception as e:
    print(f"Failed: {e}")
