"""Test that qwen2.5vl responds correctly via ollama."""
import urllib.request, json, base64
from PIL import Image

img = Image.new("RGB", (640, 480), color=(70, 130, 180))
img.save("test_qwen.png")
print("Saved test image")

with open("test_qwen.png", "rb") as f:
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
with urllib.request.urlopen(req, timeout=180) as r:
    result = json.loads(r.read())

print(f"\nQwen response:\n  {result.get('response', '')}")
print(f"\n[stats] eval_count={result.get('eval_count')}, "
      f"duration={result.get('total_duration')/1e9:.1f}s")
