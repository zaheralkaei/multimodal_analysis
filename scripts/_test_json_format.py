"""Test if Ollama cloud respects 'format: json' for stricter JSON output."""
import urllib.request, json, base64
from PIL import Image

key = open(r"C:\github_projects\multimodal_analysis\.env").read().split("OLLAMA_API_KEY=")[1].split("\n")[0].strip()

# Make a test image
img = Image.new("RGB", (224, 224), color=(70, 130, 180))
img.save(r"C:\github_projects\multimodal_analysis\data\processed\_test_format.jpg")
with open(r"C:\github_projects\multimodal_analysis\data\processed\_test_format.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

# Test 1: format: "json"
payload = {
    "model": "gemini-3-flash-preview",
    "prompt": "What is the dominant color in this image?",
    "images": [b64],
    "stream": False,
    "format": "json",
    "options": {"temperature": 0, "num_predict": 200},
}
url = "https://ollama.com/api/generate"
req = urllib.request.Request(
    url, data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
)
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        result = json.loads(r.read())
    print("Test 1 (format='json'):")
    print(f"  Response: {result.get('response', 'NO RESPONSE')!r}")
    print()
except Exception as e:
    print(f"Test 1 err: {e}")
    print()

# Test 2: format: schema
payload["format"] = {
    "type": "object",
    "properties": {
        "color": {"type": "string"},
        "mood": {"type": "string"}
    },
    "required": ["color", "mood"]
}
payload["prompt"] = "What's the color and mood of this image?"
try:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        result = json.loads(r.read())
    print("Test 2 (format=schema):")
    print(f"  Response: {result.get('response', 'NO RESPONSE')!r}")
except Exception as e:
    print(f"Test 2 err: {e}")