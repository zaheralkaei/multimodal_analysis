# multimodal_analysis

Multimodal analysis of a music video across 8 pipelines:

1. **Input prep** (yt-dlp + ffmpeg) — frames at 1fps + 16kHz mono WAV
2. **Shot detection** (TransNetV2) — finds scene cuts
3. **Per-shot visual analysis** (cloud vision model via Ollama API)
4. **Camera motion** (OpenCV optical flow) — pan/tilt/zoom/handheld
5. **Transcription** (faster-whisper) — lyrics/speech
6. **Audio tagging** (CLAP) — mood/section/instrument per 5s window
7. **Music structure** (librosa) — tempo, key, beats, RMS/centroid
8. **Sync + dashboard** (Plotly) — synchronized timeline, cross-modal findings

## Setup

```bash
git clone <repo>
cd multimodal_analysis
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# Edit .env: paste your OLLAMA_API_KEY
```

## Run

```bash
export PATH="/path/to/ffmpeg/bin:$PATH"  # need ffmpeg
.venv/bin/python scripts/phase0_input.py "https://www.youtube.com/watch?v=..." --fps 1
.venv/bin/python scripts/phase1_shots.py
.venv/bin/python scripts/phase2_vision.py
.venv/bin/python scripts/phase3_camera.py
.venv/bin/python scripts/phase4_transcribe.py --model small.en
.venv/bin/python scripts/phase5_audio.py
.venv/bin/python scripts/phase6_music.py
.venv/bin/python scripts/phase7_sync.py
.venv/bin/python scripts/phase8_dashboard.py
open reports/dashboard.html
```

## Configuration (.env)

- `OLLAMA_API_KEY` — get at https://ollama.com/settings/keys
- `OLLAMA_BASE_URL` — `https://ollama.com/api` (cloud) or `http://localhost:11434` (local)
- `VISION_MODEL` — default `gemini-3-flash-preview`. Other options: `gemma3:27b`, `gemma4:31b`

Each phase writes inspectable CSVs/JSONs to `data/processed/` before aggregation.

## Current results

Analysis of Tyla — "SHE DID IT AGAIN" feat. Zara Larsson (3:35, 215 frames).

- **2 shots** detected (212s + 2s trailing) — TransNetV2 sees almost no hard cuts
- **Camera**: 1 tilt-down, 1 static
- **Lyrics**: 5 segments, 281 chars ("Deep down you want me", "Dictive baby what you want me to do...")
- **Music**: 129.2 BPM, B minor, 444 beats
- **Audio mood** (CLAP): "powerful and confident" dominant
- **Visual** (Gemini): "A woman in a bikini top, joyful, medium close-up"
