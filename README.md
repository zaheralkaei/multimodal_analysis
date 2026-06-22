# multimodal_analysis

End-to-end multimodal analysis of music videos. The pipeline extracts
**7 synchronized streams** from a video (frames, audio, visual captions,
camera motion, lyrics, audio mood, music structure) and produces an
inspectable set of CSVs/JSONs plus an interactive HTML dashboard.

Designed to be:
- **Reproducible** — every number in the dashboard is computed dynamically
  from raw outputs; nothing is hardcoded.
- **Auditable** — every phase writes inspectable artifacts before the next
  one runs. You can open the CSVs to see what each model said.
- **Composable** — swap any phase (e.g., TransNetV2 → PySceneDetect) without
  breaking the rest of the pipeline.

---

## Quickstart

```bash
# 1. Install
git clone https://github.com/zaheralkaei/multimodal_analysis.git
cd multimodal_analysis
python -m pip install -r requirements.txt
choco install ffmpeg   # Windows; or use your package manager

# 2. Configure (cloud vision model)
cp .env.example .env
# Edit .env: paste your OLLAMA_API_KEY from https://ollama.com/settings/keys

# 3. Run on any YouTube video
python scripts/run_pipeline.py "https://www.youtube.com/watch?v=rtwpk9rb1Dc"

# 4. Open the dashboard
# reports/<video_id>/dashboard.html
```

The first run takes ~15-25 minutes (mostly the Gemini API calls + CLAP model
download). Subsequent runs on the same video resume in seconds.

---

## Install

### Requirements

| Dependency | Version | Why |
|---|---|---|
| Python | 3.11+ | tested on 3.11, 3.13 |
| ffmpeg | 4.4+ | video frame extraction, audio decode |
| yt-dlp | 2024+ | YouTube download |
| ~5 GB free disk | | CLAP model (~2 GB) + per-video data |

### Python packages

```bash
python -m pip install -r requirements.txt
```

This installs:
- `yt-dlp` — YouTube downloading
- `scenedetect[opencv]` — shot detection (BSD-3-Clause)
- `opencv-python-headless` — optical flow for camera motion
- `faster-whisper` — speech-to-text (multilingual)
- `transformers` + `torch` — CLAP for audio tagging
- `librosa`, `soundfile` — music analysis
- `plotly`, `Pillow` — dashboard rendering
- `python-dotenv` — .env loading (we use a zero-dep alternative instead)

### ffmpeg (separate install)

```bash
# Windows
choco install ffmpeg
# or download from https://www.gyan.dev/ffmpeg/builds/

# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg   # Debian/Ubuntu
sudo dnf install ffmpeg   # Fedora
```

Verify with `ffmpeg -version`.

### Configuration (.env)

The vision model is **Gemini 3 Flash** running on **Ollama Cloud** (not local).
This is faster and free for our usage volume.

1. Get an API key at https://ollama.com/settings/keys
2. Copy the template: `cp .env.example .env`
3. Edit `.env`:
   ```
   OLLAMA_API_KEY=ollama_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
   OLLAMA_BASE_URL=https://ollama.com/api
   VISION_MODEL=gemini-3-flash-preview
   ```
4. `.env` is gitignored — never commit your key

For local vision (no API key, slower on CPU):
```
OLLAMA_BASE_URL=http://localhost:11434
VISION_MODEL=gemma3:4b   # smaller, English-friendly
```

---

## Run

### Basic usage

```bash
# YouTube video — ID is auto-derived from the URL
python scripts/run_pipeline.py "https://www.youtube.com/watch?v=rtwpk9rb1Dc"

# Local file — ID is auto-derived from the filename
python scripts/run_pipeline.py ~/videos/my_clip.mp4

# Explicit ID (overrides auto-derive)
python scripts/run_pipeline.py "https://www.youtube.com/watch?v=Z2ki180nHCI" --id 4blocks
```

The script:
1. Derives a `video_id` from the URL or filename
2. Downloads the video (if URL) to `data/raw/<video_id>.mp4`
3. Runs all 8 phases, writing per-video output to `data/<video_id>/`
4. Builds the dashboard at `reports/<video_id>/dashboard.html`

### Phases

| # | Phase | What | Output |
|---|---|---|---|
| 0 | Input prep | Download video, extract frames + audio | `frames/`, `audio.wav`, `metadata.json` |
| 1 | Shot detection | PySceneDetect ContentDetector | `shots.json`, `shot_predictions.csv` |
| 2 | Vision | Gemini 3 Flash via Ollama Cloud (JSON-mode prompt) | `shot_vision.csv` |
| 3 | Camera motion | OpenCV Farneback optical flow | `shot_camera.csv` |
| 4 | Transcription | faster-whisper (multilingual) | `transcript.json`, `transcript.csv` |
| 5 | Audio tagging | CLAP (laion/clap-htsat-fused) | `audio_clap.csv` |
| 6 | Music structure | librosa tempo + key detection | `music_features.csv`, `music_summary.json` |
| 7 | Sync | Join everything per shot | `sync_per_shot.csv`, `sync_stats.json` |
| 8 | Dashboard | Plotly HTML | `reports/<video_id>/dashboard.html` |

### Skip slow phases

```bash
# No vision captions (no API calls — fast, but no Emotion/Camera/Caption columns)
python scripts/run_pipeline.py URL --skip-phase2

# No audio tagging (skip 2GB CLAP download)
python scripts/run_pipeline.py URL --skip-phase5
```

### Resume from a specific phase

```bash
# Re-run only phase 7 and 8 (after tweaking phase 6)
python scripts/run_pipeline.py URL --start-from 7
```

### Run individual phases manually

```bash
export PROCESSED_DIR="data/rtwpk9rb1Dc"
export REPORTS_DIR="reports/rtwpk9rb1Dc"

python scripts/phase1_shots.py --threshold 35 --min-scene-len 30
python scripts/phase2_vision.py --model gemini-3-flash-preview
python scripts/phase8_dashboard.py
```

### Multi-language support

The default Whisper model is `small` (multilingual, auto-detects language).
For specific languages:
```bash
python scripts/run_pipeline.py URL --whisper-language de   # German
python scripts/run_pipeline.py URL --whisper-language en   # English
python scripts/run_pipeline.py URL --whisper-language ar   # Arabic
```

---

## Directory structure

```
multimodal_analysis/
├── data/
│   ├── raw/
│   │   └── <video_id>.mp4            ← downloaded video
│   └── <video_id>/
│       ├── audio.wav                 ← 16 kHz mono
│       ├── frames/frame_*.jpg        ← JPEGs at --fps
│       ├── metadata.json
│       ├── shots.json                ← shot boundaries + mid-frames
│       ├── shot_vision.csv           ← VLM per-shot data
│       ├── shot_camera.csv           ← OpenCV per-shot motion
│       ├── transcript.json           ← Whisper segments
│       ├── audio_clap.csv            ← CLAP scores per 5s window
│       ├── music_features.csv       ← per-second features (tempo, beats, RMS)
│       ├── music_summary.json        ← tempo, key, beat count
│       ├── sync_per_shot.csv         ← joined table (one row per shot)
│       └── sync_stats.json
├── reports/
│   └── <video_id>/
│       ├── dashboard.html            ← open in any browser
│       └── frames/                   ← thumbnails embedded in dashboard
├── scripts/
│   ├── phase0_input.py               ← download + extract
│   ├── phase1_shots.py               ← shot detection
│   ├── phase2_vision.py              ← VLM per-shot Q&A
│   ├── phase3_camera.py              ← optical flow
│   ├── phase4_transcribe.py          ← Whisper
│   ├── phase5_audio.py               ← CLAP
│   ├── phase6_music.py               ← librosa
│   ├── phase7_sync.py                ← join all streams
│   ├── phase8_dashboard.py           ← Plotly HTML
│   ├── run_pipeline.py               ← wrapper that runs all 8
│   ├── _env.py                       ← .env loader
│   ├── _normalize_emotion.py         ← emotion synonym map
│   └── _renormalize_existing.py
├── docs/
│   ├── STRUCTURE_V3.md               ← folder naming history
│   ├── CAMERA_DETECTION.md           ← why we use OpenCV, not the VLM
│   ├── COMPARISON_1FPS_VS_2FPS.md
│   └── PLAN_ROUND_2.md
├── .env                              ← gitignored
├── .env.example
├── requirements.txt
└── README.md
```

---

## Models and what they do

| Phase | Model | What it analyzes | Capabilities | Limits |
|---|---|---|---|---|
| 1 (shots) | PySceneDetect ContentDetector | Visual scene changes | BSD-3-Clause, ~5K⭐, 5 detector algorithms | Reads video directly, no frame-rate dep |
| 2 (vision) | **Gemini 3 Flash** via Ollama Cloud | Per-shot caption, camera, emotion, colors, entities, location, lighting, composition | 1-2s per call, JSON-mode, free tier ~10K calls/day | Single mid-frame per shot (can't see motion) |
| 3 (camera) | OpenCV Farneback optical flow | Camera motion (pan/tilt/zoom/static) | Local, free, no API | 8 discrete classes (no magnitude) |
| 4 (transcription) | **faster-whisper small** (multilingual) | Spoken/sung words | 99 languages, ~5MB model | Hallucinates on silence/music |
| 5 (audio) | **CLAP** (laion/clap-htsat-fused) | 27 audio tags (mood, section, instrument) | 48kHz, 27 predefined tags | Vocabulary locked at design time |
| 6 (music) | librosa | Tempo, beats, key, RMS energy | Pure signal processing | Beat detection fails on rubato music |
| 7 (sync) | rule-based | Cross-modal joins + stats | Deterministic | ±100ms beat tolerance is arbitrary |
| 8 (dashboard) | Plotly | HTML visualization | Interactive, single-file | Self-contained but not mobile-optimized |

### What each vision model is good at

**Gemini 3 Flash** is the default because:
- 1-2s per call (vs 30s+ for gemma3:4b locally)
- Free tier at ollama.com
- Good at structured JSON output
- ~10K calls/day limit on free tier

For **local-only** use:
- `gemma3:4b` — small, English-friendly, ~3GB RAM
- `gemma3:27b` — much better, needs ~16GB RAM
- `gemma4:31b` — best open, needs 24GB+ RAM

For **other clouds**:
- `gpt-4o-mini` — pay-per-call, fastest in benchmarks
- `claude-3.5-sonnet` — most accurate on nuanced prompts
- `qwen2.5-vl-72b` — best open multimodal (Ollama cloud, not local)

**Whisper model sizes** (faster-whisper):
- `tiny` — 39M params, ~1GB RAM, fast
- `base` — 74M params, ~1GB RAM
- `small` — 244M params, ~2GB RAM, **default**
- `medium` — 769M params, ~5GB RAM, slower
- `large-v3` — 1550M params, ~10GB RAM, best accuracy

Multilingual versions: drop the `.en` suffix (e.g. `small` instead of `small.en`).

### What CLAP can and can't do

CLAP is a contrastive audio-text model. It scores how well each audio window
matches each text label. **Good for**: matching a known tag vocabulary.
**Bad for**: open-ended sound description (no way to get new vocabulary).

Our 27 tags cover:
- 12 mood tags (happy, sad, aggressive, romantic, …)
- 7 section tags (intro, verse, chorus, bridge, outro, instrumental break, vocal only)
- 8 instrument tags (acoustic guitar, electric guitar, piano, drums, bass, synth, strings, vocal only no instruments)

In practice on a single 3-4 minute pop song, only 6-9 tags have meaningful
variance — the rest are zero. See `phase5_audio.py` for the active/inactive
tag report.

---

## Troubleshooting

### "ffmpeg not found"
Install ffmpeg (see Install section). Verify with `ffmpeg -version`.

### "No module named 'scenedetect'"
```bash
pip install scenedetect[opencv]
```

### "Health check failed: model didn't respond"
- **Cloud**: check `OLLAMA_API_KEY` in `.env`, verify at https://ollama.com/settings/keys
- **Local**: is ollama running? `ollama serve` in another terminal, then `ollama pull gemma3:4b`

### "shot_vision.csv has truncated captions"
Increase `num_predict` in `phase2_vision.py` (currently 1500). Or check the
parse_json_response output for "parse_error" rows.

### "Dashboard shows the wrong thumbnails"
This was a known bug: stale `reports/<video_id>/frames/` from a previous run.
Fix is in `phase8_dashboard.py`: it now checks file size before skipping the
copy, so different-size files (different videos) get re-copied.

If the issue persists, force a clean rebuild:
```bash
rm -rf reports/<video_id>/
python scripts/phase8_dashboard.py
```

### "Lyrics look like English instead of German/Arabic/etc."
Default Whisper is `small.en` (English-only) in older runs. Use the
multilingual version:
```bash
python scripts/run_pipeline.py URL --whisper-model small --whisper-language de
```

### "Camera motion is 'static' for everything"
The VLM sees only the mid-frame, which can't show motion. Use OpenCV
optical flow (already done in phase 3) for camera motion — see
`docs/CAMERA_DETECTION.md`.

---

## Design notes

- [docs/CAMERA_DETECTION.md](docs/CAMERA_DETECTION.md) — why we use
  OpenCV for camera, not the VLM (with failed experiments)
- [docs/COMPARISON_1FPS_VS_2FPS.md](docs/COMPARISON_1FPS_VS_2FPS.md) —
  frame-rate sensitivity analysis
- [docs/PLAN_ROUND_2.md](docs/PLAN_ROUND_2.md) — round-2 audit plan
- [docs/STRUCTURE_V3.md](docs/STRUCTURE_V3.md) — folder naming history

- [docs/AUDIT.md](docs/AUDIT.md) — round-3 audit report
  (code quality, data correctness, methodology, engineering hygiene)

- [docs/METHODOLOGY_REVIEW.md](docs/METHODOLOGY_REVIEW.md) — alternatives
  to current shot detection, camera motion, transcription (free + paid)

## License

Pipeline code: MIT. Data sources: see [docs/STRUCTURE_V3.md](docs/STRUCTURE_V3.md)
for each model's license.
