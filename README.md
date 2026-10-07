# multimodal_analysis

End-to-end multimodal analysis of music videos. The pipeline extracts
**7 synchronized streams** from a video (frames, audio, visual captions,
camera motion, lyrics, audio mood, music structure), joins them per shot,
tests cross-modal questions statistically ("are cuts on the beat?", "are shots
shorter when the music is louder?") and produces inspectable CSVs/JSONs plus a
self-contained interactive HTML dashboard.

Designed to be:
- **Reproducible**: every number in the dashboard is computed from raw
  outputs, and every phase records its parameters, code version and input
  fingerprints (`run_info.json`).
- **Auditable**: every phase writes inspectable artifacts before the next
  one runs. You can open the CSVs to see what each model said.
- **Tested**: a test suite runs the pipeline on synthetic videos whose
  ground truth is known exactly (shot boundaries, camera motion, tempo, beat
  alignment), with no API key or model download needed.
- **Validatable**: a labelling page and agreement report measure how well the
  model labels match human judgement.

> **Note on the committed dashboards** (`reports/rtwpk9rb1Dc`, `reports/Z2ki180nHCI`):
> they were generated before the fixes in [docs/AUDIT_2026-10.md](docs/AUDIT_2026-10.md)
> (frames taken from the wrong part of each video, inverted camera directions).
> Re-run the pipeline on both videos before using their numbers.

---

## Quickstart

```bash
# 1. Install
git clone https://github.com/zaheralkaei/multimodal_analysis.git
cd multimodal_analysis
python -m pip install -r requirements.txt
choco install ffmpeg   # Windows; or brew/apt (see below)

# 2. Configure (cloud vision model)
cp .env.example .env
# Edit .env: paste your OLLAMA_API_KEY from https://ollama.com/settings/keys

# 3. Run on any YouTube video
python scripts/run_pipeline.py "https://www.youtube.com/watch?v=rtwpk9rb1Dc"

# 4. Open the dashboard
# reports/<video_id>/dashboard.html
```

The first run takes ~15-25 minutes (mostly the vision-model calls and the CLAP
model download). Running the same command again only re-runs phases whose
inputs, arguments or code changed.

---

## Install

### Requirements

| Dependency | Version | Why |
|---|---|---|
| Python | 3.10+ | CI runs 3.12 and 3.13 |
| ffmpeg | 4.4+ | frame extraction, audio decode |
| ~5 GB free disk | | CLAP model (~2 GB) + Whisper + per-video data |

### Python packages

```bash
python -m pip install -r requirements.txt        # runtime
python -m pip install demucs                     # optional: --separate-vocals
python -m pip install -r requirements-ci.txt     # development: pinned test/lint set
```

`requirements.txt` gives version ranges capped at the next major release;
`requirements-ci.txt` pins the exact versions the test suite is verified with.

### ffmpeg (separate install)

```bash
choco install ffmpeg      # Windows (or https://www.gyan.dev/ffmpeg/builds/)
brew install ffmpeg       # macOS
sudo apt install ffmpeg   # Debian/Ubuntu
```

Verify with `ffmpeg -version`.

### Configuration (.env)

The default vision model is **Gemini 3 Flash** on **Ollama Cloud**.

1. Get an API key at https://ollama.com/settings/keys
2. `cp .env.example .env` and edit:
   ```
   OLLAMA_API_KEY=ollama_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
   OLLAMA_BASE_URL=https://ollama.com/api
   VISION_MODEL=gemini-3-flash-preview
   ```
3. `.env` is gitignored, so never commit your key

For local vision (no API key, slower on CPU):
```
OLLAMA_BASE_URL=http://localhost:11434
VISION_MODEL=gemma3:4b
```

---

## Run

### Basic usage

```bash
# YouTube video: ID is auto-derived from the URL
python scripts/run_pipeline.py "https://www.youtube.com/watch?v=rtwpk9rb1Dc"

# Local file: ID is auto-derived from the filename
python scripts/run_pipeline.py ~/videos/my_clip.mp4

# Explicit ID (letters, digits, - and _ only)
python scripts/run_pipeline.py "https://www.youtube.com/watch?v=Z2ki180nHCI" --id 4blocks
```

The script derives a `video_id`, downloads the video (if a URL) to
`data/raw/<video_id>.mp4`, runs the phases into `data/<video_id>/`, and writes
`reports/<video_id>/dashboard.html`.

### Phases

| # | Phase | What | Output |
|---|---|---|---|
| 0 | Input prep | Download video, extract frames (2 fps) + 16 kHz audio | `frames/`, `audio.wav`, `metadata.json` |
| 1 | Shot detection | PySceneDetect AdaptiveDetector + fade detection | `shots.json` (with 3 key frames per shot) |
| 2 | Vision | Vision model on 3 frames per shot, fixed vocabularies via JSON schema | `shot_vision.csv`, `shot_vision_meta.json` |
| 3 | Camera motion | Feature tracking + RANSAC similarity fit | `shot_camera.csv` |
| 4 | Transcription | faster-whisper (multilingual), word timestamps, optional Demucs | `transcript.csv/json`, `transcript_words.csv` |
| 5 | Audio tagging | CLAP, per-group probabilities (mood / section / instrument) | `audio_clap.csv` |
| 6 | Music structure | librosa tempo, beats, key, per-second energy and onsets | `music_features.csv`, `music_summary.json` |
| 7 | Sync + statistics | Join per shot; cut-on-beat test; correlations | `sync_per_shot.csv`, `sync_stats.json` |
| 8 | Dashboard | Self-contained Plotly HTML | `reports/<video_id>/dashboard.html` |

### Useful options

```bash
# Skip phases: 2 = vision (no API calls), 4 = Whisper, 5 = CLAP (no 2 GB download)
python scripts/run_pipeline.py URL --skip 2,5

# Better lyrics on dense mixes: isolate the vocals first (pip install demucs)
python scripts/run_pipeline.py URL --separate-vocals

# Force a language for Whisper
python scripts/run_pipeline.py URL --whisper-language de

# Re-run everything from phase 4, even if up to date
python scripts/run_pipeline.py URL --start-from 4 --force

# Classic fixed-threshold shot detector
python scripts/run_pipeline.py URL --detector content
```

Phases are skipped automatically when their arguments, inputs and code are
unchanged. When one phase re-runs, its outputs change, so everything
downstream re-runs too.

### Run individual phases manually

```bash
export PROCESSED_DIR="data/rtwpk9rb1Dc"
export REPORTS_DIR="reports/rtwpk9rb1Dc"

python scripts/phase3_camera.py --pan-thresh 0.03   # every threshold is a flag
python scripts/phase7_sync.py
python scripts/phase8_dashboard.py --offline        # embed plotly.js (works without internet)
```

---

## Checking the labels against humans

The vision model's emotion labels and both camera-motion signals are model
judgements. To measure how far to trust them:

```bash
export PROCESSED_DIR=data/<video_id> REPORTS_DIR=reports/<video_id>
python scripts/label_shots.py --n 50          # writes reports/<video_id>/labeling.html
# open it, label the shots, click "Download CSV", save as data/<video_id>/human_labels.csv
python scripts/validate_labels.py             # accuracy (95% CI) + Cohen's kappa
python scripts/phase8_dashboard.py            # the dashboard now shows the agreement table
```

The labelling page does not show the model's answers, so they can't bias the
labeller.

## Comparing videos

```bash
python scripts/compare_videos.py              # every processed video under data/
python scripts/compare_videos.py rtwpk9rb1Dc Z2ki180nHCI
```

Writes `reports/comparison.html` and `reports/comparison.csv` (cuts per minute,
shot length distributions, on-beat rate vs chance, camera/emotion mix). It
warns when videos were processed with different settings or code versions.

## How the statistics work

- **Cut on beat**: a cut is on beat if it lands within ±100 ms (also reported
  at ±50 and ±200 ms) of a detected beat. The chance level is the share of the
  timeline within that distance of a beat. At 120 BPM and ±100 ms that is
  already ~40%, so a raw percentage means little on its own. The p-value comes
  from a permutation test that moves each cut by a random fraction of its local
  beat period.
- **Correlations**: Spearman's rho across shots with a moving-block bootstrap
  95% CI (neighbouring shots share a song section, so they are resampled
  together). No significance is claimed with fewer than 20 shots.

Both live in `scripts/crossmodal.py`.

---

## Development

```bash
pip install -r requirements-ci.txt
ruff check scripts tests
pytest                      # ~20 s; needs ffmpeg, no network
```

The tests build synthetic videos with known content: shots in distinct colours
cut exactly on a 120 BPM click track, each with a known camera move. They check
that every stage recovers that ground truth: shot boundaries, thumbnails from
the right shot, camera labels and speeds, tempo, beat alignment, key, the
dashboard and the skip logic. Phase 2 runs against a fake Ollama server. CI
(`.github/workflows/ci.yml`) runs lint and tests on every push.

---

## Directory structure

```
multimodal_analysis/
├── data/                              ← all generated, gitignored
│   ├── raw/<video_id>.mp4             ← downloaded video
│   └── <video_id>/
│       ├── frames/frame_*.jpg         ← JPEGs at --fps
│       ├── audio.wav, metadata.json
│       ├── shots.json                 ← boundaries, mid-frame, 3 key frames
│       ├── shot_vision.csv            ← vision model per shot
│       ├── shot_camera.csv            ← camera motion per shot
│       ├── transcript.*, transcript_words.csv
│       ├── audio_clap.csv             ← CLAP probabilities per 5 s window
│       ├── music_features.csv, music_summary.json
│       ├── sync_per_shot.csv, sync_stats.json
│       ├── human_labels.csv, validation.json   ← optional, from the labelling workflow
│       └── run_info.json              ← provenance for every phase
├── reports/
│   ├── <video_id>/dashboard.html      ← self-contained, open in any browser
│   └── comparison.html                ← from compare_videos.py
├── scripts/
│   ├── run_pipeline.py                ← runs phases 0-8
│   ├── phase0_input.py … phase8_dashboard.py
│   ├── common.py                      ← paths, vocabularies, frame timing, provenance
│   ├── crossmodal.py                  ← statistics (beat test, correlations)
│   ├── label_shots.py, validate_labels.py, compare_videos.py
│   ├── _env.py, _normalize_emotion.py, _renormalize_existing.py
├── tests/                             ← pytest suite on synthetic media
├── experiments/                       ← one-off scripts behind docs/CAMERA_DETECTION.md
├── docs/
├── requirements.txt, requirements-ci.txt, pyproject.toml
└── README.md
```

---

## Models and what they do

| Phase | Model / method | What it analyzes | Limits |
|---|---|---|---|
| 1 (shots) | PySceneDetect AdaptiveDetector + ThresholdDetector | Hard cuts, fades to black | Cross-dissolves not detected |
| 2 (vision) | **Gemini 3 Flash** via Ollama Cloud (configurable) | Caption, camera, emotion, colours, entities, location, lighting, composition | 3 frames per shot; labels are judgements, so validate them |
| 3 (camera) | OpenCV feature tracking + RANSAC | Pan/tilt (with direction), zoom in/out, static, handheld, with speeds | Large moving subjects filling the frame can look like camera motion |
| 4 (transcription) | **faster-whisper small** (+ optional Demucs) | Sung/spoken words with word timestamps | Hallucinates on silence/music; Demucs helps |
| 5 (audio) | **CLAP** (laion/clap-htsat-fused) | 12 mood, 7 section, 8 instrument tags | Fixed vocabulary; probabilities are relative within each group |
| 6 (music) | librosa | Tempo, beats, key, RMS, onsets, brightness | Beat tracking fails on rubato; relative major/minor confusable |
| 7 (sync) | `crossmodal.py` | Joins + permutation test + bootstrap CIs | Needs ≥20 shots for correlations |
| 8 (dashboard) | Plotly | Self-contained HTML with filters | Plotly loads from CDN unless `--offline` |

### Vision model choices

**Gemini 3 Flash** (default): 1-2 s per call, free tier at ollama.com. For
**local-only** use: `gemma3:4b` (~3 GB RAM), `gemma3:27b` (~16 GB),
`gemma4:31b` (24 GB+). Requests run 4 at a time (`--workers`).

**Whisper model sizes** (faster-whisper): `tiny`, `base`, `small` (**default**),
`medium`, `large-v3`. The `.en` variants are English-only.

### What CLAP can and can't do

CLAP scores how well each 5-second audio window matches each text label. It is
good at ranking a known vocabulary but cannot describe sounds outside it. Each
tag group (mood, section, instrument) gets its own softmax, so a strong
instrument match cannot suppress the mood scores. Phase 5 prints how much each
tag varies over the song; flat tags carry no timing information.

---

## Troubleshooting

### "ffmpeg not found"
Install ffmpeg (see Install). Verify with `ffmpeg -version`.

### "Health check failed: model didn't respond"
- **Cloud**: check `OLLAMA_API_KEY` in `.env`, verify at https://ollama.com/settings/keys
- **Local**: is ollama running? `ollama serve`, then `ollama pull gemma3:4b`
- If your endpoint rejects JSON schemas, phase 2 falls back to plain JSON mode
  automatically; `--no-schema` skips the attempt.

### "shot_vision.csv has [error] or [parse_error] rows"
Re-run phase 2: it keeps successful rows and retries only the failed ones.

### "Lyrics are missing or wrong"
Try `--separate-vocals` (needs `pip install demucs`) and/or a larger model
(`--whisper-model medium`). Force the language with `--whisper-language`.

### "Camera motion is 'static' for a slow pan"
Motion below 4% of the frame per second counts as static by design. Lower it
with `python scripts/phase3_camera.py --pan-thresh 0.02` (also `--tilt-thresh`,
`--zoom-thresh`, `--jitter-thresh`).

### "A phase didn't re-run after I changed something"
Phases re-run when their arguments, input files or code change. Anything else
(e.g. a changed `.env`) needs `--force`.

---

## Design notes

- [docs/AUDIT_2026-10.md](docs/AUDIT_2026-10.md): latest audit (critical
  frame-alignment fix, implemented improvements)
- [docs/CAMERA_DETECTION.md](docs/CAMERA_DETECTION.md): why camera motion comes
  from the frames rather than the vision model (with failed experiments in `experiments/`)
- [docs/METHODOLOGY_REVIEW.md](docs/METHODOLOGY_REVIEW.md): alternatives for
  shot detection, camera motion and transcription
- [docs/COMPARISON_1FPS_VS_2FPS.md](docs/COMPARISON_1FPS_VS_2FPS.md): frame-rate
  sensitivity (computed before the frame-alignment fix; needs re-running)
- [docs/AUDIT.md](docs/AUDIT.md), [docs/AUDIT_R2.md](docs/AUDIT_R2.md),
  [docs/PLAN_ROUND_2.md](docs/PLAN_ROUND_2.md), [docs/STRUCTURE_V3.md](docs/STRUCTURE_V3.md): earlier rounds

## License

Pipeline code: MIT. Data sources: see [docs/STRUCTURE_V3.md](docs/STRUCTURE_V3.md)
for each model's license.
