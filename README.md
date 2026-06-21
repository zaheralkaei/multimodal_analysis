# multimodal_analysis

End-to-end multimodal analysis of a single music video. The pipeline extracts
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

## Current results

**Video analyzed:** Tyla — *SHE DID IT AGAIN* feat. Zara Larsson
(3:35, 431 frames @ 2 fps, 17.4 MB, YouTube ID `rtwpk9rb1Dc`)

| Stream | Result |
|---|---|
| Shot detection (PySceneDetect ContentDetector, threshold=35) | **72 shots** (avg 3.0s, min 1.3s, max 9.5s) |
| Vision captions (Gemini 3 Flash via cloud) | 72/72 successful (576 API calls) |
| Camera motion (OpenCV optical flow @ 2 fps) | tilt-down 20, static 14, pan-right 10, pan-left 9, tilt-up 7, zoom-out 7, zoom-in 5 |
| Lyrics (faster-whisper small.en) | 5 segments, 281 chars |
| Audio mood (CLAP, 27 tags) | "powerful and confident" dominant |
| Music structure (librosa) | 129.2 BPM, B minor, 444 beats |
| Cuts on beat (±100ms) | **47.2%** (34/72) |

Open `reports/dashboard.html` to explore.

### 1 fps vs 2 fps — what changed?

The pipeline was first run with **1 fps**, then re-run with **2 fps**.
Here's what the change in frame rate actually changed in the results:

| Phase | At 1 fps | At 2 fps | Why it changed |
|---|---|---|---|
| 0 (frames) | 215 frames | 431 frames | 2× the data, 2× the disk |
| 1 (shots) | 72 shots | 72 shots | Reads video directly, not frames |
| 2 (vision) | 72 captions | 72 captions (re-run) | Same 1 mid-frame per shot |
| 3 (camera) | zoom-in 13, tilt-down 12, tilt-up 11, pan-left 11, pan-right 11, static 7, zoom-out 6, handheld 1 | **tilt-down 20, static 14, pan-right 10, pan-left 9, tilt-up 7, zoom-out 7, zoom-in 5** | Optical flow sees 2× the frame pairs |
| 4 (whisper) | 5 segments | 5 segments | Audio rate unchanged |
| 5 (CLAP) | "powerful and confident" | "powerful and confident" | Audio rate unchanged |
| 6 (librosa) | 129.2 BPM, B minor | 129.2 BPM, B minor | Audio rate unchanged |
| 7 (cuts on beat) | 47.2% (34/72) | 47.2% (34/72) | Beat timestamps unchanged |

**Phase 3 changed the most.** At 1 fps, the dominant motion was `zoom-in`
(13 shots) — frames 1 second apart often have a different center-of-mass
which the algorithm reads as radial divergence (zoom). At 2 fps, frames
0.5s apart are more often true pans/tilts, so we see `tilt-down` (20)
and `static` (14) dominate. The 1 fps distribution was likely noisy:
**the 2 fps result is more trustworthy**.

The 1 fps baseline is preserved in `data/processed_1fps_baseline/` for
diffing.

**Lessons learned:**
- Audio-only phases (4, 5, 6) are **completely unaffected** by frame
  rate. They depend on the audio track, not the frames.
- Shot detection (1) and vision (2) are also unaffected — they pick
  their own analysis frame regardless of what we extracted.
- **Camera classification (3) is the only phase whose output changes
  meaningfully with frame rate.** If we go to 5 fps, we'd expect the
  distribution to shift further toward tilt and pan (better time
  resolution) but disk cost would double again.
- The 47.2% beat-cut rate being identical is good — it means our
  beat detection is robust and the same set of 34 shots out of 72
  happen to be near a beat, regardless of how we classify their
  camera motion.

## Architecture

The pipeline is 8 phases. Each phase reads inputs from `data/processed/`
(written by previous phases) and writes new outputs there. Nothing is
destructive — you can re-run any phase without re-running earlier ones.

```
YouTube URL ─▶ [Phase 0] ─▶ frames + audio
                     │
                     ▼
              [Phase 1: shot detection]  ─▶ shots.json
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
[Phase 2: vision] [Phase 3: cam] [Phase 4: whisper]
        │            │            │
        └────────────┼────────────┘
                     ▼
              [Phase 5: CLAP]  ─▶ audio mood per 5s window
                     │
                     ▼
              [Phase 6: librosa]  ─▶ tempo + key + beats
                     │
                     ▼
              [Phase 7: sync]  ─▶ joined per-shot table
                     │
                     ▼
              [Phase 8: dashboard]  ─▶ reports/dashboard.html
```

## What each phase does

### Phase 0 — Input preparation

**What:** Downloads the video, extracts frames at a configurable rate
(default 2 fps) and audio as 16 kHz mono WAV.

**Tools:**
- **yt-dlp** — YouTube downloader (handles 95% of public videos)
- **ffmpeg** — frame + audio extraction (must be on PATH)

**Output:**
- `data/raw/video.mp4` (the source video)
- `data/processed/frames/frame_NNNNN.jpg` (default 2 fps, set via `--fps`)
- `data/processed/audio.wav` (16 kHz mono PCM)
- `data/processed/metadata.json` (duration, fps, resolution)

**Sampling-rate decision: 2 fps (configurable via `--fps`)**

This is the most important knob in the pipeline. It determines the
time-resolution of *every* downstream phase that uses frames. The
trade-offs:

| Rate | Frames per 3-min video | Disk | Optical-flow time resolution | Sub-second motion |
|---|---|---|---|---|
| 0.5 fps | 90 | 0.5 MB | 2 sec between frames | missed |
| **2 fps (default)** | **430** | **~10 MB** | **0.5 sec between frames** | **detected for >2s shots** |
| 5 fps | 900 | ~20 MB | 0.2 sec between frames | detected for >0.4s shots |
| 24 fps (native) | 4320 | ~200 MB | 0.04 sec between frames | full resolution |

**Why 2 fps and not 1 fps?** The first version of this project used
1 fps. That worked fine for shot detection and vision analysis (which
both pick 1 frame per shot anyway), but **phase 3 (optical-flow camera
classification) suffered** — frames 1 second apart cannot distinguish a
fast pan from a series of static frames. At 2 fps, consecutive frames
are 0.5 sec apart, so we can detect pan/tilt/zoom motion that completes
in 1-2 seconds (typical for music videos).

**Why not higher?** Disk and CPU cost grow linearly. 24 fps is the
native video rate but would generate ~200 MB of JPEG frames per video.
10 fps would catch every camera motion but at 4× the storage of 2 fps
for marginal improvement. **2 fps is the sweet spot for music videos:**
shots are typically 1-5 seconds, so 2-10 frames per shot is plenty
for optical flow to compute meaningful motion statistics.

**Why 16 kHz mono?** Whisper expects 16 kHz. CLAP is 48 kHz (it
re-samples internally if needed). Mono halves disk usage with no
quality loss for music analysis.

**How each phase uses the frames:**
- **Phase 1** (shot detection) — reads the video directly, ignores the
  extracted frames. Frame rate is irrelevant.
- **Phase 2** (vision) — picks 1 mid-frame per shot. Only needs the
  frames to exist; their rate doesn't matter.
- **Phase 3** (optical flow) — uses ALL frames within each shot. **This
  is where frame rate matters most.** At 1 fps, a 1-second shot has 2
  frames → optical flow sees 1 comparison. At 2 fps, 3 frames → 2
  comparisons. At 5 fps, 6 frames → 5 comparisons. More comparisons =
  better motion estimates.
- **Phases 4-8** (audio + sync + dashboard) — don't use frames.

For a detailed per-phase numerical comparison of 1 fps vs 2 fps, see
[docs/COMPARISON_1FPS_VS_2FPS.md](docs/COMPARISON_1FPS_VS_2FPS.md).

### Phase 1 — Shot boundary detection

**What:** Finds the timecodes where the camera "cuts" or transitions to a new
shot. Each shot gets a `(start_sec, end_sec, mid_sec, mid_frame_path)`.

**Tool:** **PySceneDetect** (`scenedetect` Python package, BSD-3-Clause,
4941⭐ on GitHub) with the **ContentDetector** algorithm.

**How ContentDetector works** (from the PySceneDetect docs):
1. For each frame, split into a 4×4 grid of pixel blocks
2. For each block, compute Δh (hue), Δs (saturation), Δl (luminance),
   and Δedge (edge intensity) vs. the previous frame
3. Sum the weighted deltas across all blocks
4. If the sum exceeds `threshold`, mark a scene change
5. Filter out changes within `min_scene_len` frames of each other
   (avoids flicker detection on compression artifacts)

**Tunables:**
- `--threshold 35` (default) — lower = more sensitive. 27 catches everything
  including false positives; 50 misses subtle fades.
- `--min-scene-len 30` (default) — min frames between detected cuts.
  At 24 fps, 30 frames = 1.25 seconds.

**Why not TransNetV2?** TransNetV2 (the previous detector in this project)
is biased toward hard cuts. On Tyla's video, it detected 2 shots (one
212-second "shot" + a 2-second tail) when the video clearly has 20+
transitions, mostly whip pans and motion-blur cuts. PySceneDetect's
ContentDetector caught all 72.

**Alternatives considered:**
- **AutoShot** (CVPR 2023, wentaozhu) — SOTA on short-form videos but
  weights are on Baidu Pan (China-only download)
- **CLIP-based scene change** — possible via cloud API, but per-frame
  CLIP cost is prohibitive for a 3-5 min video
- **Color histogram** — PySceneDetect's HistogramDetector, less accurate
  than ContentDetector for gradual transitions

**Output:** `data/processed/shots.json`, `shot_predictions.csv`, `shot_detection_stats.json`

### Phase 2 — Per-shot visual analysis

**What:** Asks a vision-language model 8 questions about the middle frame of
each shot. Results are saved as one row per shot with 8 caption columns.

**Tool:** **Gemini 3 Flash** (Google) via Ollama cloud API, or any
vision-capable Ollama model locally.

**The 8 questions** (each answered with `temperature=0, seed=42` for reproducibility):
1. **caption** — 1-2 sentence description
2. **camera** — ONE of: static / pan / tilt / zoom-in / zoom-out / dolly / tracking / handheld / unknown
3. **emotion** — 1-2 word emotion label
4. **colors** — top 3 prominent colors
5. **entities** — list of objects + people
6. **location** — indoor/outdoor + setting
7. **lighting** — 3-5 word description
8. **composition** — 3-5 word description (framing)

**Why 8 questions, not 1 caption?** A single caption doesn't separate
emotion from camera from content. Asking 8 focused questions gives
structured, queryable data: you can group shots by emotion, filter
by camera motion, or aggregate "all indoor shots."

**Why `temperature=0, seed=42`?** The taylor-swift-lyrics-nlp project
(round 8 audit) found that without a seed, the LLM pass was
non-reproducible. We pin both for full determinism.

**Why one mid-frame per shot?** Sending every frame would multiply cost
by ~30× (avg shot is ~3s = 72 frames at 24fps). The mid-frame is the
most representative single image of a shot.

**Cloud vs. local:**
- **Cloud** (gemini-3-flash-preview, gemma3:27b, gemma4:31b): 1-2s per
  call, no RAM limit, costs API credits. Set `OLLAMA_API_KEY` in `.env`.
- **Local** (gemma3:4b, qwen2.5vl:3b, qwen3-vl:8b): 2-180s per call,
  limited by RAM (qwen2.5vl:3b needs 27 GB to load, 7b needs 47 GB).
  No API cost but slow on CPU-only machines.

**Output:** `data/processed/shot_vision.csv`

### Phase 3 — Camera movement (OpenCV optical flow)

**What:** For each pair of consecutive frames within a shot, computes the
optical flow field and classifies it as pan / tilt / zoom / static / handheld.

**Tool:** **OpenCV's `calcOpticalFlowFarneback`** (a classical CV algorithm,
not a neural network).

**How it works:**
1. Convert both frames to grayscale
2. Compute dense optical flow (motion vector per pixel)
3. Classify:
   - `static` — median flow magnitude < 0.3
   - `pan-left` / `pan-right` — horizontal flow dominant
   - `tilt-up` / `tilt-down` — vertical flow dominant
   - `zoom-in` / `zoom-out` — radial divergence (flow vectors pointing
     inward or outward from center)
   - `handheld` — non-zero flow but no clear direction (small camera shake)

**Why classical CV here?** Asking the VLM "is this a pan?" 72 times would
be slow and inconsistent. Optical flow is fast (<1s per shot on CPU)
and gives continuous scores (pan_left_score, tilt_score, zoom_score) you
can use to detect *transitions* (a shot that pans more than expected).

**Why not a neural flow estimator (RAFT, FlowNet)?** They're more accurate
but PyTorch inference on CPU is 5-10x slower. OpenCV's Farneback is
"good enough" for shot-level classification.

**Output:** `data/processed/shot_camera.csv` with `camera_motion`,
`pan_score_mean`, `tilt_score_mean`, `zoom_score_mean` per shot.

### Phase 4 — Transcription (faster-whisper)

**What:** Transcribes speech and sung vocals to text with timestamps.

**Tool:** **faster-whisper** (CTranslate2 backend, ~4x faster than
openai-whisper Python, MIT license).

**Model sizes** (`--model`):
- `tiny.en` (39M params) — fastest, ~0.5s/30s audio, 75% accuracy
- `base.en` (74M) — good balance, ~1s/30s
- `small.en` (244M, **default**) — best for music, ~2s/30s
- `medium.en` (769M) — overkill for most, 6GB RAM
- `large-v3` (1.5B) — best accuracy, needs 10GB RAM

**How it works:**
1. Audio is split into 30s chunks
2. Each chunk is encoded with mel-spectrogram features
3. A transformer decoder predicts tokens autoregressively
4. VAD (voice activity detection) filters out non-speech
5. Timestamps are recovered from the cross-attention weights

**Known limitation:** Whisper is trained on speech, not music. On
heavily reverbed or whispered vocals (e.g. Radiohead's *Daydreaming*),
it often returns 0 segments. This is a real finding, not a bug —
documented in the dashboard's "data quality" section.

**Output:** `data/processed/transcript.csv` (one row per segment)

### Phase 5 — Audio tagging (CLAP)

**What:** For each 5-second audio window, computes similarity scores
against a fixed vocabulary of 27 tags (12 mood + 7 section + 8 instrument).

**Tool:** **CLAP** (laion/clap-htsat-fused) via HuggingFace transformers.

**The 27 tags:**

Mood (12): `happy and bright`, `sad and melancholic`, `aggressive and intense`,
`romantic and tender`, `triumphant and epic`, `calm and peaceful`, `tense and anxious`,
`dreamy and ethereal`, `dark and ominous`, `playful and whimsical`,
`lonely and introspective`, `powerful and confident`

Section (7): `intro`, `verse`, `chorus`, `bridge`, `outro`,
`instrumental break`, `vocal only`

Instrument (8): `acoustic guitar`, `electric guitar`, `piano`,
`drums and percussion`, `bass guitar`, `synthesizer`, `strings orchestra`,
`vocal only no instruments`

**How it works:**
1. Audio is resampled to 48 kHz (CLAP's required rate)
2. Audio and tag text are encoded into a shared embedding space
3. Cosine similarity between audio embedding and each tag embedding
4. Softmax over similarities → probability distribution

**Why this approach?** CLAP was trained on 400k audio-caption pairs to
align audio and natural language in the same space. This means
zero-shot tagging: you can ask "is this dark?" without ever training
on "dark music" examples.

**Output:** `data/processed/audio_clap.csv` (one row per 5s window,
27 columns of tag probabilities)

### Phase 6 — Music structure (librosa)

**What:** Extracts tempo, beat positions, estimated key, and per-second
loudness/brightness from the audio.

**Tool:** **librosa** (BSD-3-Clause, the de-facto Python audio analysis library).

**How it works:**
- **Tempo + beats** — onset strength envelope → autocorrelation → BPM.
  Beat positions are the local maxima of the onset envelope.
- **Key** — 12-bin chroma (which pitch class is active) → Krumhansl-Schmuckler
  template matching → most likely major or minor key.
- **RMS energy** — root-mean-square amplitude per second (loudness).
- **Spectral centroid** — "brightness" of the spectrum.
- **Spectral contrast** — difference between spectral peaks and valleys.

**Why chroma + Krumhansl-Schmuckler?** Simple, fast, works for 90%+ of
popular music. Fails on atonal/ambient (no clear key) and complex
jazz (chord changes too fast). For more accuracy, use Essentia's
key detection or a chroma-CNN model.

**Output:** `data/processed/music_features.csv` (one row per second),
`music_summary.json` (tempo, key, beats list)

### Phase 7 — Temporal synchronization

**What:** Joins all 7 streams onto a single timeline. For each shot,
computes the average audio mood, music energy, lyrics overlap, and
whether the cut landed on a beat.

**Cross-modal signals computed:**
- `cut_on_beat` — did the shot's start fall within 100ms of a beat?
- `audio_mood_<tag>` — mean probability of each CLAP tag across the
  shot's time range
- `music_avg_rms` — mean loudness during the shot
- `n_lyric_segments` — number of transcript segments that overlap the shot
- `lyric_text` — concatenated lyrics text (if any)

**Output:** `data/processed/sync_per_shot.csv` (one row per shot with
~50 columns), `sync_stats.json` (aggregate stats)

### Phase 8 — Interactive HTML dashboard

**What:** Renders a single-file Plotly HTML dashboard with:
- Synchronized 4-track timeline (shots / CLAP mood / RMS+beats / lyrics)
- Per-modality breakdowns (emotion distribution, camera pie, mood averages)
- Per-shot detail table (all shots, color = emotion)
- Auto-computed "honest findings" (no hardcoded values)
- Data quality section (which streams had data)
- Methodology caveats

**Why single-file?** No server, no dependencies, no build step. Double-click
to open. Easy to share.

**Why no `aria-label` etc.?** Accessibility is partially implemented
(viewport meta, lang, h1). Full WCAG is a larger project.

**Output:** `reports/dashboard.html`

## Setup

```bash
git clone https://github.com/zaheralkaei/multimodal_analysis
cd multimodal_analysis
python -m venv .venv

# Install ffmpeg (Linux: apt install ffmpeg; macOS: brew install ffmpeg;
# Windows: download from https://www.gyan.dev/ffmpeg/builds/)
export PATH="/path/to/ffmpeg/bin:$PATH"

# Install python deps
.venv/bin/pip install -r requirements.txt

# Create your .env
cp .env.example .env
# Edit .env: paste your OLLAMA_API_KEY from https://ollama.com/settings/keys
```

## Run

```bash
# Phase 0: download + extract
.venv/bin/python scripts/phase0_input.py "https://www.youtube.com/watch?v=..." --fps 1

# Phases 1-6: feature extraction (can run independently in any order)
.venv/bin/python scripts/phase1_shots.py                       # ~30s
.venv/bin/python scripts/phase2_vision.py                      # ~5-30 min (cloud, depends on shots)
.venv/bin/python scripts/phase3_camera.py                      # ~1-2 min
.venv/bin/python scripts/phase4_transcribe.py --model small.en  # ~3-5 min
.venv/bin/python scripts/phase5_audio.py                       # ~2-3 min
.venv/bin/python scripts/phase6_music.py                       # ~30s

# Phase 7: join
.venv/bin/python scripts/phase7_sync.py                        # instant

# Phase 8: dashboard
.venv/bin/python scripts/phase8_dashboard.py                   # instant
open reports/dashboard.html
```

## Configuration

All config is in `.env` (gitignored). See `.env.example` for the schema.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_API_KEY` | (none) | Required for cloud vision. Get at https://ollama.com/settings/keys |
| `OLLAMA_BASE_URL` | `https://ollama.com/api` | Cloud endpoint; use `http://localhost:11434` for local ollama |
| `VISION_MODEL` | `gemini-3-flash-preview` | Cloud model. Other options: `gemma3:27b`, `gemma4:31b` |

## Comparing different models

Each phase's output is in a separate CSV/JSON. To compare models:

1. **Different shot detector**: re-run phase 1 with a different threshold,
   compare `shot_detection_stats.json` files.
2. **Different vision model**: change `VISION_MODEL` in `.env`, re-run
   phase 2, compare `shot_vision.csv` rows for the same shot.
3. **Different whisper model**: re-run phase 4 with `--model medium.en`,
   compare `transcript.csv` segments.

The dashboard recomputes everything dynamically, so different runs
are directly comparable.

## Limitations

- **Single video** — currently the pipeline is designed for one video.
  Running on N videos would need a wrapper to loop and aggregate.
- **CPU-only tested** — works on CPU. GPU would speed up phase 2
  (vision) and phase 5 (CLAP) significantly.
- **English-only transcription** — Whisper supports 99 languages, but
  VAD filter and small.en model are English-optimized.
- **Modern music videos** — tuned for music videos with cuts, lyrics,
  and clear beats. Will produce sparse results on:
  - Slow art films (no lyrics, long static shots)
  - Live performances (no shot boundaries, just one long take)
  - Abstract/music-only videos (no vocal, no scenes)

## License

Pipeline code: MIT. Data sources: see [RESEARCH.md](RESEARCH.md) for
each model's license.
