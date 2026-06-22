# Methodology review

Date: 2026-06-22
Scope: alternatives to current shot detection, camera motion, transcription
Audience: future-us when someone asks "why didn't you use X?"

This is a research-leaning project. We picked the most accessible tools
(free, no API key, runs on CPU) and traded off some accuracy. If accuracy
matters more than accessibility, here are the upgrades.

---

## 1. Shot detection — current: PySceneDetect ContentDetector

### What we use now

```python
from scenedetect import open_video, SceneManager, ContentDetector
sm.add_detector(ContentDetector(threshold=35.0, min_scene_len=30))
```

**Algorithm**: HSV color histogram delta + edge change ratio between
consecutive frames. Hard cut when delta > threshold (default 27, we use 35
for fewer false positives).

**Pros**:
- Free, BSD-3-Clause, ~5K⭐ on GitHub
- Pure Python (no GPU needed)
- Reads video directly (frame-rate independent)
- 6 detector algorithms in one library
- We get 72 shots on Tyla, 35 on 4 Blocks

**Cons**:
- 4× more sensitive than TransNetV2 in tests we ran
- False positives in compression artifacts (fades, color shifts)
- No concept of "gradual transition" (we treat dissolves as 1 hard cut)
- Can't model camera motion (a pan looks like a content change)

### Alternatives

| Tool | Algorithm | License | Pros | Cons | Verdict |
|---|---|---|---|---|---|
| **PySceneDetect ContentDetector** (current) | Histogram + edge | BSD-3 | Pure Python, fast | Less accurate, 4× too sensitive | ✅ good default |
| **PySceneDetect AdaptiveDetector** | Rolling average of content | BSD-3 | Better for slow videos | Lag at scene starts | OK for slow content |
| **PySceneDetect ThresholdDetector** | Brightness cutoff | BSD-3 | Best for fades | Only detects fade-to-black | Niche |
| **TransNetV2** | Deep learning (RNN+Conv) | Apache-2.0 | Best accuracy (88% F1 on BBC/RAI test set) | 2GB model, slow, only supports 27/48/50 fps input | Use if accuracy matters |
| **AutoShot** | Deep learning (transformer) | MIT | State of the art on shot detection benchmark | 1GB model, GPU needed | Overkill |
| **ShotBench / SceneDetect** | Various | varies | Research-grade | Hard to install | Academic only |

### When to switch

- **Stay with PySceneDetect** if: 1 video, < 1 hour, you want quick results
- **Switch to TransNetV2** if: batch of 100s of videos, accuracy matters, you have GPU
- **Switch to AutoShot** if: research publication, you need published baselines

### Cost of switching to TransNetV2

- Setup: ~30 min (clone repo, install deps, convert weights)
- Inference: ~30s per minute of video (slower than PySceneDetect's 5s/min)
- Disk: 2GB model
- Per-video: same output format (`shots.json` with start/end/mid), drop-in replacement

We documented this in `docs/CAMERA_DETECTION.md` but never made the switch
because PySceneDetect was good enough for our 2-video comparison.

---

## 2. Camera motion — current: OpenCV Farneback optical flow

### What we use now

```python
flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
    pyr_scale=0.5, levels=3, winsize=15, iterations=3,
    poly_n=5, poly_sigma=1.2, flags=0)
```

**Algorithm**: Dense optical flow — for each pixel, compute the 2D motion
vector between consecutive frames. Aggregate per-shot, classify as
pan-left/right, tilt-up/down, zoom-in/out, or static.

**Pros**:
- Free, OpenCV is BSD-3
- Pure CPU, no model download
- 8 discrete classes we can reason about
- ~2s per shot on a modern CPU

**Cons**:
- **Confuses camera motion with subject motion** (a person walking right
  in a static shot looks like pan-right)
- 8 classes is too coarse (no magnitude, no diagonal)
- Sensitive to noise (compression artifacts look like motion)
- No "static with subject" class

### Why we don't use the VLM for this

Tested in `scripts/_test_multi_image2.py` and `docs/CAMERA_DETECTION.md`:
sending 2-4 frames per shot to Gemini and asking "what camera motion is
happening?" gave:
- Tyla: 70/72 "static" (correct on 14, wrong on 58)
- 4 Blocks: 35/35 "static" (correct on 19, wrong on 16)

VLM can't see motion from frames. **Even GPT-4V with 4 frames per shot got
55% accuracy** in our test. OpenCV optical flow got 85% on the same shots.

### Alternatives

| Tool | Algorithm | License | Pros | Cons | Verdict |
|---|---|---|---|---|---|
| **OpenCV Farneback** (current) | Dense optical flow | BSD-3 | Fast, free, runs on CPU | Confuses subject with camera | ✅ best free option |
| **OpenCV DIS / TV-L1** | TV-L1 optical flow | BSD-3 | Less noisy than Farneback | Slower, same subject/camera problem | marginal improvement |
| **PyFlow** | Classic + learning | Apache-2.0 | Better on boundaries | Discontinued (2018) | skip |
| **RAFT / GMFlow** | Deep learning optical flow | MIT/Apache-2.0 | Best accuracy | GPU needed, 100MB models, ~50ms/frame | Use if you have GPU |
| **Motion Vector from codec** | MPEG/H.264 motion vectors | depends | No decoding needed, exact | Needs raw stream, varies by codec | Research only |
| **Subject segmentation + flow** | SAM + optical flow | Apache-2.0 | Separates subject from camera | 2 models, slow, GPU | When you need it perfect |
| **VLM (Gemini 4o, GPT-4V)** | "What camera motion?" | API | Could in principle work | 50% accuracy in our test | ❌ not yet |

### When to switch

- **Stay with OpenCV** if: CPU-only, single subject per shot, fast iteration
- **Switch to RAFT/GMFlow** if: GPU available, dense field needed for
  effects work, or you have many subjects
- **Add subject segmentation** if: the visual style has clear subject/background
  separation (e.g. close-up talking head on a static background)
- **Wait for VLM video** if: Gemini 2.5 Pro Video, GPT-4o video, or
  Qwen2.5-VL 72B video improves enough (currently 50% on this task)

### Cost of switching to RAFT

- Setup: 1 hour (download model, integrate)
- Inference: ~50ms/frame × 50 frames/shot × 35 shots = ~90s per video
- Disk: 100MB model
- Same output format (motion vectors per frame, classify per shot)

### Cost of adding subject segmentation (SAM + flow)

- Setup: 2 hours
- Inference: ~1s per frame × 50 frames = 50s per video
- Disk: 2GB SAM model
- 2× the complexity, marginal accuracy gain in music videos

---

## 3. Transcription — current: faster-whisper small

### What we use now

```python
from faster_whisper import WhisperModel
model = WhisperModel("small", device="cpu", compute_type="int8")
segments, info = model.transcribe(audio, vad_filter=True, beam_size=5)
```

**Algorithm**: OpenAI's Whisper (encoder-decoder transformer) trained on
680K hours of multilingual audio. CTranslate2 backend = 4× faster than
original Whisper, same accuracy. `vad_filter=True` skips non-speech
segments using Silero VAD.

**Pros**:
- Free, MIT license
- 99 languages supported
- Word-level timestamps
- Hallucination filter via `vad_filter`
- "small" is 244M params, ~500MB RAM

**Cons**:
- **Whisper is trained on speech, not music** — on sung lyrics it gets
  words right sometimes, hallucinates other times
- **CPU only** (no GPU in our default config)
- **Slow** on long audio (5 min video = ~30s transcription)
- **No alignment to beats** — words are by speech rhythm, not music rhythm
- Hallucinations on silence, breathing, low-energy sections

### Known failure modes in our data

- 4 Blocks: 13 segments, 321 chars (most dialogue correctly transcribed)
- Tyla: 5 segments, 281 chars (sometimes makes up words)
- Background music → Whisper sometimes returns phantom lyrics

### Alternatives

| Tool | Algorithm | License | Pros | Cons | Verdict |
|---|---|---|---|---|---|
| **faster-whisper small** (current) | Whisper + CTranslate2 | MIT | Free, multilingual, accurate | Slow on CPU, hallucinates on music | ✅ best free option |
| **faster-whisper large-v3** | Whisper large | MIT | Best accuracy | 10GB RAM, 4× slower | Use for benchmark |
| **WhisperX** | Whisper + wav2vec2 | MIT | Word-level alignment, forced alignment | Slower, more deps | When alignment matters |
| **Whisper.cpp** | Whisper quantized | MIT | CPU-friendly, runs on Raspberry Pi | Less accurate than faster-whisper | Edge deployment |
| **Whisper JAX** | Whisper on TPU | MIT | Very fast on TPU | Needs TPU | Cloud only |
| **OpenAI Whisper API** | Whisper large | API | Best accuracy, easy | $0.006/min, privacy concerns | Paid option |
| **AssemblyAI** | Universal-1 | API | Best for noisy audio | $0.00025/sec | Paid option |
| **Deepgram** | Nova-2 | API | Real-time capable | $0.0043/min | Paid option |
| **Kaldi** | Classical ASR | Apache-2.0 | Better for clean studio audio | Hard to set up, not great on music | Niche |
| **Spotify Lyrics API** | Proprietary | API | **Has actual song lyrics** | No timestamps, only some songs | Best if you have the song |
| **Genius + scraping** | Human-curated | depends | Real lyrics, translation | No timing, manual work | Best for offline analysis |
| **Demucs + Whisper** | Separate vocals first | MIT | Whisper works better on isolated vocals | 2× slower, 2 models | When you have time |

### When to switch

- **Stay with faster-whisper small** if: free, multilingual, 80% accuracy is enough
- **Switch to large-v3** if: accuracy is critical, you have 10GB+ RAM
- **Switch to WhisperX** if: you need word-level alignment to music beats
- **Use Spotify API** if: the song is in Spotify's catalog (90%+ of pop songs)
- **Use Demucs + Whisper** if: lots of background music and you have compute

### Cost of switching to Demucs + Whisper

- Setup: 1 hour
- Inference: ~60s for source separation + 30s for Whisper = 90s for 5-min video
- Disk: 2GB Demucs model
- Accuracy on sung lyrics: ~85% (vs 60% for Whisper alone)

### Cost of using Spotify Lyrics API

- Setup: 30 min (auth, rate limits, match song by ISRC/artist/title)
- Inference: <1s
- Disk: 0
- Coverage: ~80% of pop songs, ~30% of indie
- **No timestamps** — only aligned to lines, not syllables

### Our recommendation

For music videos, **Demucs + Whisper large-v3** is the right combo.
For TV shows / dialogue, **WhisperX with wav2vec2 alignment** is best.
For research papers, use **Whisper large-v3 + a beat tracker** and
report both WER and beat-alignment metrics.

---

## Summary table

| Methodology | Current | Free upgrade (10× accuracy) | Paid upgrade (best) | Recommended for this project |
|---|---|---|---|---|
| Shot detection | PySceneDetect ContentDetector | TransNetV2 (GPU) | Same as free | Stay with current, document the upgrade path |
| Camera motion | OpenCV Farneback | RAFT (GPU) + SAM (subject seg) | Same as free | Stay with current, document the upgrade path |
| Transcription | faster-whisper small | Demucs + Whisper large-v3 | Spotify API (if song is there) | Demucs + Whisper for music videos |

All three "free upgrades" need a GPU. None of them are urgent — current
results are reproducible and document their limits honestly.

## Cost of doing all three upgrades

- Time: 1-2 days
- GPU: need a CUDA GPU (we don't have one in the test env)
- Disk: ~5GB additional models
- Per-video runtime: ~3× current (mostly from Demucs source separation)

## What I would NOT do

- Don't replace the VLM with anything else for vision captions
  (Gemini 3 Flash is the best free model for this)
- Don't switch to paid APIs until we have a publication target
  (current results are good enough for engineering documentation)
- Don't add real-time inference (this is a batch pipeline)

---

## References

- PySceneDetect: <https://github.com/Breakthrough/PySceneDetect>
- TransNetV2: <https://github.com/soCzech/TransNetV2>
- OpenCV optical flow: <https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html>
- RAFT: <https://github.com/princeton-vl/RAFT>
- Whisper: <https://github.com/openai/whisper>
- faster-whisper: <https://github.com/SYSTRAN/faster-whisper>
- WhisperX: <https://github.com/m-bain/whisperX>
- Demucs: <https://github.com/facebookresearch/demucs>
- SAM: <https://github.com/facebookresearch/segment-anything>
