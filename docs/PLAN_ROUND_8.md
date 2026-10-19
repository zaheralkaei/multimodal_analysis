# Round 8 plan — review findings, model strategy, upgrade roadmap

Date: 2026-10-19 · Status: **validity fixes done (8a); model upgrades W0–W3 awaiting approval**

## 8a — done 2026-10-19: test-before-spend + analysis-validity fixes

All offline; 159 tests pass; both videos re-run end-to-end on copies
(`--id dry_rtw`, `--id dry_z2k`, `--provider mock`).

| Fix | Evidence |
|---|---|
| Phase 3 camera signs were **inverted on every axis** (pan right → "pan-left", tilt down → "tilt-up", zoom in → "zoom-out"); replaced by KLT + RANSAC similarity at 8 fps inside each shot, cut frames excluded, camera vs subject separated, `camera_confidence` flag | synthetic-motion tests `tests/test_camera.py`; dry run: 60/146 shots high-confidence on rtwpk9rb1Dc |
| Cut-on-beat now tested against chance (exact per-cut Poisson-binomial; Rayleigh phase test; circular-shift sensitivity) and shot 0 no longer counted as a cut (R1) | rtwpk9rb1Dc 42.8 % vs 42.2 % chance, p = 0.45; Z2ki180nHCI 50 % vs 45 %, p = 0.31 — **neither video cuts on the beat more than chance** |
| 48 kHz full-band `audio_music.wav` for CLAP + librosa (R2); CLAP softmax per tag group + raw cosine file (R3) | instrument tag flipped electric→acoustic guitar on the same song: CLAP instrument tags are not robust |
| Whisper `suspect` flags (compression ratio / no-speech heuristics), `condition_on_previous_text=False`, word timestamps (H4) | |
| Phase 7 refuses stale per-shot rows (start-time check) | |
| Phase 2: `--provider mock`, `--dry-run`, `--max-usd` (estimate + live spend cap), retry/backoff, model-slug check, no `"\n\n"` stop, `response_format`, per-row `status`/`model`, error rows retried (H3) | |
| `scripts/preflight.py [--online]`: env, packages, ffmpeg, caches, key, model exists/takes images/params/price | Gemma 4 31B: $0.20/$0.40 per M → ≈ $0.04–0.08 per video |
| `run_manifest.json`: git commit, package versions, exact phase commands, input SHA-256 | |
| ffmpeg found via imageio-ffmpeg (not on PATH here); OpenCV metadata fallback without ffprobe; UTF-8-safe output (phase 5 crashed when piped); runner exit code non-zero on phase-0 failure (web UI showed failed downloads as "done") | |
| `min_scene_len` 30 → 15 frames: 72 → 146 shots on rtwpk9rb1Dc. Spot-check of 8 new cuts: 5 real, 3 false (light flare, motion). **Open — gold labels decide** | |

Still unvalidated: every threshold (shots, camera) — the gold set (W0) is next.

Constraints agreed with the user:
- Compute: this machine is CPU-only (torch `+cpu`, no CUDA). Cloud APIs via
  OpenRouter are allowed, budget **≈ $1 per video**.
- Purpose: **both** — validated core metrics *and* rich descriptive layers.
- The user will hand-label a small gold set (~1 h) so model choices are measured, not assumed.
- Scope of this round: camera + shots, vision model, audio/music, cross-modal statistics.

---

## 1. Should all decisions go to a vision model (Gemma / Qwen)?

**No for measurement, yes for interpretation.** The 2026 evidence is consistent:

| Task | Best general VLM result | Specialist / geometric result | Verdict |
|---|---|---|---|
| Camera motion (CamChoreo, 20 labels, frame micro-F1) | Gemini-3.1-Pro 43.3 · GPT-5.4 38.0 · Qwen3-VL-235B 33.4 · Qwen3-VL-8B 28.3 | CamDistill-4B (Qwen3-VL fine-tuned with geometry) 67.5 | General VLMs lose; fine-tuned + geometry wins |
| Camera motion (ACaM, 17 classes) | Qwen3-VL-32B and Gemini-3-Pro near/below chance on low-intensity push-in; label moving cameras "static" | fine-tuned Qwen3-VL-8B beats Gemini 3.1 Pro by ~10% | Same failure we saw: 70/72 "static" |
| Camera motion (CameraBench, NeurIPS 2025) | VLMs miss *geometric* primitives (trajectories) | SfM misses *semantic* primitives (e.g. "follows the subject") | Hybrid is the right design |
| Shot boundaries | TransVLM (Mar 2026) uses a VLM **plus optical flow** for gradual transitions | TransNetV2 / AutoShot still the standard for cuts | Keep a specialist detector |
| Captions, setting, lighting, shot size, entities, OCR, scene grouping | Strong (this is what VLMs are good at) | — | VLM |

Gemma 4 specifically has **no video encoder**: video is split into ≤32 timestamped
frames at 70 tokens each through the image encoder — fine for semantics, weak for motion.
Qwen3-VL has text-timestamp alignment (better temporal grounding) and is cheap on
OpenRouter (8B $0.25/$0.75, 235B-A22B $0.20/$0.88 per M tokens). Gemini 3 Flash
accepts native **video + audio** ($0.50/$3.00 per M).

**Design principle for round 8:** *specialists measure, the VLM interprets.*
Geometry/DSP produce numbers (motion vectors, beats, boundaries); the VLM gets those
numbers as context and produces the semantic layer. We never ask a VLM a question a
measurement can answer, and we score every VLM claim against the gold set.

---

## 2. Code review findings (verified against code + current outputs)

### Critical — the numbers are wrong or meaningless today
| # | Where | Finding | Evidence |
|---|---|---|---|
| R1 | [phase7_sync.py:90](../scripts/phase7_sync.py#L90) | **"Cuts on beat" is at chance level.** ±100 ms window at ~130 BPM covers ~40% of the timeline, so random cuts score ~40%. No null model. | rtwpk9rb1Dc: 48% observed vs 41±6% random; Z2ki180nHCI: 41% vs 45±9% — headline stat is noise |
| R2 | [phase0_input.py:139](../scripts/phase0_input.py#L139) | Audio is extracted **once at 16 kHz mono** and reused for music. CLAP then upsamples to 48 kHz ([phase5_audio.py:49](../scripts/phase5_audio.py#L49)) — everything above 8 kHz is gone (cymbals, air, brightness). Spectral centroid/contrast in phase 6 are capped too. | ffmpeg `-ar 16000` |
| R3 | [phase5_audio.py:73](../scripts/phase5_audio.py#L73) | **One softmax over 27 mixed tags** (12 moods + 7 sections + 8 instruments) — "piano" competes with "sad". Section tags ("verse", "chorus") from an isolated 5 s window are not a meaningful CLAP question. | 21/27 tags flat-zero noted in round 2 is a symptom |
| R4 | [phase3_camera.py:42](../scripts/phase3_camera.py#L42) | The first frame of every shot's range is sampled **before** `start_sec` → the first flow pair usually **spans the cut** and injects a huge fake motion. | `first = floor(start*fps)+1` → t ≤ start |
| R5 | [phase3_camera.py](../scripts/phase3_camera.py) | Flow on frames **0.5 s apart** (2 fps): Farneback breaks on large displacements; global mean flow mixes **subject motion with camera motion**; "handheld" is just the fallback label ([:142](../scripts/phase3_camera.py#L142)), not a shake measure; `/5.0` thresholds are resolution-dependent. | rtwpk9rb1Dc: "tilt-down" is the most common label (17/72) — implausible for a music video |
| R6 | docs | **No ground truth anywhere.** VLM camera accuracy was scored against OpenCV, which is itself unvalidated; the "85% vs 55%" in METHODOLOGY_REVIEW has no labeled source. | — |

### High
| # | Where | Finding |
|---|---|---|
| H1 | [phase1_shots.py:156](../scripts/phase1_shots.py#L156) | `min_scene_len=30` frames (~1.25 s) — music videos routinely cut faster; shortest detected shot is exactly 1.29 s. Sub-second cuts are systematically merged. TransNetV2 "found 2 shots" is almost certainly a usage bug, not a model property. |
| H2 | [phase2_vision.py:57](../scripts/phase2_vision.py#L57) | VLM gets **one mid-frame**; the `camera` question on a still is guaranteed noise (70/72 and 35/35 "static"). Emotion from one frame is fragile. |
| H3 | [phase2_vision.py](../scripts/phase2_vision.py) | No `response_format` JSON schema, no retry/backoff on 429/5xx, stop sequence `"\n\n"` ([:80](../scripts/phase2_vision.py#L80)) can truncate pretty-printed JSON. |
| H4 | [phase4_transcribe.py](../scripts/phase4_transcribe.py) | Whisper `small` on the **full mix** (no vocal separation), no word timestamps, and the hallucination signals it already records (`no_speech_prob`, `compression_ratio`) are never used to filter. |
| H5 | [phase6_music.py:69](../scripts/phase6_music.py#L69) | One global tempo, no downbeats, no song sections; key from a whole-song `chroma_stft` average. |

### Medium / hygiene
- [phase1_shots.py:190-199](../scripts/phase1_shots.py#L190) — `shot_predictions.csv` loop is O(frames × shots).
- Dashboard "VLM vs OpenCV camera agreement" panel compares two unvalidated sources — should become "vs gold".
- `_test_*.py` experiments use Ollama/Gemini 3 Flash of June 2026; results predate the current models.

---

## 3. Target architecture

```
                 ┌─ shots: TransNetV2 (CPU ONNX) ⊕ PySceneDetect  → shots.json (+ transition type)
video ──┬────────┤
        │        └─ camera: per-shot dense sampling (8 fps) → KLT tracks + RANSAC similarity/homography
        │                   → pan/tilt/zoom/roll magnitudes, shake (jitter), subject-motion ratio
        │
        ├─ VLM (OpenRouter): 4–6 timestamped frames per shot + phase-3 numbers as context
        │     → caption, shot size, angle, setting, lighting, entities, on-screen text, scene id
        │     → "camera" becomes semantic only (e.g. "follows subject"), never the geometric label
        │
        └─ audio 44.1 kHz stereo master
              ├─ Demucs (htdemucs, CPU) → vocals / accompaniment stems
              ├─ vocals → Whisper large-v3-turbo (word timestamps, hallucination filter)
              ├─ mix → Beat This! (beats + downbeats) · SongFormer/allin1 (sections) · key per section
              ├─ mix → CLAP music checkpoint, grouped softmax (mood | instrument) — sections removed
              └─ per section → audio-LLM (Gemini Flash / Qwen Omni Flash) description: mood, genre, instrumentation
                                                         │
                         sync: null-model statistics (cuts vs beats/downbeats, energy vs cut rate, sections vs scenes)
```

---

## 4. Workstreams (in order)

### W0 — Evaluation harness + gold labels *(first; everything else is scored against it)*
1. `scripts/label_app` route in the existing Flask UI: per-shot clip player, keyboard
   shortcuts for camera label (static / pan L/R / tilt U/D / zoom in/out / dolly / track /
   handheld / compound) and "missed cut here" / "false cut" marks. Saves `gold/<id>.json`.
2. User labels ~100 shots over 2–3 videos (the two existing + one with fast cutting).
3. `scripts/evaluate.py`: shot-boundary F1 (±1 frame / ±0.5 s), camera macro-F1 + confusion
   matrix, lyric WER when a reference lyric file is supplied. Writes `eval/<id>.json`.
4. Baseline scores of the current pipeline recorded before any change.

### W1 — Shots + camera motion (CPU)
1. Shots: add TransNetV2 (ONNX/PyTorch CPU build) next to PySceneDetect; lower
   `min_scene_len` to ~8 frames; union/vote strategy chosen by gold F1.
2. Camera: decode each shot's own frames at ~8 fps directly from the video (ffmpeg seek),
   skipping a guard band at both ends (fixes R4/R5).
3. Replace mean dense flow with **sparse KLT tracking + RANSAC similarity/homography**:
   inliers = background (camera), outliers = subjects. Output continuous pan/tilt (deg-ish
   normalized by frame size), zoom (scale), roll, shake (high-frequency residual), and
   subject-motion share. Label from these with thresholds tuned on gold.
4. Keep Farneback path as fallback; compare both on gold.
5. *Optional later (needs GPU or hosted endpoint):* CamDistill-4B — not in this round.

### W2 — Vision model upgrade (OpenRouter)
1. Multi-frame input: 4–6 frames per shot with `t=mm:ss.s` labels, JSON-schema
   `response_format`, retry with backoff, no `"\n\n"` stop.
2. New fields: shot size (ECU…EWS), camera angle, on-screen text, people count,
   performance vs narrative, `scene_id` (shots sharing location/characters).
   Phase-3 measurements passed in the prompt; VLM `camera` field becomes "semantic motion"
   (follows subject / reveals / static framing) only.
3. **Bake-off** script on the gold videos: `google/gemma-4-31b-it` vs Qwen3-VL
   (8B and 235B-A22B) vs `gemini-3-flash` (native clip incl. audio). Scored on gold
   (shot size, setting, people count, camera-semantic) + a blind side-by-side caption
   rating page for the descriptive fields. Default model chosen from results.
4. Cost guard: estimate tokens before the run, abort if > `--max-usd` (default 1.00).

### W3 — Audio & music (CPU + optional audio-LLM)
1. Phase 0 keeps a **44.1 kHz stereo** master; 16 kHz mono derived only for ASR (fixes R2).
2. Demucs htdemucs vocal separation (cached; ~1–3 min/song on CPU).
3. Whisper `large-v3-turbo` int8 on the vocal stem, `word_timestamps=True`, drop segments
   with high `no_speech_prob` + low `avg_logprob` / high `compression_ratio`. `small` kept
   as a fast flag. (VocalParse — Qwen3-ASR-1.7B fine-tuned for singing, May 2026 — evaluated
   in the bake-off if it runs on CPU.)
4. Beat This! for beats **and downbeats**; local tempo curve.
5. Song sections: SongFormer (Oct 2025, beats All-In-One on HarmonixSet) if it runs on CPU,
   else allin1; fallback: audio-LLM section labeling. Key per section.
6. CLAP: switch to a music-trained checkpoint, **separate softmax per tag group**, remove
   section tags (sections now come from step 5). Report raw cosine too.
7. Per-section audio-LLM description (Gemini Flash audio or Qwen 3.8 Omni Flash on
   OpenRouter, $0.15/$0.47 per M): mood, genre, instrumentation, vocal style — shown as
   text, flagged as model opinion.
8. Speech content: pyannote diarization (who speaks when) behind `--content-type speech/film`.

### W4 — Cross-modal statistics (fixes R1)
1. Cut-on-beat and cut-on-downbeat reported with a **circular-shift null** (shift the cut
   train, keep its structure) → observed %, chance %, z/p-value; plot beat-phase histogram.
2. Cut rate vs RMS/energy (per section), shot length vs tempo, section boundary ↔ scene
   change alignment, lyric line onsets ↔ cuts. Every stat ships with its chance baseline.
3. Dashboard shows "not distinguishable from chance" instead of a bare percentage.

### W5 — Dashboard, docs, tests
- New tracks: downbeats, sections, vocal activity, scene ids, camera magnitude curves.
- Replace VLM-vs-OpenCV panel with "model vs gold" when gold exists.
- README model table + CAMERA_DETECTION.md + METHODOLOGY_REVIEW.md rewritten (remove the
  unsourced 85%/55% claim).
- Tests for each new pure function (null model, RANSAC classifier on synthetic flows,
  grouped softmax, guard-band frame ranges, cost estimator).

---

## 5. Cost per video (≈4 min song, ~70 shots)

| Item | Tokens (approx.) | Cost |
|---|---|---|
| VLM, 5 frames × 70 shots (Qwen3-VL-235B or Gemma 4) | ~0.3–0.5 M in, 30 k out | $0.07–0.15 |
| Gemini 3 Flash native clips (bake-off / optional) | ~80 k in, 30 k out | ~$0.13 |
| Audio-LLM per section (~8 sections) | ~20 k in, 5 k out | < $0.02 |
| **Total default** | | **≈ $0.10–0.30** (bake-off run ≈ $0.60) |

CPU time adds ~5–10 min (Demucs + Whisper large-v3-turbo + dense camera sampling).

---

## 6. Risks / open items
- SongFormer, Beat This!, TransNetV2 and VocalParse must be verified to install and run on
  Windows CPU; each has a named fallback.
- OpenRouter's video-input request format must be checked against current docs before
  W2.3's Gemini path; frames-as-images works everywhere.
- Gold set is small (~100 shots): good for choosing between models, not for publishable
  accuracy claims — publishing would need a larger set or a public benchmark slice.
- Model listings deprecate fast (Qwen3-VL-32B deprecation notice seen on OpenRouter) —
  model slugs stay configurable in `.env`.
- Phase 2 re-run for both existing videos is still pending an `OPENROUTER_API_KEY`.

## 7. Sources
- CamChoreo / CamDistill (2026): https://ddz16.github.io/cammotion.github.io/ · https://arxiv.org/pdf/2608.10932
- ACaM, Natural Language Camera Movement Understanding (2026): https://arxiv.org/html/2607.03043
- MotionBlind (2026): https://www.alphaxiv.org/abs/2609.09528
- CameraBench (NeurIPS 2025): https://arxiv.org/abs/2504.15376
- ShotBench / ShotVL (NeurIPS 2025): https://arxiv.org/html/2506.21356v2
- TransVLM shot transitions (HeyGen, 2026): https://www.heygen.com/research/transvlm
- AutoShot (CVPR 2023): https://arxiv.org/pdf/2304.06116
- Gemma 4 multimodal implementation (vLLM): https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/gemma4_mm/
- Qwen3-VL on OpenRouter: https://openrouter.ai/qwen/qwen3-vl-8b-instruct · https://openrouter.ai/qwen/qwen3-vl-235b-a22b-instruct
- Qwen3.8 Omni Flash on OpenRouter: https://openrouter.ai/qwen/qwen3.8-omni-flash/api · Qwen3-Omni report: https://arxiv.org/abs/2509.17765v1
- Gemini 3 Flash pricing: https://computeprices.com/providers/openrouter/models/gemini-3-flash
- Music Flamingo (ICLR 2026): https://arxiv.org/pdf/2511.10289
- Beat This! (ISMIR 2024): https://huggingface.co/papers/2407.21658
- SongFormer (2025): https://arxiv.org/html/2510.02797v2
- VocalParse (2026): https://huggingface.co/ngoctham/VocalParse
- Music emotion encoders (MERT etc.): https://arxiv.org/pdf/2409.15545
