# multimodal_analysis — round 2 plan

Date: 2026-06-21
Status: round 1 complete (72-shot Tyla analysis, dashboard, docs, pushed)
Goal: fix data-quality bugs, validate against an end-to-end VLM, build a small human-labeled validation set.

---

## What's wrong with the current data (from inspection, before fixing)

Audit findings on the current `data/processed/` outputs (Tyla video):

| Finding | Severity | Where |
|---|---|---|
| 34 distinct emotion strings returned, only 12 in our palette (Gemini rambles + paraphrases) | High | `shot_vision.csv::emotion` |
| 8/72 rows have Gemini meta-commentary in location ("Based on the image...") instead of the answer | High | `shot_vision.csv::location` |
| 47/72 captions are <50 chars (likely cut off mid-sentence) | Medium | `shot_vision.csv::caption` |
| CLAP: 21/27 tags have variance ≈ 0 (flat zero — those tags are not active for this audio) | Medium | `audio_clap.csv` |
| `sync_per_shot.csv` does not have `mid_frame` (added mid-round but no commit-time check) | Low | already fixed |
| Camera motion is classification (pan-left/right) not continuous score | Low | `shot_camera.csv` |
| No ground-truth comparison for any of the numbers | High | whole pipeline |

## Landscape check (re-verified)

Re-checked GitHub + arxiv + HuggingFace. Findings:

**Closest open-source match:** `SuchethTata/Multimodal-Video-Understanding-System-` (0⭐, 2026-06)
- Same architecture: PySceneDetect → BLIP captioning → Whisper → timeline
- Difference: uses BLIP (smaller) and OpenAI Whisper (not faster-whisper); no CLAP, no music analysis, no beat-sync
- Confirms our architecture is the right shape

**Other matches (all ≤2⭐):**
- `hediKS7/AI-Powered-Video-Scene-Emotion-Analysis` (2⭐) — YOLOv8 + DeepFace + BLIP-2 + Streamlit
- `harinii-b/Multimodal-Fusion-Model-for-Emotion-Detection` (1⭐) — emotion classification with fusion
- `Sibikrish3000/multimodal-video-pipeline` (0⭐) — description generation
- `akashshawdev/Multi-Agent-Video-Insight-Pipeline` (0⭐) — agentic

**End-to-end video LLMs (no CSV output, just free text):**
- `OpenGVLab/InternVideo2_5_Chat_8B` (6.4K⬇, Apache 2.0)
- `DAMO-NLP-SG/VideoLLaMA3-7B` (4.0K⬇, Apache 2.0)

**Academic / dataset:**
- MuVi (Kramer et al., 2022) — 80h music-video dataset, valence/arousal labels. Closest existing benchmark for our use case.

**Industry:** all major video AI is generative (Runway, Sora, Veo, Kling, Hailuo, Luma). **Analysis is underexplored.**

**Conclusion:** Our pipeline's structure is in the right shape. Our weakness is data quality (Gemini outputs) and lack of validation, not architecture.

---

## Plan

### Part A — Fix data quality bugs (DONE)

A1. **Structured-output prompting + post-parse.** ✅ Combined JSON-mode prompt + `parse_json_response()`. 1 call/shot instead of 8.

A2. **Bump `num_predict` and use `stop` tokens.** ✅ num_predict=1500, stop tokens for `\n\n`, `###`, etc.

A3. **Emotion normalization.** ✅ `_normalize_emotion.py` with 16 canonical + ~80 synonyms. 34→6 distinct emotions.

A4. **Filter zero-variance CLAP tags post-hoc.** ✅ Active-tag report in phase5 output.

A5. **Save continuous camera motion scores.** ✅ Already in shot_camera.csv; propagated through phase7.

**Bonus experiment (B-prep):** Tested whether sending the actual video or multi-frame sequences to Gemini improves camera detection. Result: **no**. Single mid-frame is the same quality as 5-frame sequence (Gemini misclassifies tilt as zoom-in). OpenCV optical flow on 50+ frames per shot is genuinely better for camera motion. Decision: **keep optical flow for camera, keep VLM for semantics**. Added VLM-vs-OpenCV comparison table to dashboard.

### Part B — Compare to end-to-end VLM (REVISED after Part A experiment)

**Findings from Part A bonus experiment:**
- Sending the actual video or 5-frame sequences to Gemini does NOT improve camera detection.
  Single mid-frame: 71/72 shots = static (can't see motion from 1 frame).
  5-frame sequence: Gemini misclassifies clear tilt-down as zoom-in.
  YouTube URL: not supported by Ollama cloud, model hallucinated.
- **Decision:** OpenCV optical flow is genuinely better for camera motion. Keep optical flow as primary,
  keep VLM as secondary (semantic understanding). Both signals kept in sync_per_shot.csv.

**Revised Part B scope (still valuable):**

B1. **End-to-end VLM as narrative-summary generator (NOT camera detector).** Ask Gemini 3 Flash to look at the structured per-shot data we've already produced and write a 200-word narrative ("This music video uses mostly close-ups with consistent warm lighting..."). Tests whether structured CSVs + LLM beats pure vision for high-level understanding.

B2. **Local end-to-end VLM alternative.** Try InternVideo2.5 or VideoLLaMA3 on a small clip to see if local models work at all on our machine. They probably won't (CPU + 16GB RAM) but it's worth a 30-min attempt.

B3. **Document the design choice.** Write a `docs/CAMERA_DETECTION.md` explaining why we use OpenCV for camera and not the VLM. Include the failed experiments.

B4. **Skip B1-B2 if user prefers.** They were speculative. The real value from Part A was the data quality fixes (already done). B is "nice to have" for the round 2 close-out.

### Part C — Build a small human-labeled validation set (2-3 hours)

C1. **Pick 5 music videos** across genres: Tyla (pop, already done), Radiohead (art), Daft Punk (electronic), BTS (K-pop), Billie Eilish (moody/indie). Use 3-5 min clips.

C2. **Build a labeling tool.** Simple HTML form: shows a video + a frame, asks "What's the dominant emotion?" with our 12-emotion palette. "Where does the shot boundary fall?" with a slider. Save labels to JSON.

C3. **Label yourself.** Don't need 3 humans for a useful signal — even 1-author labels establish a baseline. Aim for 30 shot boundaries + 30 emotions labeled across the 5 videos.

C4. **Compute agreement.** For each shot: did our pipeline's `start_sec` match the human label within 0.5s? F1 score. For emotions: does our `vision_emotion` match after normalization?

C5. **Write a "Validation" section in the README** with the agreement numbers. If F1 < 0.5, that's a real finding about pipeline limitations. If > 0.8, that's a publishable result.

### Part D — Run on more videos (1-2 hours per video)

D1. After A is done, re-run on Tyla to confirm bug fixes work.

D2. Run on the other 4 videos. Document genre differences:
- Number of shots per minute
- Camera motion distribution
- Cuts-on-beat rate
- Dominant CLAP mood

D3. **Update README's "Current results" section** to be a multi-video table.

---

## Order of operations

1. A1+A2+A3 (the bug fixes) — 1 session, ~2 hours
2. Re-run on Tyla, verify improvements
3. B1+B3 (InternVideo2.5 comparison) — 1 session
4. C1+C2+C3 (build + label) — can be done across multiple sessions
5. D1+D2 (re-run on Tyla, then 4 new videos) — 1-2 sessions
6. Update README, push, close out the round

## What this round does NOT do

- Doesn't switch to an end-to-end VLM as the primary pipeline (inspects > black-box)
- Doesn't add agentic reasoning (that's a separate research direction)
- Doesn't build a web app (we have a self-contained HTML dashboard)
- Doesn't train any models (we're a benchmarking + integration project)

## Risk / open questions

- **Cloud model availability.** We depend on `gemini-3-flash-preview` and Ollama cloud. If either changes API or pricing, parts A/B break.
- **CPU-only constraint.** Running InternVideo2.5 8B on CPU is slow (~30 min/video). May need to use a smaller variant or skip on certain machines.
- **Human labeling.** Author-only labeling introduces bias. If we want publishable results, recruit 2+ labelers. For internal validation, 1 is fine.
- **5 videos might not be enough.** Genre-level patterns might need 20-30 videos. Plan to call this out as a limitation.

## Files I'll touch

- `scripts/phase2_vision.py` — JSON-mode prompting, num_predict, stop tokens
- `scripts/_normalize_emotion.py` — new file, emotion synonym map
- `scripts/phase5_audio.py` — mark inactive tags
- `scripts/phase9_narrative.py` — new file, InternVideo2.5 wrapper
- `data/processed_2fps_v2/` — new baseline after A fixes
- `data/labeled/` — new, human labels
- `reports/dashboard.html` — re-rendered after A
- `README.md` — validation section, multi-video table
- `docs/VALIDATION.md` — new, methodology + agreement scores