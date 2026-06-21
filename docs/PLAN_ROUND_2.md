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

### Part A — Fix data quality bugs (3-5 hours)

A1. **Structured-output prompting + post-parse.** Change phase2 to send a JSON schema with each question. Use `format: "json"` or include a "respond with ONLY valid JSON" instruction. Parse the response and reject malformed rows. **Expected:** emotion count drops from 34 to ≤12.

A2. **Bump `num_predict` and use `stop` tokens.** Increase from 300 to 800 so captions complete, and add `stop: ["\n\n", "###"]` to stop Gemini from continuing into another question. **Expected:** 0 truncated captions.

A3. **Emotion normalization.** Build a small synonym map: `{sensual: romantic, sultry: romantic, seductive: romantic, ...}` (maybe 30-50 entries). Apply after Gemini returns. **Expected:** 34 → 12 distinct emotions.

A4. **Filter zero-variance CLAP tags post-hoc.** In phase 5, mark tags as "active" only if max score across windows > 0.1. Surface "inactive" in the dashboard. **Expected:** cleaner CLAP section.

A5. **Save continuous camera motion scores.** Add `pan_score`, `tilt_score`, `zoom_score` per shot (already in `shot_camera.csv` actually — verify it's being read by phase 7). Plot these as scatter or violin in the dashboard.

### Part B — Compare to end-to-end VLM (4-6 hours)

B1. **Run InternVideo2.5 / VideoLLaMA3 on the same video.** Download via HuggingFace transformers. Ask the same 8 questions. Compare answers shot-by-shot with Gemini's. **Expected:** see how a true video model vs. mid-frame VLMs differ (esp. on camera motion which a single frame can't capture).

B2. **Add a phase 9 — narrative summary.** Take our structured per-shot data + the video, ask InternVideo2.5 to write a 200-word narrative. Compare with what the same model would write from just frames. This tests whether structured inputs help.

B3. **Cost / time comparison table.** InternVideo2.5 8B on a 3:35 video at 24fps = 5163 frames. On a 16GB GPU it's 30-60s. On CPU it's ~30 min. Document this.

B4. **Don't replace our pipeline.** InternVideo2.5 gives free-text, we give structured CSVs. They're complementary. Keep ours as the primary, add InternVideo2.5 as a "phase 9 narrative" option.

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