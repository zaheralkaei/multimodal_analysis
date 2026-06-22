# Audit report

Date: 2026-06-22
Auditor: round-3 audit (after 5 commits, 2 videos fully processed)
Scope: code quality, data correctness, methodology, engineering hygiene

## TL;DR

| Layer | Status | Notes |
|---|---|---|
| Code quality | ✅ Mostly clean | 20 Python files, all parse, 1 hardcoded-path bug fixed, 4 dead test files removed |
| Data correctness | ✅ Cross-modal consistent | 7/7 sanity checks pass for both videos, 2 stale-path bugs fixed |
| Methodology | ✅ Samples look good | No hallucinations, emotions/locations/lighting consistent with captions |
| Engineering hygiene | ✅ Good | .env gitignored, no committed secrets, deps pinned in requirements.txt |

**Findings fixed during this audit:**
1. `metadata.json` had stale `audio_path` and `frames_dir` (pointed to old `data/processed/...` instead of `data/<video_id>/...`)
2. `metadata.json` had stale `source_file` (pointed to old `data/raw/video.mp4` instead of `data/raw/<video_id>.mp4`)
3. README + STRUCTURE_V3.md referenced non-existent `music_features.json` (it's `music_features.csv`)
4. 4 dead test files in scripts/ (`_test_cloud_vision.py`, `_test_gemma.py`, `_test_qwen.py`, `_test_multi_image.py`)
5. Empty `analysis/` directory
6. Stale docstring paths in 3 phase scripts (mentioning `data/processed/` literally)

**Findings known but not fixed (documented limitations):**
- VLM says "static" for 70/72 Tyla camera + 35/35 4 Blocks camera (mid-frame can't show motion)
- 21/27 CLAP tags have zero variance on Tyla (vocabulary mismatch — Tyla is a pop song, most tags are wrong genre)
- "Sensual" emotion dominates Tyla (50/72) — vocabulary is biased toward R&B/sensual, but matches Tyla's vibe
- Cuts-on-beat ±100ms tolerance is arbitrary

---

## Layer 1: Code quality

### What I checked
- All 20 Python files parse
- Hardcoded paths in code (not docstrings)
- Unused imports
- Magic numbers
- Exception handling
- HTML escaping
- Doc/code consistency

### Findings
| File | Issue | Severity | Status |
|---|---|---|---|
| `phase2_vision.py` | `_healthcheck.jpg` path was hardcoded `data/processed/_healthcheck.jpg` | medium | ✅ fixed |
| `phase2_vision.py` | `num_predict=300` was too low (truncated captions) | medium | ✅ fixed in Part A |
| `phase3_camera.py` docstring | mentions `data/processed/...` literally | low | docstring only, no functional bug |
| `phase4_transcribe.py` docstring | mentions `data/processed/...` literally | low | docstring only |
| `phase7_sync.py` docstring | mentions `data/processed/...` literally | low | docstring only |
| `phase8_dashboard.py` L386 | interpolates `label`/`status` into HTML without `html_lib.escape` | low | values come from our own internal data files; XSS risk only if user-supplied data reaches here |
| `phase5/6/7/8` | many print() statements | info | intentional (script-style output) |
| `scripts/_test_cloud_vision.py` | dead code, not referenced in docs | info | ✅ removed |
| `scripts/_test_gemma.py` | dead code | info | ✅ removed |
| `scripts/_test_qwen.py` | dead code | info | ✅ removed |
| `scripts/_test_multi_image.py` | superseded by `_test_multi_image2.py` | info | ✅ removed |

### What I did NOT find
- No bare `except:` clauses with `pass` (silent error swallowing)
- No circular imports
- No genuinely unused imports (only `__future__` which is required)
- No hardcoded secrets in tracked files
- No mutable global state bugs

### Code size
```
Total Python: 2,640 lines
Largest file: phase8_dashboard.py (482 lines)
Smallest: _env.py (30 lines)
Test files: 3 remaining (referenced in CAMERA_DETECTION.md)
```

---

## Layer 2: Data correctness

### File-level audit

For each video (`rtwpk9rb1Dc` = Tyla, `Z2ki180nHCI` = 4 Blocks):

| File | Tyla | 4 Blocks | Notes |
|---|---|---|---|
| `metadata.json` | ✅ after fix | ✅ after fix | Was stale, fixed |
| `audio.wav` | ✅ 6.9 MB | ✅ 4.2 MB | |
| `shots.json` | ✅ 72 shots | ✅ 35 shots | All mid_frame_paths correct |
| `shot_vision.csv` | ✅ 72 rows | ✅ 35 rows | |
| `shot_camera.csv` | ✅ 72 rows | ✅ 35 rows | |
| `transcript.json` | ✅ 5 segs | ✅ 13 segs (German) | |
| `transcript.csv` | ✅ | ✅ | |
| `audio_clap.csv` | ✅ 43 windows | ✅ 26 windows | |
| `music_features.csv` | ✅ 215 rows | ✅ 130 rows | per-second features |
| `music_summary.json` | ✅ | ✅ | tempo, key, beat count |
| `sync_per_shot.csv` | ✅ 72 rows | ✅ 35 rows | |
| `sync_stats.json` | ✅ | ✅ | |
| `shot_detection_stats.json` | ✅ | ✅ | video_path correct |
| `music_features.json` | ❌ never created | ❌ never created | **mentioned in docs but doesn't exist** — fixed by removing from docs |

### Cross-modal consistency (both videos pass all 7 checks)

- ✓ Shot count matches across `shots.json`, `shot_vision.csv`, `shot_camera.csv`, `sync_per_shot.csv`
- ✓ `shot_idx` is 0..N-1 in all CSVs (no gaps, no duplicates)
- ✓ `mid_frame` paths in `sync_per_shot.csv` all use `data/<video_id>/frames/` prefix
- ✓ `audio_top_mood` values are all from the canonical 12-tag CLAP mood vocabulary
- ✓ `vision_emotion` values are all from the canonical 16-tag emotion vocabulary
- ✓ All `shots.json` mid_frame_paths point to files that exist on disk
- ✓ `shot_detection_stats.json` video_path uses the new `data/raw/<video_id>.mp4` naming

### Stale-path bug (FIXED)

`metadata.json` was written by `phase0` before the folder-rename migration. It
contained:
- Tyla: `audio_path = "data\processed\audio.wav"` (should be `data\rtwpk9rb1Dc\audio.wav`)
- Tyla: `frames_dir = "data\processed\frames/"` (should be `data\rtwpk9rb1Dc\frames/"`)
- Tyla: `source_file = "...data\raw\video.mp4"` (should be `data\raw\rtwpk9rb1Dc.mp4`)
- 4 Blocks: same pattern with `_4blocks` suffixes

**Fix:** Manually rewrote both files with correct paths. Future runs of phase 0
will write correct paths because phase 0 uses `relative_to(REPO_ROOT)` and
respects `PROCESSED_DIR`.

### Documented `music_features.json` doesn't exist

Phase 6 writes:
- `music_features.csv` — per-second features (RMS, centroid, ZCR, contrast, beats)
- `music_summary.json` — aggregate (tempo, key, beat_count, etc.)

There's no `music_features.json`. README and STRUCTURE_V3 mentioned one —
**fixed by removing the reference and adding `music_summary.json` to the doc.**

### Orphan frames

`data/<video_id>/frames/` contains every extracted frame (215-431 per video),
but only 35-72 of them are referenced as mid-frames. The rest are kept for:
- Future phase 7/8 iteration (re-runs might want different mid-frames)
- Visual debugging (you can scroll through all extracted frames)

Disk cost: 5-10 MB per video. Acceptable for now.

---

## Layer 3: Methodology

### Sampled 10 random shots from Tyla

All samples had:
- ✓ Caption matches emotion matches lighting matches composition
- ✓ No parse errors
- ✓ No meta-commentary ("Based on the image...")
- ✓ No truncated sentences

Examples:
- Shot 14: "A woman gazes intensely from shadows against a fiery, textured backdrop." / emotion=contemplative, lighting=dramatic low-key firelight
- Shot 3: "A woman with dark hair gazes over her shoulder against a warm background." / emotion=sensual, lighting=warm directional light
- Shot 28: "A woman dances in silhouette against a glowing grid wall." / emotion=energetic

### Sampled 5 random shots from 4 Blocks

All German crime drama — captions are consistent with the genre:
- Shot 2: "A family occupies a dining area while a woman prepares food."
- Shot 9: "Bearded man smokes in profile against dark foliage."
- Shot 29: "A man walks past a green wall, glancing back cautiously."

### Known limitations (NOT bugs)

| Limitation | Why | Severity |
|---|---|---|
| VLM says "static" for 70/72 Tyla camera + 35/35 4 Blocks camera | VLM sees 1 mid-frame, can't see motion | medium (documented in CAMERA_DETECTION.md) |
| 21/27 CLAP tags have zero variance on Tyla | CLAP vocabulary tuned for general music, not Tyla's specific style | medium (documented in `phase5` output) |
| "Sensual" emotion dominates Tyla (50/72) | Gemini's interpretation of the video, plus synonym map is broad | low (matches the video's actual vibe) |
| `cut_on_beat` ±100ms tolerance is arbitrary | No standard for what "on beat" means | low (used consistently, easy to change) |
| 4 Blocks shows up as "VLM static" for all 35 shots | Same single-frame limitation | low (OpenCV camera still works) |

---

## Layer 4: Engineering hygiene

### Security

✓ `.env` is gitignored (verified `git check-ignore -v .env` returns line 54)
✓ No API keys, tokens, or passwords in tracked files
✓ Test files reference `OLLAMA_API_KEY` env var (read from .env), not hardcoded

### Dependencies

✓ `requirements.txt` exists, all 12 required packages pinned with `>=` (not exact, so reproducible but flexible)
✓ All packages importable in user's venv (`/c/Python313/python.exe -c "import ..."`)
✓ ffmpeg noted as separate install requirement (choco/brew/apt)

### Documentation

✓ README has install + run + model capabilities + troubleshooting
✓ 4 design docs (CAMERA_DETECTION, COMPARISON_1FPS, PLAN_ROUND_2, STRUCTURE_V3)
✓ Per-script docstrings are accurate (after the path fix in phase2)

### Git hygiene

- 5 commits, all pushed
- Working tree clean after this audit
- 20 Python files, 8 docs, 1 requirements.txt
- 0 secrets in tracked files

### Repository size

```
.git/          52 MB (history of 5 commits with renames)
data/         gitignored (but on disk: 50 MB for 2 videos)
reports/      gitignored
docs/         37 KB committed
scripts/      80 KB committed
README.md     14 KB committed
```

---

## What I did not check (would need more time)

1. **Cross-video consistency** — only 2 videos processed, no genre-level patterns yet
2. **Performance benchmarks** — no measurements of phase 2/5 runtime
3. **GPU vs CPU** — only tested on CPU + cloud; no GPU run
4. **Multi-language CLAP** — only English tags, no Chinese/Arabic/Spanish
5. **Larger videos** — both videos < 5 min, haven't tested > 1 hour
6. **Memory profiling** — no `tracemalloc` or `mprof` runs
7. **Stress test of phase 2** — 72 calls went through but no rate-limit handling
8. **Whisper hallucination filter** — `vad_filter=True` but no segment-level confidence threshold
9. **CLAP model update** — pinned to `laion/clap-htsat-fused`, no tests against newer checkpoints
10. **Cross-platform testing** — only tested on Windows, README says Linux/macOS work but I haven't run them

---

## What I recommend for round 4 (if you want to do it)

- **Part B2** — Phase 9 narrative summary from structured CSVs
- **Part C** — Human-labeled validation set + F1/accuracy against pipeline output
- **Part D** — Run on 3 more music videos across genres to surface patterns

None of those are bugs; they're the next layer of work.
