# Audit report (round 4)

Date: 2026-10-10
Scope: **correctness of frame-index math vs fps, path contracts, data provenance, crash paths**

Auditor's note on round 3: AUDIT.md checked shots.json with "✓ All mid_frame_paths
correct" and "✓ paths point to files that exist on disk" — but the check was only
*existence*, not *is this the right frame*. That is why the biggest bug in this
report (F1) survived three audit rounds: the paths exist, they just point into the
wrong part of the video.

---

## TL;DR

The pipeline's default extraction rate is 2 fps, but the frame-index arithmetic in
phase 1 and phase 3 still assumes 1 fps. **Every frame the vision model was shown,
and every shot phase 3 classified, comes from roughly *half the timestamp* it was
supposed to — i.e., different footage, not just a slightly different frame** (F1).
All visual analysis of both existing videos is built on wrong frames.

| # | File | Issue | Severity | Fix |
|---|------|-------|----------|-----|
| F1a | `phase1_shots.py:76` | `mid_frame_idx = int(mid_sec) + 1` assumes 1 fps; at 2 fps the picked frame is at ~mid_sec/2 — wrong footage | **critical** | convert seconds → frame via extraction fps |
| F1b | `phase3_camera.py:129-133` | same assumption: shot `start_sec..end_sec` mapped 1:1 to frame file numbers → optical flow runs on footage from ~half the time, often crossing real shot boundaries | **critical** | same fix |
| F2 | `phase0_input.py:40` | downloads hardcode `data/raw/video.mp4`, not `data/raw/<video_id>.mp4` (README contract) → run_pipeline's `--video` injection never fires; videos overwrite each other | high | save as `<video_id>.mp4` |
| F3 | `run_pipeline.py:141-147` | local-video runs never have `data/raw/<id>.mp4` → phase 1 silently falls back to default `data/raw/video.mp4` = whatever the *last* downloaded video was | high | pass the phase-0 `source_file` through metadata.json |
| F4 | `phase8_dashboard.py:377` + `phase2_vision.py` | the vision model actually used is **never recorded** by any phase; dashboard falls back to hardcoded `gemini-3-flash-preview` / "Gemini 3 Flash" text → dashboards can misattribute results | medium | phase 2 writes `vision_model` into stats |
| F5 | `phase8_dashboard.py:39-50` | `EMOTION_COLORS` is missing 6 canonical emotions (`sensual`, `energetic`, `intense`, `fearful`, `surprised`, `disgusted`) → they render gray; "sensual" is the *dominant* emotion for Tyla | medium | sync the color map with `_normalize_emotion`'s 16 |
| F6 | `phase8_dashboard.py:157,176,241,145` | crash paths: missing `music_features.csv`/`transcript.csv` → FileNotFoundError; 0 shots → ZeroDivisionError; single CLAP window → `statistics.variance` error; header-only clap CSV → `clap[0]` IndexError | medium | guard like `clap_path` is guarded |
| F7 | `_renormalize_existing.py:11` | ignores `PROCESSED_DIR` env → broken since the v3 per-video restructure (reads `data/processed/` which no longer exists) | medium | honor `PROCESSED_DIR` like every other script |
| F8 | `phase4_transcribe.py:63` | argparse default still `small.en` while `run_pipeline.py` passes `small` and the transcribe() signature defaults to `small` — standalone runs differ from pipeline runs (round 2 fixed only the signature) | medium | `default="small"` |
| F9 | low sev., listed below | dead code, ignored args, duplicated tag list, eval(), doc drift | low | see below |

### Round-4 fix status (all fixed; 57 tests in `tests/`)

| Finding | Status | Where |
|---|---|---|
| F1a | ✅ fixed — `mid_frame_index()` fps-aware + bounds-check + nearest-existing fallback | `phase1_shots.py`; verified: all 72/35 mid-frames of both videos now inside their shots (was 71/72 wrong) |
| F1b | ✅ fixed — `shot_frame_range()` floor/ceil over extraction fps | `phase3_camera.py` |
| F2 | ✅ fixed — downloads to `data/raw/<video_id>.<ext>` + deterministic `resolve_downloaded()` | `phase0_input.py` |
| F3 | ✅ fixed — `stage_local()` copy + run_pipeline reads `source_file` from metadata | `phase0_input.py`, `run_pipeline.py` |
| F4 | ✅ fixed — phase 2 records provider/model/endpoint in `shot_vision_stats.json`, phase 7 propagates, phase 8 renders actual model | `phase2/7/8` |
| F5 | ✅ fixed — all 16 canonical emotions colored; word-boundary matcher (no more "intens**e**"→tense) | `phase8_dashboard.py` |
| F6 | ✅ fixed — missing CSVs warn-and-skip; zero shots, single CLAP window, header-only CSV guarded | `phase8_dashboard.py` (tested) |
| F7 | ✅ fixed — honors `PROCESSED_DIR` | `_renormalize_existing.py` |
| F8 | ✅ fixed — argparse default `small` | `phase4_transcribe.py` |
| F9 | ✅ fixed — shared `_clap_tags.py` replaces 3 copies; eval→Fraction; `--device` honored; dead code removed; sync CSV always written; docstrings/defaults aligned; healthcheck always cleaned up | various |
| data repair | ✅ done — `scripts/_refit_midframes.py` recomputed mid-frames in `shots.json` + propagated to CSVs for both existing videos without re-detecting shots; phases 3/7/8 re-run locally | both videos |

**Outstanding:** phase 2's captions/emotions still describe the old (wrong) frames
until it is re-run against OpenRouter — phase 2's resume logic re-analyzes exactly
the changed mid-frames once `OPENROUTER_API_KEY` is set. `docs/COMPARISON_1FPS_VS_2FPS.md`
is confounded by the old bug (1 fps runs accidentally matched the broken formula).

---

## F1a — Mid-frame at the wrong time (verified on the committed data)

`phase1_shots.py`:

```python
mid_frame_idx = int(mid_sec) + 1  # 1-indexed
```

This says "frame number == seconds + 1", which is only true for **1 fps** extraction.
Extraction default is 2 fps (`run_pipeline.py --fps 2`), where frame N is at
`t = (N-1)/2`. The consequence: the frame given to the vision model sits at
approximately **mid_sec / 2**, and the error grows linearly with video time.

Verified against `data/rtwpk9rb1Dc/shots.json` (frame_fps=2 in metadata):

| shot | mid_sec | file picked | its real time | inside the shot? |
|---|---|---|---|---|
| 0 | 1.42s | frame_00002 | 0.50s | yes (by luck — early shots overlap) |
| 1 | 4.03s | frame_00005 | 2.00s | **no — frame belongs to shot 0** |
| 10 | 28.34s | frame_00029 | 14.00s | no |
| 20 | 59.02s | frame_00060 | 29.50s | no |

From shot 1 onward, **every single mid-frame is in a different shot**, and from
~shot 10 onward the frames come from a different part of the video entirely. So the
captions, emotions, colors, entities, locations, lighting and composition in
`shot_vision.csv` — and every chart, table and thumbnail in both committed
dashboards — were generated from frames that don't correspond to the shots they
label. The round-3 "Methodology" conclusions ("captions consistent",
"emotions match") were computed against this corrupted mapping.

**Fix:** read `frame_fps`/`frames_extracted` from `metadata.json` and compute
`mid_frame_idx = round(mid_sec * fps) + 1`; also bounds-check against the extracted
frame count. The existing per-video outputs need regeneration after the fix.

## F1b — Same bug in phase 3's frame range

`phase3_camera.py:129-133`:

```python
# Shot's start_sec and end_sec correspond to frame indices at 1 fps
start_frame = int(shot["start_sec"]) + 1
end_frame = int(shot["end_sec"]) + 1
```

At 2 fps, files `start_frame..end_frame` cover the time window
`[start_sec/2, end_sec/2]`. Verified: shot 10 (26.86–29.82s) reads files 27–30,
which are at **13.0–14.5s** — completely unrelated footage, spanning >1 real shot,
which inflates apparent motion/boundaries. `camera_motion` in both committed
`shot_camera.csv` files labels the wrong footage.

This also invalidates the 1-fps-vs-2-fps comparison doc: the measured difference
between the two rates is confounded by this bug, not purely a sampling-rate effect
(1 fps runs were accidentally "more correct" because the formula matched them).

## F2 — Download path breaks the video-ID naming contract

`phase0_input.py:40`: YouTube downloads always land at `data/raw/video.mp4`.
README ("Downloads the video to `data/raw/<video_id>.mp4`") and `run_pipeline.py:143`
both expect `data/raw/<video_id>.mp4`. Effects:

- run_pipeline never finds the raw file, prints a warning, and phase 1 only works
  *accidentally* via its own default `data/raw/video.mp4`.
- Processing two videos overwrites the same file — the current
  `data/raw/rtwpk9rb1Dc.mp4` name was presumably fixed up manually.
- `yt-dlp -f best[ext=mp4]/best` with a fixed `.mp4` template writes
  `video.webm` if the best format isn't mp4 — nothing downstream finds it.

**Fix:** after deriving the video id (or receiving `--id`), download to
`data/raw/<video_id>.mp4`. Phase 0 currently doesn't know the video id — it should
get one (env var from run_pipeline, or derive it the same way run_pipeline does).

## F3 — Local-video runs analyze the previous video in phase 1

For a local source, run_pipeline passes `--video "<abs path>"` to phase 0 only.
Nothing copies the file under `data/raw/<video_id>.mp4`, so run_pipeline's phase-1
injection (`if raw_path.exists()`) never fires and phase 1 runs with its default
`data/raw/video.mp4` — i.e., **whatever video was last downloaded**, or an error if
none exists. F2/F3 are one combined fix: make phase 0 stage the source video at
`data/raw/<video_id>.mp4` (copy for local files), and have run_pipeline read the
actual path from `metadata.json['source_file']` instead of predicting it.

## F4 — Vision model provenance is hardcoded

`sync_stats.json` (written by phase 7) has no `vision_model` key, and phase 2 never
writes the model anywhere. `phase8_dashboard.py:377` therefore always falls back:

```python
vision_model_name = sync_stats_full.get("vision_model", "gemini-3-flash-preview")
```

…while the HTML *labels* ("Gemini 3 Flash", caveat text, DQ table) are literal
strings. If the user runs `--model gemma3:27b`, the dashboard still claims Gemini 3
Flash produced the captions. The DQ section even states "model names come from the
upstream files," which is currently false. Fix: phase 2 writes the model name (e.g.,
into `shot_vision_stats.json` or a `stats` column), phase 7 propagates it, phase 8
renders what was actually used.

## F5 — Emotion color map doesn't match the canonical 16

`phase8_dashboard.py` `EMOTION_COLORS` lacks `sensual`, `energetic`, `intense`,
`fearful`, `surprised`, `disgusted` — all canonical outputs of
`_normalize_emotion.py`. They fall through to gray. Notably **"sensual" is the
dominant emotion on the committed Tyla dashboard (50/72 shots)** and renders as
plain gray. Additionally the substring matcher mis-hits: `color_for_emotion("intense")`
returns the *tense* red because `"tense" in "intense"`.

## F6 — Crash paths in phase 8 (standalone / edge runs)

- `:157` `music_features.csv`, `:176` `transcript.csv` are opened unguarded →
  FileNotFoundError if phase 4/6 outputs are absent for any reason (clap is guarded;
  these aren't).
- `:241` `100*top_opencv[1]/len(shots)` → ZeroDivisionError when `sync_per_shot.csv`
  has a header but no rows.
- `:146` `statistics.variance([...])` needs ≥ 2 rows; a single CLAP window raises
  StatisticsError.
- `:144` `clap[0]` → IndexError on a header-only clap CSV.

These don't fire on the committed data, but phase 7 tolerates missing/empty inputs,
so a partially completed pipeline will crash the dashboard at the last step.

## F7 — `_renormalize_existing.py` ignores `PROCESSED_DIR`

Every phase script honors `PROCESSED_DIR`; this helper reads
`data/processed/shot_vision.csv`, which hasn't existed since the per-video
restructure → the helper is currently broken (always prints "not found").

## F8 — Whisper default inconsistency

`phase4_transcribe.py:63` argparse `--model` default is `small.en` (English-only),
but `run_pipeline` passes `small`, the phase docstring and the function signature
say `small`. Round 2 fixed the function default only; standalone phase-4 runs still
behave differently from pipeline runs.

## Low-severity findings

| File | Issue |
|---|---|
| `phase5_audio.py:113` | `--device` arg is accepted but never used (model always loads on CPU) |
| `phase6_music.py:142-143` | `printed_summary` built, never used — dead code |
| `phase7_sync.py:86` | `total_cuts` counts *shots*, not cuts (cosmetic; also `join_data` duplicates the mood-tag list by hand — see below) |
| `phase5/7/8` | the 12 CLAP mood tags are hand-copied in three places (phase5 defines them, phase7 hardcodes a copy with a "must match" comment, phase8 another). One edit breaks phase 7/8 silently. Read them from the `audio_clap.csv` header or import from a shared module |
| `phase0_input.py:80` | `eval()` on ffprobe's `r_frame_rate` — safe for trusted ffprobe output but use `fractions.Fraction` instead |
| `phase7_sync.py:176-183` | writes `sync_per_shot.csv` only when rows exist, then unconditionally prints "[ok] wrote … (0 rows)" — on a re-run a *stale* sync CSV from the previous video can survive and feed phase 8 |
| `phase2_vision.py:209-211, phase3:138` | shots whose mid-frame/frames are missing are silently skipped → row gaps in `shot_vision.csv` / `shot_camera.csv` with only a stderr-ish warn (`[warn]` goes to stdout, easy to miss); phase 7 fills them with empty strings |
| `phase0/3/4/5/6/8` docstrings | still say "data/processed/…" (round 3 flagged this; only partially cleaned) |
| `phase2_vision.py:252` vs `run_pipeline.py:78` | phase-2 standalone default `gemma3:4b` vs pipeline default `$VISION_MODEL or gemini-3-flash-preview` — inconsistent defaults |
| `phase1_shots.py:90` | type hint says `-> list[dict]` but returns a 3-tuple |
| `phase2_vision.py:221-233` | block is indented one level too deep (works, but confusing) |
| `phase2_vision.py:279-285` | `_healthcheck.jpg` is left behind if the health check raises (only unlinked on success) — `.gitignore` covers it |

## What I checked and did NOT find

- No secrets in tracked files (`.env` ignored; test scripts read it at runtime)
- HTML escaping of user-visible strings in the dashboard tables/hovertemplates is
  done (`html_lib.escape` on captions, lyrics, emotions, camera, mood)
- Resume logic in phase 2 (CSV append + rewrite) is correct, including the
  DictWriter missing-key behavior
- Phase 7's empty-input guards (`max(1, …)`) are correct
- `slice_audio` trailing-partial-window handling and the 48 kHz resample are correct
- Krumhansl-Schmuckler key estimation math is correct
- `.gitignore` negations and the tracked `reports/*/dashboard.html` are consistent

## Recommended fix order

1. F1a + F1b (fps-aware frame math) — then **regenerate both videos' outputs**;
   every existing "visual" number is invalid until then.
2. F2 + F3 (one change: stage video at `data/raw/<video_id>.mp4`).
3. F4 (record the actual model name).
4. F5, F7, F8 (small standalone fixes).
5. F6 guards + low-severity cleanup.