# Round 10 — does the assembled pipeline hold together?

Date: 2026-10-20 · Step 1 of the integration plan (see the round-9/10
discussion: "specialists measure, the VLM interprets"). Local only, no paid
calls. 215 tests pass; calibration video 14/14 checks.

## What was added

```
P0 input ──► ROUTE (new) ──► P1 … P6 ──► P7 sync ──► QA (new) ──► P8 dashboard
               │                 │          │  ▲            │
               │ content_route   │ every    │  │ lineage    │ qa_report.json
               │ .json           │ step     │  │ check      │ + dashboard section
               ▼                 ▼ records  ▼  │            ▼
        music analyses      lineage.json ───────┘    validity flags + coverage
        on / off                                       per shot
```

| Piece | File | What it does |
|---|---|---|
| Lineage | `scripts/_lineage.py` | Each step records SHA-256 of inputs/outputs, settings and its code. Phase 7 refuses streams whose inputs changed after they ran ("stale") or whose outputs were replaced ("overwritten"). |
| Early routing | `scripts/route_content.py` | After phase 0: CLAP music/speech on ≤24 windows spread over the whole file, silence by loudness (< −45 dBFS). Without music, beats/tempo/key/cut-on-beat are marked invalid. An explicit `--content-type` wins but disagreement with the audio is warned. |
| Validity + coverage | `scripts/phase7_sync.py` | Per-shot `valid_vision`, `valid_camera`, `valid_camera_high`, `valid_audio_mood`, `audio_window_shared`; overlap-weighted audio averages; `coverage` table in `sync_stats.json`. |
| QA report | `scripts/_qa.py` | Cross-module checks: lineage, mock data, vision errors, low coverage, route warnings, unusual tempo / weak key, lyrics over silence, uncertain language, near-identical neighbouring shots (possible false cut), abrupt change inside a shot (possible missed cut), "no people" vs. independent motion, audio resolution. Ranked "review these shots first" list. |
| Calibration | `scripts/calibration.py` | Seeded 36 s synthetic video with known answers; `run` = build + full pipeline (offline) + 14 scored checks. Opt-in pytest: `RUN_CALIBRATION=1`. |

## What the calibration video found (all fixed)

| Bug | Effect | Fix |
|---|---|---|
| **Frame timing.** ffmpeg's `fps` filter keeps the *last* source frame of each 1/fps slot. | Frame *N* (assumed `(N-1)/fps`) actually showed **t + 0.21 s** (2 fps, 24 fps source). The vision model saw frames 0.2 s later than intended; frame-based QA compared across cuts. | `fps=N:round=up` (measured offset 0.0). Phase 2 now reuses an earlier answer only when the **image hash** matches, and depends on `metadata.json` in lineage. The calibration has a cut off the 0.5 s grid plus a `frame_timing` check, and FAILS with the old extraction (verified). |
| **Router sampled only the start** of short files (spacing used n−1 instead of m−1). | A 36 s file was judged from its first 15 s: "100 % music" though a third is speech/silence. | Windows spread over the whole file; the calibration `route` check now requires speech and silence to be heard. |
| **CLAP calls silence "speech"** (p = 0.88 on a near-silent window). | Silence/speech shares wrong; lyrics-over-silence check blind. | Silence decided by loudness (`rms_dbfs` column), CLAP only decides music vs. speech. |

## What the QA found on real footage

Video 1 (Tyla, 146 shots after `min_scene_len` 15): **7 flagged missed cuts**.
I checked all 7 visually and each is a genuinely different picture, ≥ 0.35 s
inside a detected shot. **PySceneDetect at threshold 35 misses cuts**
in this video, which supports adding TransNetV2 as a second detector (plan step 2+).

The first version of the frame heuristics (colour histograms only) produced
12 flags on video 2, of which about 11 were false alarms on dark footage. Now
a flag needs both colour and layout (grey-thumbnail correlation) evidence,
frames too dark or too flat to judge are skipped, and a 0.25 s guard band is
kept at shot edges.

## Open

- QA thresholds are heuristics checked on 3 videos; tune them on the gold set.
- Every shot of video 1 is shorter than the 5 s CLAP window, so per-shot
  audio mood is shared with neighbours (flagged as info). Section-level
  analysis needs song sections (plan step 2: beats/downbeats/sections).
- `min_scene_len` (15 vs 30 frames) still needs to be settled by the gold labels.
