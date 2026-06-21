# 1 fps vs 2 fps — detailed comparison

This document records the actual numerical differences between running the
pipeline at 1 fps and 2 fps on Tyla's *SHE DID IT AGAIN* (3:35, 5163
frames at 24 fps native).

## Frame extraction

| | 1 fps | 2 fps |
|---|---|---|
| Total frames extracted | 215 | 431 |
| Disk for frames (~10 KB each) | 2.2 MB | 4.3 MB |
| Extraction time | ~5 sec | ~5 sec |

## Phase 1 — shot detection

Identical: 72 shots, avg 3.0s, min 1.3s, max 9.5s.

PySceneDetect reads the video directly, not the extracted frames.
Frame rate is irrelevant to this phase.

## Phase 2 — vision captions

Identical API count: 576 calls (72 shots × 8 questions). The mid-frame
chosen for each shot is the same (the geometric center of the shot in
time), regardless of what fps we extracted at.

## Phase 3 — camera motion (the interesting one)

| Camera motion | 1 fps | 2 fps |
|---|---|---|
| static | 7 (9.7%) | **14 (19.4%)** ↑ |
| pan-left | 11 (15.3%) | 9 (12.5%) |
| pan-right | 11 (15.3%) | 10 (13.9%) |
| tilt-up | 11 (15.3%) | 7 (9.7%) |
| tilt-down | 12 (16.7%) | **20 (27.8%)** ↑ |
| zoom-in | **13 (18.1%)** ↑ | 5 (6.9%) |
| zoom-out | 6 (8.3%) | 7 (9.7%) |
| handheld | 1 (1.4%) | 0 (0%) |

**What changed:**
- 1 fps over-classified `zoom-in` (13) and under-classified `static` (7).
- 2 fps rebalanced toward `tilt-down` (20) and `static` (14).
- `handheld` disappeared at 2 fps (was 1 at 1 fps).

**Why?** OpenCV's `calcOpticalFlowFarneback` computes a motion vector per
pixel between consecutive frames. Then we measure the radial divergence
of those vectors — if they point outward from the center, it's a
zoom-out; inward, zoom-in.

At **1 fps** (frames 1s apart), camera shake + subject motion often
creates a slight apparent radial divergence that the algorithm reads as
zoom-in. At **2 fps** (frames 0.5s apart), the same scenes show
mostly vertical motion (tilt) or no motion (static).

**Trust the 2 fps result.** The 0.5s time resolution is enough to see
real motion without confusing camera shake for zoom.

## Phases 4-7 — audio + structure

All identical. These don't use frames; they use the audio track which
is unaffected by frame extraction rate.

- 5 transcript segments ("Deep down you want me", etc.)
- 281 chars of lyrics
- "powerful and confident" audio mood
- 129.2 BPM, B minor, 444 beats
- 47.2% cuts on beat (34/72) — exact same 34 shots, frame rate doesn't
  matter for this metric

## How to reproduce

```bash
# 1 fps baseline (already preserved in data/processed_1fps_baseline/)
python scripts/phase0_input.py "https://www.youtube.com/watch?v=rtwpk9rb1Dc" --fps 1
python scripts/phase1_shots.py
python scripts/phase2_vision.py
python scripts/phase3_camera.py
# (etc.)

# 2 fps (default)
python scripts/phase0_input.py "https://www.youtube.com/watch?v=rtwpk9rb1Dc" --fps 2
python scripts/phase1_shots.py
python scripts/phase2_vision.py
python scripts/phase3_camera.py
# (etc.)
```

Diff the resulting `shot_camera.csv` files to see the camera-motion
shift documented above.
