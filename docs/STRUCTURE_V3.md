# Directory structure (round 3)

## Current (messy)

```
data/
├── raw/                              ← downloaded video files
├── processed/                        ← default = Tyla @ 2 fps (current best)
├── processed_1fps_baseline/          ← Tyla @ 1 fps (for comparison)
├── processed_tyla_baseline/          ← backup of Tyla @ 2 fps
└── processed_4blocks/                ← 4 Blocks TV scene @ 2 fps
reports/
├── dashboard.html                    ← Tyla dashboard
└── frames/                           ← Tyla thumbnails
reports_4blocks/                      ← 4 Blocks dashboard (inconsistent!)
├── dashboard.html
└── frames/
```

Problems:
- "processed_tyla_baseline" vs "processed_4blocks" — inconsistent suffixes
- "reports/" vs "reports_4blocks/" — inconsistent locations
- New users have to guess which is the canonical output

## New (round 3)

```
data/
├── raw/                              ← downloaded video files (gitignored)
└── <video_id>/                       ← one folder per video
    ├── audio.wav
    ├── frames/                       ← extracted JPEGs
    ├── metadata.json
    ├── shots.json
    ├── shot_vision.csv
    ├── shot_camera.csv
    ├── transcript.json
    ├── audio_clap.csv
    ├── music_features.csv
    ├── music_summary.json
    ├── sync_per_shot.csv
    └── sync_stats.json
reports/
└── <video_id>/
    ├── dashboard.html                ← Plotly HTML
    └── frames/                       ← thumbnails for self-contained display
```

Where `<video_id>` is the YouTube ID (e.g. `rtwpk9rb1Dc` for Tyla, `Z2ki180nHCI` for 4 Blocks).

Or for local files / no YouTube ID: a slug like `tyla-said-it-again` or `4blocks-tony-arrested`.

## Migration plan

1. **Migrate Tyla current**: `data/processed/` → `data/rtwpk9rb1Dc/` (using YouTube ID)
2. **Drop** `data/processed_1fps_baseline/` and `data/processed_tyla_baseline/` (1 fps baseline is in git history as commit `28c71ee` if needed; tyla backup is just a copy of current)
3. **Migrate 4 Blocks**: `data/processed_4blocks/` → `data/Z2ki180nHCI/`
4. **Migrate reports**: `reports/` → `reports/rtwpk9rb1Dc/`, `reports_4blocks/` → `reports/Z2ki180nHCI/`

## Default video

The default video is the first YouTube ID we processed. If `data/<id>/` exists, use it. If multiple exist, list them.

If no `data/<id>/` exists, the pipeline downloads the URL and creates the folder using the YouTube ID.

## Phase script behavior

Phase scripts read `PROCESSED_DIR` env var (already supported). Default falls back to:
- If `VIDEO_ID` env var set: `data/<VIDEO_ID>/`
- Else: `data/rtwpk9rb1Dc/` (the canonical Tyla run)
- Else: derive from a global "current video" pointer file

Simpler: require `VIDEO_ID` env var. Fall back to interactive prompt if missing.
Even simpler: pass `--video-id` to every phase. Or have a single wrapper that sets both.

Best: `run_pipeline.py` sets `VIDEO_ID` from the URL, exports `PROCESSED_DIR` and `REPORTS_DIR`, and runs all phases. Phases only need to know `PROCESSED_DIR`.

## Naming convention summary

| What | Format | Example |
|---|---|---|
| Data folder | `data/<video_id>/` | `data/rtwpk9rb1Dc/` |
| Reports folder | `reports/<video_id>/` | `reports/rtwpk9rb1Dc/` |
| Dashboard file | `reports/<video_id>/dashboard.html` | `reports/rtwpk9rb1Dc/dashboard.html` |
| Raw video | `data/raw/<video_id>.mp4` | `data/raw/rtwpk9rb1Dc.mp4` |
| Run command | `python scripts/run_pipeline.py <url> --id <id>` | |

## What needs to change

1. **Default behavior**: drop "default = Tyla" assumption. Every run needs to specify a video ID.
2. **`run_pipeline.py`**: derive ID from URL (use yt-dlp `--get-id`), or accept `--id` arg.
3. **README**: update install + run instructions. Add Model Capabilities section.
4. **Old folders**: rename in place (not delete — git tracks the move).
5. **`.gitignore`**: keep ignoring `data/*/`, `reports/*/`, `data/raw/`.
