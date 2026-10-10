# UI audit — dashboard (reports/<video_id>/dashboard.html)

Date: 2026-10-10
Scope: generated Plotly HTML dashboards + `scripts/phase8_dashboard.py` (the generator)

## Verdict

The honesty layer (Method subtitles, Data Quality ✓/⚠ table, Methodology caveats,
model provenance) is the strongest part of the UI. The main gaps are **identity,
navigation, and unexploited data** — the dashboard surfaces roughly half of what
the pipeline computed.

## What's already good

- Self-documenting: every chart/section has a Method subtitle naming the
  model/algorithm behind it.
- Data Quality table with real counts per stream; missing/empty streams are
  visible, not hidden.
- Methodology caveats (CLAP similarity ≠ truth, whisper-on-music hallucination,
  zoom-in bias at low fps, ±100ms beat tolerance).
- Timeline design is right: 4 shared-x tracks (shots/emotion, CLAP moods, RMS +
  beats, lyrics) so zooming links across tracks.
- Model provenance is dynamic (never hardcoded; renders "unknown" honestly).
- Thumbnails lazy-load and open full-size on click; grid collapses ≤900px.

## Findings

| # | Issue | Severity | Status |
|---|---|---|---|
| U1 | No identity on the page: `<title>`/`<h1>` are generic ("Multimodal Video Analysis Dashboard"), no video ID, duration, source URL, or generation time — two dashboards open side-by-side are indistinguishable | high | **fixed** (round 5) |
| U2 | Time-indexed analysis with no clickable timestamps — every shot/beat/lyric could deep-link to the source video | high | **fixed** (round 5; yt timestamp links on every shot row) |
| U3 | Dashboard surfaces ~half the computed data: `vision_colors/entities/location/lighting/composition`, `camera_pan/tilt/zoom_score` (continuous scores), `cut_on_beat`, per-window CLAP tags are all in `sync_per_shot.csv` but never rendered | high | **fixed** (round 5: colors/location/lighting/composition/entities/cut-on-beat columns; camera scores as tooltips) |
| U4 | Per-shot table unusable at 72 rows: no sort, no search, no filter, non-sticky header; "Honest findings" bullets are unclickable text | medium | **fixed** (round 5: search box, emotion/camera dropdowns, cut-on-beat filter, sort-by-time, sticky header, findings → table links) |
| U5 | Interactivity is per-chart, not cross-chart; timeline ↔ table not linked | medium | open |
| U6 | Offline breaks most of the page: main chart loads Plotly from cdn.plot.ly, other charts embed no JS at all | medium | open |
| U7 | Latent bug: standalone phase 8 (no REPORTS_DIR) writes `reports/frames/` + `reports/dashboard.html` **shared across videos** — stale-thumbnail hazard between videos | medium | open (mitigated per-video inside run_pipeline) |
| U8 | Legend merges emotion colors and CLAP tag lines — ambiguous which trace a legend entry belongs to | low | open |
| U9 | 11–12px monospace tables dense; no dark mode; no `alt` text on thumbnails | low | open |
| U10 | Timeline height fixed 1000px, tracks not collapsible on short screens | low | open |

## Missing analyses (not bugs — absence)

- Cross-video comparison (the caveat text itself says "compare across videos for
  genre-level patterns" — but the UI offers no aggregate view).
- Correlation views, e.g. cut_on_beat × motion class, visual-emotion ↔ audio-mood
  timeline agreement.
- Camera scores per shot as a continuous chart (only the 8-way class is shown).

## Round-5 status

U1–U4 fixed in `phase8_dashboard.py` (video ID in title/banner, youtu.be deep
links — sourced from `metadata.json['source_url']` written by phase 0, new table
columns incl. cut-on-beat badge and camera-score tooltips, client-side
sort/filter/search). U5–U10 deferred.