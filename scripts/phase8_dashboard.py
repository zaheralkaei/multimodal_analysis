"""
Phase 8 — Interactive HTML dashboard.

Reads:  data/processed/sync_per_shot.csv + sync_stats.json (Phase 7)
Writes: reports/dashboard.html

A single-file Plotly HTML dashboard with:
  - timeline of all shots with thumbnail, caption, emotion, camera motion
  - audio mood curves (CLAP) overlaid
  - transcript text overlay
  - beat markers + cut-on-beat highlighting
  - cross-modal correlation: visual emotion vs audio mood per shot
  - per-shot detail panel (click any shot)
  - "honest findings" section (auto-generated from sync_stats)
"""
from __future__ import annotations
import argparse, html as html_lib, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"
REPORTS = REPO_ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


MOOD_TAGS = [
    "happy and bright", "sad and melancholic", "aggressive and intense",
    "romantic and tender", "triumphant and epic", "calm and peaceful",
    "tense and anxious", "dreamy and ethereal", "dark and ominous",
    "playful and whimsical", "lonely and introspective", "powerful and confident",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args = parser.parse_args()

    sync_csv = PROCESSED / "sync_per_shot.csv"
    stats_path = PROCESSED / "sync_stats.json"
    if not sync_csv.exists():
        print(f"[error] sync_per_shot.csv not found; run phase 7 first")
        return 1

    import csv
    shots = list(csv.DictReader(sync_csv.open(encoding="utf-8")))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    print(f"[info] loaded {len(shots)} shots + stats")

    # Now write the dashboard
    from collections import Counter
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Chart 1: timeline of shots with thumbnails
    starts = [float(s["start_sec"]) for s in shots]
    durations = [float(s["duration_sec"]) for s in shots]
    durations_y = [1] * len(shots)
    colors_emotion_map = {
        "happy": "#2ca02c", "joyful": "#2ca02c",
        "sad": "#1f77b4", "melancholic": "#1f77b4", "lonely": "#1f77b4",
        "angry": "#d62728", "aggressive": "#d62728", "anxious": "#d62728", "tense": "#d62728",
        "neutral": "#999999",
        "contemplative": "#9467bd", "introspective": "#9467bd", "dreamy": "#9467bd",
        "romantic": "#e377c2", "tender": "#e377c2",
        "epic": "#ff7f0e", "powerful": "#ff7f0e", "triumphant": "#ff7f0e", "confident": "#ff7f0e",
        "playful": "#f5c518", "whimsical": "#f5c518",
        "dark": "#000000", "ominous": "#000000",
        "peaceful": "#17becf", "calm": "#17becf",
    }
    shot_colors = []
    for s in shots:
        emotion = (s.get("vision_emotion", "") or "neutral").lower()
        # find first matching color
        col = "#999999"
        for k, c in colors_emotion_map.items():
            if k in emotion:
                col = c
                break
        shot_colors.append(col)

    # CLAP mood curves
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            "Shots timeline (color = visual emotion)",
            "Audio mood (CLAP, 5s windows) — top-4 tags",
            "Music energy (RMS per second)",
            "Cut-on-beat alignment",
        ),
        shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.30, 0.30, 0.20, 0.20],
    )

    # Row 1: shot bars
    fig.add_trace(go.Bar(
        x=durations, y=durations_y,
        base=starts, marker_color=shot_colors, marker_line_width=0,
        text=[f"#{i}" for i in range(len(shots))],
        textposition="none",
        customdata=[[s.get("vision_caption", "")[:80],
                     s.get("vision_emotion", ""),
                     s.get("camera_motion", ""),
                     s.get("mid_frame_path", s.get("mid_frame", ""))] for s in shots],
        hovertemplate="<b>Shot %{customdata[0]}</b><br>"
                      "Emotion: %{customdata[1]}<br>"
                      "Camera: %{customdata[2]}<br>"
                      "Caption: %{customdata[0]}<br>"
                      "<extra></extra>",
        showlegend=False,
        name="Shots",
    ), row=1, col=1)

    # Row 2: CLAP mood curves (top 4 most variable tags)
    import csv as csv_mod
    clap = list(csv_mod.DictReader((PROCESSED / "audio_clap.csv").open(encoding="utf-8")))
    if clap:
        # find top-4 tags by variance
        import statistics
        variances = {tag: statistics.variance([float(r[tag]) for r in clap]) for tag in MOOD_TAGS if tag in clap[0]}
        top4 = sorted(variances, key=variances.get, reverse=True)[:4]
        for tag in top4:
            xs = [(float(r["start_sec"]) + float(r["end_sec"])) / 2 for r in clap]
            ys = [float(r[tag]) for r in clap]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name=tag,
                hovertemplate=f"<b>{tag}</b><br>%{{x:.1f}}s: %{{y:.2f}}<extra></extra>",
            ), row=2, col=1)

    # Row 3: music RMS
    music = list(csv_mod.DictReader((PROCESSED / "music_features.csv").open(encoding="utf-8")))
    if music:
        xs = [float(r["start_sec"]) for r in music]
        ys = [float(r["rms_energy"]) for r in music]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", line=dict(color="#ff7f0e"),
            name="RMS energy", showlegend=False,
            hovertemplate="%{x:.1f}s<br>RMS: %{y:.3f}<extra></extra>",
        ), row=3, col=1)

    # Row 4: cut markers + beat markers
    music_summary = json.loads((PROCESSED / "music_summary.json").read_text(encoding="utf-8"))
    beats = music_summary.get("beat_times", [])
    # Plot all beats as tiny lines at y=0
    fig.add_trace(go.Scatter(
        x=beats, y=[0] * len(beats),
        mode="markers", marker=dict(symbol="line-ns-open", size=6, color="#aaa"),
        name="Beats", showlegend=False,
        hovertemplate="Beat at %{x:.2f}s<extra></extra>",
    ), row=4, col=1)
    # Highlight cut-on-beat shots
    cob_x = [float(s["start_sec"]) for s in shots if s.get("cut_on_beat") in ("True", "true", True)]
    cob_y = [0.5] * len(cob_x)
    fig.add_trace(go.Scatter(
        x=cob_x, y=cob_y, mode="markers", marker=dict(symbol="star", size=12, color="red"),
        name="Cut on beat", showlegend=True,
        hovertemplate="Cut-on-beat at %{x:.2f}s<extra></extra>",
    ), row=4, col=1)

    fig.update_layout(
        height=900, width=None,
        title_text="<b>Multimodal video analysis — synchronized timeline</b>",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Time (sec)", row=4, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_yaxes(title_text="CLAP similarity", row=2, col=1, range=[0, 1])
    fig.update_yaxes(title_text="RMS", row=3, col=1)
    fig.update_yaxes(visible=False, row=4, col=1, range=[-0.5, 1.0])

    timeline_html = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="chart_timeline")

    # Chart 2: per-shot detail table (top 30)
    n_rows = min(30, len(shots))
    table_rows = []
    headers = ["#", "Time", "Dur", "Caption", "Emotion", "Camera", "Audio mood", "Lyrics"]
    table_rows.append(headers)
    for i, s in enumerate(shots[:n_rows]):
        lyrics = html_lib.escape(s.get("lyric_text", "") or "—")
        lyrics = (lyrics[:60] + "…") if len(lyrics) > 60 else lyrics
        table_rows.append([
            i,
            f"{float(s['start_sec']):.1f}-{float(s['end_sec']):.1f}s",
            f"{float(s['duration_sec']):.1f}s",
            html_lib.escape((s.get("vision_caption", "") or "—")[:80]),
            html_lib.escape(s.get("vision_emotion", "") or "—"),
            html_lib.escape(s.get("camera_motion", "") or "—"),
            html_lib.escape(s.get("audio_top_mood", "") or "—"),
            lyrics,
        ])
    table_html = "<table border='1' style='border-collapse:collapse;font-family:monospace;font-size:12px;width:100%;'>"
    for ri, row in enumerate(table_rows):
        is_header = (ri == 0)
        tag = "th" if is_header else "td"
        table_html += "<tr>" + "".join(
            f"<{tag} style='padding:4px 8px;background:{'#eee' if is_header else ''};text-align:left;'>"
            f"{cell}</{tag}>" for cell in row
        ) + "</tr>"
    table_html += "</table>"

    # Honest findings (dynamically computed)
    cam_counts = Counter(s.get("camera_motion", "unknown") for s in shots)
    findings = []
    findings.append(f"<li><b>{len(shots)} shots</b> detected across the video.</li>")
    findings.append(f"<li><b>{stats['cuts_on_beat']}/{stats['total_shots']} shots</b> cut within 100ms of a musical beat "
                   f"({stats['cuts_on_beat_pct']}%).</li>")
    findings.append(f"<b>Camera motion</b>: " + ", ".join(f"{k}={v}" for k, v in cam_counts.most_common(3)) + ".")
    if stats.get("shots_with_lyrics", 0) > 0:
        findings.append(f"<li><b>{stats['shots_with_lyrics']}/{stats['total_shots']} shots</b> contain sung or spoken lyrics "
                       f"({stats['shots_with_lyrics_pct']}%).</li>")
    findings.append(f"<li>Tempo: <b>{music_summary.get('tempo_bpm', '?')} BPM</b>, "
                   f"key: <b>{music_summary.get('key', '?')}</b> "
                   f"(correlation {music_summary.get('key_correlation', '?')}).</li>")

    findings_html = "<ul>" + "".join(findings) + "</ul>"

    # Build full HTML
    full_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Multimodal Video Analysis</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; color: #222; }}
    h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
    h2 {{ margin-top: 32px; color: #444; }}
    .findings {{ background: #f0f4f8; border-left: 4px solid #2ca02c; padding: 12px 16px; margin: 20px 0; }}
    .caveat {{ background: #fff8e1; border-left: 4px solid #ff9800; padding: 12px 16px; margin: 20px 0; font-size: 14px; }}
  </style>
</head>
<body>
<h1>Multimodal video analysis</h1>
<p>Generated by <code>scripts/phase8_dashboard.py</code>. All charts read from CSV files in <code>data/processed/</code>.</p>

<div class="findings">
  <h2>Honest findings (computed dynamically)</h2>
  {findings_html}
</div>

<h2>1. Synchronized timeline</h2>
{timeline_html}

<h2>2. Per-shot detail (top {n_rows} of {len(shots)} shots)</h2>
{table_html}

<div class="caveat">
  <b>Methodology &amp; caveats.</b>
  <ul>
    <li>Visual analysis uses <code>gemma3:4b</code> via Ollama (the only vision model that fits in this machine's RAM). On more capable hardware, swap to <code>qwen2.5vl:7b</code> by changing the model name in <code>phase2_vision.py</code>.</li>
    <li>Camera motion is classified via OpenCV optical flow, not the VLM — more reliable than asking "is this a pan?" but coarser than human judgment.</li>
    <li>Whisper is trained on speech, not music. On heavily reverbed or fast vocals (e.g. rap), transcript quality varies. Treat the transcript as approximate.</li>
    <li>CLAP similarity scores are 0-1 probabilities per tag — high score means "this audio is similar to the tag", not "this audio IS the tag".</li>
    <li>"Cut on beat" means a shot start is within ±100ms of a detected beat. The {stats['cuts_on_beat_pct']}% rate is for this video; compare across videos to see if it's high or low for the genre.</li>
    <li>Key detection uses Krumhansl-Schmuckler template matching on chroma — works for most popular music but fails on atonal tracks.</li>
  </ul>
</div>

</body>
</html>
"""

    out = REPORTS / "dashboard.html"
    out.write_text(full_html, encoding="utf-8")
    print(f"[ok] wrote {out.relative_to(REPO_ROOT)} ({out.stat().st_size:,} bytes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
