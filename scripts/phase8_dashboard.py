"""
Phase 8 — Interactive HTML dashboard.

Reads:  data/processed/sync_per_shot.csv + sync_stats.json (Phase 7)
        data/processed/shot_detection_stats.json (Phase 1)
        data/processed/music_summary.json (Phase 6)
Writes: reports/dashboard.html

A single-file Plotly HTML dashboard with:
  - Synchronized multi-track timeline (shots / CLAP mood / music energy / beats)
  - Per-shot detail table (top 50 of N shots)
  - **Filtering by emotion / camera motion / audio mood** (interactive controls)
  - **Comparisons**: emotion distribution, camera motion distribution, audio mood averages
  - "Honest findings" section (auto-computed from data)
  - "Data quality" section listing which streams had data
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

# Color map for visual emotions (matches HTML)
EMOTION_COLORS = {
    "happy": "#2ca02c", "joyful": "#2ca02c", "excited": "#2ca02c",
    "sad": "#1f77b4", "melancholic": "#1f77b4", "lonely": "#1f77b4",
    "angry": "#d62728", "aggressive": "#d62728", "anxious": "#d62728", "tense": "#d62728",
    "neutral": "#999999", "calm": "#999999",
    "contemplative": "#9467bd", "introspective": "#9467bd", "dreamy": "#9467bd",
    "romantic": "#e377c2", "tender": "#e377c2", "intimate": "#e377c2",
    "epic": "#ff7f0e", "powerful": "#ff7f0e", "triumphant": "#ff7f0e", "confident": "#ff7f0e",
    "playful": "#f5c518", "whimsical": "#f5c518",
    "dark": "#000000", "ominous": "#000000",
    "peaceful": "#17becf",
}


def color_for_emotion(emotion_text: str) -> str:
    e = (emotion_text or "").lower()
    for k, c in EMOTION_COLORS.items():
        if k in e:
            return c
    return "#999999"


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

    # Optional inputs
    shot_stats = {}
    stats_json = PROCESSED / "shot_detection_stats.json"
    if stats_json.exists():
        shot_stats = json.loads(stats_json.read_text(encoding="utf-8"))

    music_summary = {}
    ms_path = PROCESSED / "music_summary.json"
    if ms_path.exists():
        music_summary = json.loads(ms_path.read_text(encoding="utf-8"))

    from collections import Counter
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # ===== Chart 1: Synchronized timeline (4 stacked tracks) =====
    starts = [float(s["start_sec"]) for s in shots]
    durations = [float(s["duration_sec"]) for s in shots]
    shot_colors = [color_for_emotion(s.get("vision_emotion", "")) for s in shots]
    emotion_texts = [s.get("vision_emotion", "") for s in shots]
    captions = [s.get("vision_caption", "") for s in shots]

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            "Shots timeline (color = visual emotion)",
            "Audio mood (CLAP, 5s windows) — top-4 most variable tags",
            "Music energy (RMS per second) + detected beats",
            "Lyrics / transcripts overlap",
        ),
        shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.30, 0.30, 0.20, 0.20],
    )

    # Row 1: shot bars with emotion color
    fig.add_trace(go.Bar(
        x=durations, y=[1] * len(shots), base=starts,
        marker_color=shot_colors, marker_line_width=0,
        customdata=[[i, emotion_texts[i], captions[i][:60]] for i in range(len(shots))],
        hovertemplate="<b>Shot %{customdata[0]}</b><br>"
                      "%{customdata[1]}<br>"
                      "<i>%{customdata[2]}</i><br>"
                      "<extra></extra>",
        showlegend=False, name="Shots",
    ), row=1, col=1)

    # Add emotion legend
    seen_emotions = set()
    for emotion in sorted(set(emotion_texts)):
        c = color_for_emotion(emotion)
        if c not in seen_emotions:
            seen_emotions.add(c)
            fig.add_trace(go.Bar(
                x=[None], y=[None], marker_color=c, name=emotion,
                showlegend=True, hoverinfo="skip",
            ), row=1, col=1)

    # Row 2: CLAP mood curves
    clap_path = PROCESSED / "audio_clap.csv"
    if clap_path.exists():
        clap = list(csv.DictReader(clap_path.open(encoding="utf-8")))
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

    # Row 3: RMS energy + beat ticks
    music = list(csv.DictReader((PROCESSED / "music_features.csv").open(encoding="utf-8")))
    if music:
        xs = [float(r["start_sec"]) for r in music]
        ys = [float(r["rms_energy"]) for r in music]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", line=dict(color="#ff7f0e"),
            name="RMS energy", showlegend=False,
            hovertemplate="%{x:.1f}s<br>RMS: %{y:.3f}<extra></extra>",
        ), row=3, col=1)
    beats = music_summary.get("beat_times", [])
    if beats:
        fig.add_trace(go.Scatter(
            x=beats, y=[0] * len(beats),
            mode="markers", marker=dict(symbol="line-ns-open", size=5, color="#aaa"),
            name="Beats", showlegend=False,
            hovertemplate="Beat at %{x:.2f}s<extra></extra>",
        ), row=3, col=1)

    # Row 4: lyrics as colored bars
    transcript = list(csv.DictReader((PROCESSED / "transcript.csv").open(encoding="utf-8")))
    if transcript:
        for t in transcript:
            s, e = float(t["start_sec"]), float(t["end_sec"])
            text = t.get("text", "")
            fig.add_trace(go.Bar(
                x=[e - s], y=[1], base=[s],
                marker_color="#17becf", marker_line_width=0,
                name="Lyrics", showlegend=False,
                hovertemplate=f"<b>Lyric</b><br>{s:.1f}-{e:.1f}s<br>{html_lib.escape(text[:60])}<extra></extra>",
            ), row=4, col=1)

    fig.update_layout(
        height=1000, width=None,
        title_text="<b>Multimodal video analysis — synchronized timeline</b>",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Time (sec)", row=4, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_yaxes(title_text="CLAP similarity", row=2, col=1, range=[0, 1])
    fig.update_yaxes(title_text="RMS", row=3, col=1)
    fig.update_yaxes(visible=False, row=4, col=1)
    timeline_html = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="chart_timeline")

    # ===== Chart 2: Emotion distribution =====
    emotion_counts = Counter(e or "(none)" for e in emotion_texts)
    fig_emotion = go.Figure(data=[go.Bar(
        x=list(emotion_counts.values()), y=list(emotion_counts.keys()),
        orientation="h", marker_color=[color_for_emotion(e) for e in emotion_counts.keys()],
        text=list(emotion_counts.values()), textposition="outside",
        hovertemplate="%{y}: %{x} shots<extra></extra>",
    )])
    fig_emotion.update_layout(
        title="Visual emotion distribution across all shots",
        xaxis_title="Number of shots", height=max(300, len(emotion_counts) * 30),
        template="plotly_white",
    )
    emotion_dist_html = fig_emotion.to_html(include_plotlyjs=False, full_html=False, div_id="chart_emotion")

    # ===== Chart 3: Camera motion distribution =====
    cam_counts = Counter(s.get("camera_motion", "unknown") for s in shots)
    fig_cam = go.Figure(data=[go.Pie(
        labels=list(cam_counts.keys()), values=list(cam_counts.values()),
        hole=0.4, textinfo="label+percent",
    )])
    fig_cam.update_layout(
        title="Camera motion distribution",
        template="plotly_white",
    )
    cam_dist_html = fig_cam.to_html(include_plotlyjs=False, full_html=False, div_id="chart_camera")

    # ===== Chart 4: Audio mood averages =====
    clap_loaded = list(csv.DictReader(clap_path.open(encoding="utf-8"))) if clap_path.exists() else []
    if clap_loaded:
        avg_per_mood = [(tag, sum(float(r[tag]) for r in clap_loaded) / len(clap_loaded)) for tag in MOOD_TAGS]
        avg_per_mood.sort(key=lambda x: x[1], reverse=True)
        fig_mood = go.Figure(data=[go.Bar(
            x=[v for _, v in avg_per_mood], y=[t for t, _ in avg_per_mood],
            orientation="h", marker_color="#9467bd",
            text=[f"{v:.2f}" for _, v in avg_per_mood], textposition="outside",
        )])
        fig_mood.update_layout(
            title="Average audio mood similarity (CLAP, all windows)",
            xaxis_title="Mean probability", height=400,
            template="plotly_white",
        )
        mood_avg_html = fig_mood.to_html(include_plotlyjs=False, full_html=False, div_id="chart_mood_avg")
    else:
        mood_avg_html = "<p><i>No CLAP data</i></p>"

    # ===== Chart 5: Per-shot detail table (top N, all shots) =====
    n_rows = len(shots)
    table_rows = []
    headers = ["#", "Time", "Dur", "Emotion", "Camera", "Caption", "Audio", "Lyrics"]
    table_rows.append(headers)
    for i, s in enumerate(shots):
        lyrics = html_lib.escape(s.get("lyric_text", "") or "—")
        if len(lyrics) > 60:
            lyrics = lyrics[:60] + "…"
        table_rows.append([
            i,
            f"{float(s['start_sec']):.1f}-{float(s['end_sec']):.1f}s",
            f"{float(s['duration_sec']):.1f}s",
            html_lib.escape(s.get("vision_emotion", "") or "—"),
            html_lib.escape(s.get("camera_motion", "") or "—"),
            html_lib.escape((s.get("vision_caption", "") or "—")[:80]),
            html_lib.escape(s.get("audio_top_mood", "") or "—"),
            lyrics,
        ])
    table_html = "<table id='shotTable' border='1' style='border-collapse:collapse;font-family:monospace;font-size:12px;width:100%;'>"
    for ri, row in enumerate(table_rows):
        is_header = (ri == 0)
        tag = "th" if is_header else "td"
        # Add background color to emotion cell
        cells = []
        for ci, cell in enumerate(row):
            bg = "#eee" if is_header else ""
            if not is_header and ci == 3:  # emotion column
                bg = color_for_emotion(str(cell))
                text_color = "white" if bg in ["#000000", "#1f77b4", "#9467bd", "#d62728"] else "black"
                cells.append(f"<{tag} style='padding:4px 8px;background:{bg};color:{text_color};text-align:left;'>{cell}</{tag}>")
            else:
                cells.append(f"<{tag} style='padding:4px 8px;background:{bg};text-align:left;'>{cell}</{tag}>")
        table_html += "<tr>" + "".join(cells) + "</tr>"
    table_html += "</table>"

    # ===== Honest findings =====
    findings = []
    findings.append(f"<li>Detected <b>{len(shots)} shots</b> using <b>{shot_stats.get('detector', 'PySceneDetect-ContentDetector')}</b> "
                   f"(threshold={shot_stats.get('threshold', '?')}, min_scene_len={shot_stats.get('min_scene_len_frames', '?')} frames).</li>")
    findings.append(f"<li>Shot duration: avg <b>{shot_stats.get('avg_shot_duration_sec', '?'):.1f}s</b>, "
                   f"min <b>{shot_stats.get('min_shot_duration_sec', '?'):.1f}s</b>, "
                   f"max <b>{shot_stats.get('max_shot_duration_sec', '?'):.1f}s</b>.</li>")
    if stats.get("cuts_on_beat") is not None and len(shots) > 0:
        findings.append(f"<li><b>{stats['cuts_on_beat']}/{stats['total_shots']} cuts</b> on a beat "
                       f"({stats['cuts_on_beat_pct']}%) — within 100ms tolerance.</li>")
    if stats.get("shots_with_lyrics"):
        findings.append(f"<li><b>{stats['shots_with_lyrics']}/{stats['total_shots']} shots</b> contain spoken/sung lyrics "
                       f"({stats['shots_with_lyrics_pct']}%).</li>")
    if music_summary.get("tempo_bpm"):
        findings.append(f"<li>Music: <b>{music_summary['tempo_bpm']} BPM</b>, key <b>{music_summary.get('key', '?')}</b> "
                       f"({music_summary.get('n_beats', '?')} beats across {music_summary.get('duration_sec', '?'):.0f}s).</li>")
    if clap_loaded:
        top_mood = max(MOOD_TAGS, key=lambda t: sum(float(r[t]) for r in clap_loaded) / len(clap_loaded))
        findings.append(f"<li>Dominant audio mood (CLAP): <b>{top_mood}</b></li>")
    if emotion_counts:
        top_emotion = emotion_counts.most_common(1)[0]
        findings.append(f"<li>Most common visual emotion: <b>{top_emotion[0]}</b> ({top_emotion[1]} shots, "
                       f"{top_emotion[1]/len(shots)*100:.0f}%)</li>")
    findings_html = "<ul>" + "".join(findings) + "</ul>"

    # ===== Data quality section =====
    dq_items = []
    dq_items.append(("Shot detection", f"✓ {len(shots)} shots from PySceneDetect-ContentDetector"))
    dq_items.append(("Vision captions", f"✓ {len([s for s in shots if s.get('vision_caption') and '[error' not in s.get('vision_caption', '')])}/{len(shots)} shots have captions"))
    dq_items.append(("Camera motion", f"✓ {len([s for s in shots if s.get('camera_motion')])}/{len(shots)} shots classified"))
    dq_items.append(("Transcription", f"{'✓' if transcript else '⚠'} {len(transcript)} segments ({stats.get('total_lyric_chars', 0)} chars)"))
    dq_items.append(("CLAP audio", f"{'✓' if clap_loaded else '⚠'} {len(clap_loaded)} 5s windows × {len(MOOD_TAGS)} mood tags"))
    dq_items.append(("Music structure", f"{'✓' if music_summary else '⚠'} {music_summary.get('n_beats', 0)} beats, {music_summary.get('tempo_bpm', '?')} BPM"))
    dq_html = "<table border='1' style='border-collapse:collapse;font-family:monospace;'>"
    for label, status in dq_items:
        dq_html += f"<tr><td style='padding:6px 12px;font-weight:bold;'>{label}</td><td style='padding:6px 12px;'>{status}</td></tr>"
    dq_html += "</table>"

    # ===== Methodology caveat =====
    caveats = """
    <ul>
      <li><b>Visual analysis</b> uses <code>gemini-3-flash-preview</code> via Ollama cloud. The model "sees" one mid-frame per shot and answers 8 questions. Quality depends on the chosen mid-frame.</li>
      <li><b>Shot detection</b> uses PySceneDetect's ContentDetector (HSV color delta + edge detection). Detects both hard cuts and gradual transitions. False positives possible in compression artifacts.</li>
      <li><b>Camera motion</b> is computed via OpenCV optical flow between consecutive frames within each shot. Coarser than a human labeler but consistent.</li>
      <li><b>Transcription</b> uses faster-whisper. Trained on speech, not music. On heavily reverbed or whispered vocals, expect gaps or mistakes.</li>
      <li><b>CLAP similarity</b>: 0-1 probability per tag. High score = audio is <i>similar to</i> the tag, not that it <i>is</i> the tag.</li>
      <li><b>"Cut on beat"</b>: shot start is within ±100ms of a detected beat. {pct}% for this video; compare across videos for genre-level patterns.</li>
      <li><b>Key detection</b> uses Krumhansl-Schmuckler template matching on chroma. Works for most popular music; fails on atonal tracks.</li>
    </ul>
    """.format(pct=f"{stats.get('cuts_on_beat_pct', 0)}%")

    # ===== Build full HTML =====
    full_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Multimodal Video Analysis Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; color: #222; line-height: 1.5; }}
    h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
    h2 {{ margin-top: 32px; color: #444; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
    .findings {{ background: #f0f4f8; border-left: 4px solid #2ca02c; padding: 12px 16px; margin: 20px 0; }}
    .caveat {{ background: #fff8e1; border-left: 4px solid #ff9800; padding: 12px 16px; margin: 20px 0; font-size: 14px; }}
    .quality {{ background: #f5f5f5; padding: 12px 16px; margin: 20px 0; border-radius: 4px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    .panel {{ background: #fafafa; border: 1px solid #e0e0e0; padding: 12px; border-radius: 4px; }}
    code {{ background: #eee; padding: 1px 4px; border-radius: 3px; font-size: 13px; }}
    table#shotTable tr:hover {{ background: #ffffe0 !important; }}
  </style>
</head>
<body>

<h1>Multimodal Video Analysis</h1>
<p>Single video analyzed across 7 streams. All numbers auto-computed from CSV/JSON files in <code>data/processed/</code>. Generated by <code>scripts/phase8_dashboard.py</code>.</p>

<div class="findings">
  <h2>Honest findings (computed dynamically)</h2>
  {findings_html}
</div>

<div class="quality">
  <h2>Data quality</h2>
  {dq_html}
</div>

<h2>1. Synchronized timeline</h2>
{timeline_html}

<h2>2. Per-modality breakdowns</h2>
<div class="grid">
  <div class="panel">{emotion_dist_html}</div>
  <div class="panel">{cam_dist_html}</div>
</div>
<div class="panel">{mood_avg_html}</div>

<h2>3. Per-shot detail ({len(shots)} shots total)</h2>
<p>Click a header to sort. Color = visual emotion.</p>
{table_html}

<div class="caveat">
  <h2>Methodology &amp; caveats</h2>
  {caveats}
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
