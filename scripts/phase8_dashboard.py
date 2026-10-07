"""
Phase 8 — Interactive HTML dashboard.

Reads:  <PROCESSED>/sync_per_shot.csv + sync_stats.json (Phase 7)
        <PROCESSED>/shot_detection_stats.json (Phase 1), music_summary.json (Phase 6),
        audio_clap.csv (Phase 5), transcript.csv (Phase 4), metadata.json (Phase 0),
        run_info.json (provenance from every phase)
Writes: <REPORTS>/dashboard.html

A single self-contained HTML file (thumbnails are embedded as base64 JPEGs, so
it renders anywhere, including when committed to GitHub):
  - Synchronized multi-track timeline (shots / CLAP mood / music energy / beats / lyrics)
  - Distributions: visual emotion, camera motion, audio mood
  - Cut-on-beat table with chance level and permutation p-values
  - Visual↔audio correlations with 95% bootstrap confidence intervals
  - Per-shot table with thumbnails and filters (emotion / camera / audio mood)
  - Findings, data quality and provenance (models, parameters, git commit)

Plotly is loaded from its CDN by default; --offline embeds it (~3.5 MB) so the
file also works without internet.
"""
from __future__ import annotations
import argparse, base64, csv, html, io, json, statistics, sys
from collections import Counter
from pathlib import Path

from common import MOOD_TAGS, PROCESSED, REPO_ROOT, REPORTS, display_path, load_run_info, record_run

# One colour per canonical emotion (see _normalize_emotion.CANONICAL_EMOTIONS)
EMOTION_COLORS = {
    "joyful": "#2ca02c", "playful": "#f5c518", "energetic": "#8c564b", "confident": "#ff7f0e",
    "romantic": "#e377c2", "sensual": "#c2185b", "neutral": "#999999", "contemplative": "#9467bd",
    "sad": "#1f77b4", "melancholic": "#4a6fa5", "anxious": "#d62728", "fearful": "#7f3c8d",
    "angry": "#a50f15", "intense": "#bcbd22", "surprised": "#17becf", "disgusted": "#556b2f",
}
DARK_BACKGROUNDS = {"#1f77b4", "#4a6fa5", "#9467bd", "#d62728", "#7f3c8d", "#a50f15", "#c2185b", "#8c564b", "#556b2f"}


def color_for_emotion(emotion: str) -> str:
    e = (emotion or "").lower().strip()
    if e in EMOTION_COLORS:
        return EMOTION_COLORS[e]
    for k, c in EMOTION_COLORS.items():  # legacy free-text labels
        if k in e:
            return c
    return "#cccccc"


def camera_family(label: str) -> str:
    """pan-left/pan-right → pan, tilt-up/down → tilt (shared VLM/OpenCV vocabulary)."""
    label = (label or "").strip().lower()
    for fam in ("pan", "tilt"):
        if label.startswith(fam):
            return fam
    return label


def esc(v) -> str:
    return html.escape("" if v is None else str(v))


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def thumbnail_data_uri(rel_path: str, width: int = 360) -> str:
    """Downscaled JPEG of a frame as a data: URI ('' if the frame is missing)."""
    if not rel_path:
        return ""
    src = Path(rel_path) if Path(rel_path).is_absolute() else REPO_ROOT / rel_path
    if not src.exists():
        return ""
    from PIL import Image
    with Image.open(src) as im:
        im = im.convert("RGB")
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def html_table(headers: list[str], rows: list[list], cls: str = "data") -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


# ---------------------------------------------------------------------------
# Charts

def timeline_figure(shots, clap, music, beats, transcript):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.3, 0.3, 0.2, 0.2],
        subplot_titles=("Shots (colour = visual emotion)",
                        "Audio mood (CLAP, per-window probability) — 4 most variable tags",
                        "Music energy (RMS per second) + detected beats",
                        "Lyrics / transcript"))
    emotions = [s.get("vision_emotion", "") for s in shots]
    fig.add_trace(go.Bar(
        x=[float(s["duration_sec"]) for s in shots], y=[1] * len(shots),
        base=[float(s["start_sec"]) for s in shots],
        marker_color=[color_for_emotion(e) for e in emotions], marker_line_width=0,
        customdata=[[s["shot_idx"], e or "—", (s.get("vision_caption") or "")[:80]] for s, e in zip(shots, emotions)],
        hovertemplate="<b>Shot %{customdata[0]}</b><br>%{customdata[1]}<br><i>%{customdata[2]}</i><extra></extra>",
        showlegend=False, name="Shots"), row=1, col=1)
    for emotion in sorted(set(emotions)):
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=color_for_emotion(emotion),
                             name=emotion or "(none)", hoverinfo="skip"), row=1, col=1)

    if len(clap) >= 2:
        variances = {t: statistics.variance([float(r[t]) for r in clap]) for t in MOOD_TAGS if t in clap[0]}
        xs = [(float(r["start_sec"]) + float(r["end_sec"])) / 2 for r in clap]
        for tag in sorted(variances, key=variances.get, reverse=True)[:4]:
            fig.add_trace(go.Scatter(x=xs, y=[float(r[tag]) for r in clap], mode="lines", name=tag,
                                     hovertemplate=f"<b>{esc(tag)}</b><br>%{{x:.1f}}s: %{{y:.2f}}<extra></extra>"),
                          row=2, col=1)
    if music:
        fig.add_trace(go.Scatter(x=[float(r["start_sec"]) for r in music], y=[float(r["rms_energy"]) for r in music],
                                 mode="lines", line=dict(color="#ff7f0e"), showlegend=False, name="RMS",
                                 hovertemplate="%{x:.0f}s<br>RMS: %{y:.3f}<extra></extra>"), row=3, col=1)
    if beats:
        fig.add_trace(go.Scatter(x=beats, y=[0] * len(beats), mode="markers", showlegend=False, name="Beats",
                                 marker=dict(symbol="line-ns-open", size=5, color="#888"),
                                 hovertemplate="Beat at %{x:.2f}s<extra></extra>"), row=3, col=1)
    if transcript:
        fig.add_trace(go.Bar(
            x=[float(t["end_sec"]) - float(t["start_sec"]) for t in transcript], y=[1] * len(transcript),
            base=[float(t["start_sec"]) for t in transcript], marker_color="#17becf", marker_line_width=0,
            customdata=[[(t.get("text") or "")[:80]] for t in transcript], showlegend=False, name="Lyrics",
            hovertemplate="<b>Lyric</b> %{base:.1f}s<br>%{customdata[0]}<extra></extra>"), row=4, col=1)

    fig.update_layout(height=1000, barmode="overlay", template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(title_text="Time (sec)", row=4, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_yaxes(title_text="probability", row=2, col=1, range=[0, 1])
    fig.update_yaxes(title_text="RMS", row=3, col=1)
    fig.update_yaxes(visible=False, row=4, col=1)
    return fig


def bar_figure(counts: Counter, title: str, colors=None, xaxis="Number of shots"):
    import plotly.graph_objects as go
    labels = [k for k, _ in counts.most_common()]
    values = [counts[k] for k in labels]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", text=values, textposition="outside",
                           marker_color=[colors(k) for k in labels] if colors else "#4a6fa5",
                           hovertemplate="%{y}: %{x}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title=xaxis, template="plotly_white",
                      height=max(300, 40 + 28 * len(labels)), yaxis=dict(autorange="reversed"))
    return fig


# ---------------------------------------------------------------------------
# Sections

def beat_section(stats: dict) -> str:
    rows = []
    for a in stats.get("beat_alignment", []):
        p = a.get("p_value")
        verdict = ("—" if p is None else "<b>above chance</b>" if p < 0.05 else "not distinguishable from chance")
        rows.append([f"±{a['tolerance_sec'] * 1000:.0f} ms", f"{a['hits']}/{a['n_cuts']} = {a['observed_pct']}%",
                     f"{a['chance_pct']}%", "—" if p is None else f"{p:.4f}", verdict])
    if not rows:
        return "<p><i>No beat-alignment statistics (re-run phase 7).</i></p>"
    return html_table(["Tolerance", "Cuts on a beat", "Expected by chance", "p (permutation)", "Verdict"], rows)


def correlation_section(stats: dict) -> str:
    rows = []
    for c in stats.get("correlations", []):
        ci = f"[{c['ci_low']}, {c['ci_high']}]" if c.get("ci_low") is not None else "—"
        verdict = {"yes": "<b>yes</b>", "opposite": "<b>no — the opposite</b>",
                   "no evidence": "no clear relationship"}.get(c.get("answer"), "not enough data")
        rows.append([esc(c.get("question", "")), f"{esc(c['x'])} ~ {esc(c['y'])}",
                     "—" if c.get("rho") is None else c["rho"], ci, c.get("n", 0), verdict])
    if not rows:
        return "<p><i>No correlation statistics (re-run phase 7).</i></p>"
    return html_table(["Question", "Variables", "Spearman ρ", "95% CI", "Shots", "Supported?"], rows)


def shot_table(shots: list[dict]) -> str:
    rows = []
    for s in shots:
        uri = thumbnail_data_uri(s.get("mid_frame", ""))
        thumb = (f"<img src='{uri}' loading='lazy' class='thumb' alt='shot {esc(s['shot_idx'])}'/>" if uri else "—")
        emo = s.get("vision_emotion", "") or ""
        bg = color_for_emotion(emo)
        fg = "white" if bg in DARK_BACKGROUNDS else "black"
        lyrics = s.get("lyric_text", "") or "—"
        rows.append(
            f"<tr data-emotion='{esc(emo or '—')}' data-camera='{esc(camera_family(s.get('camera_motion', '')) or '—')}' "
            f"data-mood='{esc(s.get('audio_top_mood', '') or '—')}'>"
            f"<td>{thumb}</td><td>{esc(s['shot_idx'])}</td>"
            f"<td>{float(s['start_sec']):.1f}–{float(s['end_sec']):.1f}s</td><td>{float(s['duration_sec']):.1f}s</td>"
            f"<td style='background:{bg};color:{fg}'>{esc(emo or '—')}</td>"
            f"<td>{esc(s.get('camera_motion') or '—')}</td>"
            f"<td>{esc((s.get('vision_caption') or '—')[:120])}</td>"
            f"<td>{esc(s.get('audio_top_mood') or '—')}</td>"
            f"<td>{esc(lyrics[:60] + ('…' if len(lyrics) > 60 else ''))}</td></tr>")

    def options(attr, values):
        opts = "".join(f"<option value='{esc(v)}'>{esc(v)}</option>" for v in sorted(set(values)))
        return f"<label>{attr}: <select data-filter='{attr}'><option value=''>all</option>{opts}</select></label>"
    filters = " ".join([
        options("emotion", [s.get("vision_emotion") or "—" for s in shots]),
        options("camera", [camera_family(s.get("camera_motion", "")) or "—" for s in shots]),
        options("mood", [s.get("audio_top_mood") or "—" for s in shots]),
    ])
    head = "".join(f"<th>{h}</th>" for h in
                   ["Thumb", "#", "Time", "Dur", "Emotion", "Camera", "Caption", "Audio mood", "Lyrics"])
    return (f"<div class='filters'>{filters} <span id='shotCount'></span></div>"
            f"<table id='shotTable' class='data'><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>")


FILTER_JS = """
<div id="lightbox" onclick="this.style.display='none'"><img alt=""></div>
<script>
(function () {
  const table = document.getElementById('shotTable');
  if (!table) return;
  const selects = document.querySelectorAll('select[data-filter]');
  const count = document.getElementById('shotCount');
  function apply() {
    let shown = 0;
    const rows = table.tBodies[0].rows;
    for (const tr of rows) {
      let ok = true;
      selects.forEach(s => { if (s.value && tr.dataset[s.dataset.filter] !== s.value) ok = false; });
      tr.style.display = ok ? '' : 'none';
      if (ok) shown++;
    }
    count.textContent = shown + ' / ' + rows.length + ' shots';
  }
  selects.forEach(s => s.addEventListener('change', apply));
  apply();
  const lb = document.getElementById('lightbox');
  table.addEventListener('click', e => {
    if (e.target.classList.contains('thumb')) { lb.querySelector('img').src = e.target.src; lb.style.display = 'flex'; }
  });
})();
</script>
"""

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 1400px; margin: 0 auto;
       padding: 20px; color: #222; line-height: 1.5; background: #fff; }
h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }
h2 { margin-top: 32px; color: #444; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
.box { padding: 12px 16px; margin: 20px 0; border-radius: 4px; }
.findings { background: #f0f4f8; border-left: 4px solid #2ca02c; }
.caveat { background: #fff8e1; border-left: 4px solid #ff9800; font-size: 14px; }
.quality { background: #f5f5f5; }
.warn { background: #fdecea; border-left: 4px solid #d62728; }
.method { background: #f5f5f5; padding: 10px 14px; border-radius: 4px; font-size: 13px; margin: 8px 0 16px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }
@media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
.panel { background: #fafafa; border: 1px solid #e0e0e0; padding: 12px; border-radius: 4px; overflow-x: auto; }
table.data { border-collapse: collapse; font-size: 12px; width: 100%; }
table.data th, table.data td { border: 1px solid #ccc; padding: 4px 8px; text-align: left; vertical-align: top; }
table.data th { background: #eee; }
#shotTable tr:hover td { background-color: #ffffe0; }
.thumb { width: 110px; height: auto; border: 2px solid #555; cursor: zoom-in; }
.filters { margin: 8px 0; font-size: 13px; } .filters select { margin-right: 12px; }
#lightbox { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.8); align-items: center;
            justify-content: center; cursor: zoom-out; z-index: 10; }
#lightbox img { max-width: 90vw; max-height: 90vh; image-rendering: auto; }
code { background: #eee; padding: 1px 4px; border-radius: 3px; font-size: 13px; }
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="embed plotly.js instead of loading it from a CDN")
    args = parser.parse_args()

    sync_csv = PROCESSED / "sync_per_shot.csv"
    if not sync_csv.exists():
        print("[error] sync_per_shot.csv not found; run phase 7 first")
        return 1
    shots = read_csv(sync_csv)
    stats = read_json(PROCESSED / "sync_stats.json")
    shot_stats = read_json(PROCESSED / "shot_detection_stats.json")
    music_summary = read_json(PROCESSED / "music_summary.json")
    metadata = read_json(PROCESSED / "metadata.json")
    run_info = load_run_info()
    clap = read_csv(PROCESSED / "audio_clap.csv")
    music = read_csv(PROCESSED / "music_features.csv")
    transcript = read_csv(PROCESSED / "transcript.csv")
    beats = music_summary.get("beat_times", [])
    print(f"[info] loaded {len(shots)} shots")

    vision_model = (run_info.get("phase2", {}).get("params", {}).get("model")
                    or stats.get("vision_model") or "not run")
    whisper = run_info.get("phase4", {}).get("params", {})
    fps = metadata.get("frame_fps", "?")
    plotly_js = True if args.offline else "cdn"

    timeline = timeline_figure(shots, clap, music, beats, transcript).to_html(
        include_plotlyjs=plotly_js, full_html=False, div_id="chart_timeline")
    emotion_counts = Counter(s.get("vision_emotion") or "(none)" for s in shots)
    emotion_html = bar_figure(emotion_counts, f"Visual emotion ({vision_model})", color_for_emotion).to_html(
        include_plotlyjs=False, full_html=False, div_id="chart_emotion")
    cam_counts = Counter(s.get("camera_motion") or "(none)" for s in shots)
    cam_html = bar_figure(cam_counts, "Camera motion (feature tracking + RANSAC)").to_html(
        include_plotlyjs=False, full_html=False, div_id="chart_camera")
    if clap:
        mood_avg = Counter({t: round(sum(float(r[t]) for r in clap) / len(clap), 3) for t in MOOD_TAGS})
        mood_html = bar_figure(mood_avg, "Average audio mood probability (CLAP)", xaxis="Mean probability").to_html(
            include_plotlyjs=False, full_html=False, div_id="chart_mood")
    else:
        mood_html = "<p><i>No CLAP data (phase 5 skipped).</i></p>"

    # VLM vs optical-flow camera agreement on the shared vocabulary
    compared = [s for s in shots if s.get("camera_motion") and s.get("vision_camera_from_vlm")
                and s["camera_motion"] != "unknown"]
    agree = sum(camera_family(s["camera_motion"]) == camera_family(s["vision_camera_from_vlm"]) for s in compared)
    agreement = (f"Vision model and optical flow agree on <b>{agree}/{len(compared)}</b> shots "
                 f"({100 * agree / len(compared):.0f}%), comparing pan/tilt without direction."
                 if compared else "No shots have both camera labels.")

    # Findings
    findings = [f"<li><b>{len(shots)} shots</b> detected with {esc(shot_stats.get('detector', '?'))}.</li>"]
    if shot_stats:
        findings.append(f"<li>Shot length: avg <b>{shot_stats['avg_shot_duration_sec']:.1f}s</b>, "
                        f"min {shot_stats['min_shot_duration_sec']:.1f}s, max {shot_stats['max_shot_duration_sec']:.1f}s.</li>")
    if stats.get("beat_alignment"):
        a = stats["beat_alignment"][0]
        sig = (a["p_value"] is not None and a["p_value"] < 0.05)
        findings.append(f"<li><b>{a['observed_pct']}%</b> of cuts land within ±{a['tolerance_sec'] * 1000:.0f} ms of a beat, "
                        f"vs <b>{a['chance_pct']}%</b> expected by chance (p = {a['p_value']}): "
                        + ("cuts <b>are</b> timed to the beat." if sig else "no evidence that cuts follow the beat.")
                        + "</li>")
    for c in stats.get("correlations", []):
        if c.get("answer") in ("yes", "opposite"):
            findings.append(f"<li>{esc(c['question'])} <b>{'Yes' if c['answer'] == 'yes' else 'No, the opposite'}</b>: "
                            f"ρ = {c['rho']} (95% CI {c['ci_low']} to {c['ci_high']}).</li>")
    if music_summary.get("tempo_bpm"):
        findings.append(f"<li>Music: <b>{music_summary['tempo_bpm']} BPM</b>, key <b>{esc(music_summary.get('key', '?'))}</b>.</li>")
    if clap:
        findings.append(f"<li>Dominant audio mood (CLAP): <b>{esc(mood_avg.most_common(1)[0][0])}</b>.</li>")
    if shots and emotion_counts:
        top, n = emotion_counts.most_common(1)[0]
        findings.append(f"<li>Most common visual emotion: <b>{esc(top)}</b> ({n} shots, {100 * n / len(shots):.0f}%).</li>")
    if stats.get("shots_with_lyrics") is not None:
        findings.append(f"<li>{stats['shots_with_lyrics']}/{len(shots)} shots overlap transcribed lyrics.</li>")

    # Data quality + provenance
    n_captions = sum(1 for s in shots if s.get("vision_caption") and not s["vision_caption"].startswith("["))
    dq = [
        ["Frame extraction", f"{metadata.get('frames_extracted', '?')} frames at {fps} fps"],
        ["Shot detection", f"{len(shots)} shots — {shot_stats.get('detector', '?')}"],
        ["Vision", f"{n_captions}/{len(shots)} shots captioned — model {vision_model}"],
        ["Camera motion", f"{sum(1 for s in shots if s.get('camera_motion') not in ('', 'unknown', None))}/{len(shots)} shots classified"],
        ["Transcription", f"{len(transcript)} segments — whisper {whisper.get('model', '?')}"
                          + (" + Demucs vocals" if whisper.get("separate_vocals") else "")],
        ["CLAP audio", f"{len(clap)} windows × {len(MOOD_TAGS)} mood tags (per-group softmax)" if clap else "not run"],
        ["Music", f"{music_summary.get('n_beats', 0)} beats, {music_summary.get('tempo_bpm', '?')} BPM"],
    ]
    prov = [[f"phase {k[5:]}", esc(v.get("git_commit", "?")), esc(v.get("finished_at", "?")),
             esc(", ".join(f"{pk}={pv}" for pk, pv in v.get("params", {}).items() if pk != "source"))]
            for k, v in sorted(run_info.items())]
    stale = sorted({v.get("git_commit", "").replace("-dirty", "") for v in run_info.values()} - {""})
    stale_warning = ("<div class='box warn'>Phases were run from different code versions "
                     f"({', '.join(map(esc, stale))}). Re-run the pipeline for consistent results.</div>"
                     if len(stale) > 1 else "")

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Multimodal Video Analysis</title>
<style>{CSS}</style>
</head>
<body>
<h1>Multimodal Video Analysis</h1>
<p>Source: <code>{esc(Path(str(metadata.get('source_file', '?'))).name)}</code> ·
{esc(round(float(metadata.get('duration_sec', 0)), 1))}s · frames sampled at {esc(fps)} fps.
Generated by <code>scripts/phase8_dashboard.py</code> from the files in <code>{esc(display_path(PROCESSED))}/</code>.</p>
{stale_warning}
<div class="box findings"><h2>Findings (computed from the data)</h2><ul>{''.join(findings)}</ul></div>

<h2>1. Synchronized timeline</h2>
<p class="method"><b>Method:</b> four tracks share the time axis. Shots are coloured by the vision model's emotion label;
audio mood shows the 4 CLAP mood tags that vary most; energy is librosa RMS per second with beat-tracker ticks; lyrics are faster-whisper segments.</p>
{timeline}

<h2>2. Does the editing follow the music?</h2>
<p class="method"><b>Method:</b> a cut is "on beat" if it is within the tolerance of a detected beat. "Expected by chance" is the share of
the timeline within that distance of a beat. The p-value comes from a permutation test that moves each cut by a random fraction of its local
beat period (keeping it in the same part of the song) — a small p means the observed alignment is unlikely to be accidental.</p>
{beat_section(stats)}
<p class="method"><b>Visual ↔ audio correlations:</b> Spearman rank correlation across shots. The 95% confidence interval uses a moving-block
bootstrap (consecutive shots are resampled together because neighbouring shots share the same song section). "Supported" means the interval excludes 0.</p>
{correlation_section(stats)}

<h2>3. Per-modality breakdowns</h2>
<div class="grid"><div class="panel">{emotion_html}</div><div class="panel">{cam_html}</div></div>
<div class="panel"><p>{agreement}</p></div>
<div class="panel">{mood_html}</div>

<h2>4. Per-shot detail</h2>
<p class="method">Thumbnail = the shot's middle frame. Filter rows with the menus; click a thumbnail to enlarge.</p>
{shot_table(shots)}

<div class="box quality"><h2>Data quality</h2>{html_table(["Stream", "Status"], [[esc(a), esc(b)] for a, b in dq])}
<h3>Provenance</h3>{html_table(["Phase", "Git commit", "Finished", "Parameters"], prov) if prov else "<p><i>No run_info.json.</i></p>"}</div>

<div class="box caveat"><h2>Methodology &amp; caveats</h2><ul>
<li><b>Vision</b> ({esc(vision_model)}) sees up to three frames per shot (15/50/85%) and answers from fixed vocabularies. Labels are model judgements and have not been validated against human labels unless <code>docs/VALIDATION.md</code> says otherwise.</li>
<li><b>Shot detection</b> finds hard cuts and fades to black; cross-dissolves are not detected.</li>
<li><b>Camera motion</b> fits a similarity transform to tracked features with RANSAC between frames {esc(fps)} per second apart. Motion slower than the thresholds (default 4% of the frame per second) counts as static; strong subject motion filling the frame can still be mistaken for camera motion.</li>
<li><b>Transcription</b> uses faster-whisper, trained on speech rather than singing; expect gaps on heavily produced vocals.</li>
<li><b>CLAP</b> probabilities are relative within each tag group (mood, section, instrument): a high score means "more similar than the other tags", not that the label is true.</li>
<li><b>Key detection</b> uses Krumhansl-Schmuckler profiles on chroma; relative major/minor keys are easily confused.</li>
</ul></div>
{FILTER_JS}
</body>
</html>
"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "dashboard.html"
    out.write_text(page, encoding="utf-8")
    print(f"[ok] wrote {display_path(out)} ({out.stat().st_size:,} bytes)")
    record_run(8, inputs=[sync_csv, PROCESSED / "sync_stats.json"], outputs=[out],
               params={"offline": args.offline})
    return 0


if __name__ == "__main__":
    sys.exit(main())
