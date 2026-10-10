"""
Phase 8 — Interactive HTML dashboard.

Reads:  data/<video_id>/sync_per_shot.csv + sync_stats.json (Phase 7)
        data/<video_id>/shot_detection_stats.json (Phase 1)
        data/<video_id>/music_summary.json (Phase 6)
Writes: reports/<video_id>/dashboard.html

A single-file Plotly HTML dashboard with:
  - Synchronized multi-track timeline (shots / CLAP mood / music energy / beats)
  - Per-shot detail table
  - Emotion / camera motion / audio mood charts
  - "Honest findings" section (auto-computed from data)
  - "Data quality" section listing which streams had data

Round-4 audit fixes:
  - Vision model name comes from what actually ran (phase 2 → 7 provenance),
    never hardcoded (F4)
  - EMOTION_COLORS covers all 16 canonical emotions; matching is word-boundary
    so "intense" no longer inherits the "tense" color (F5)
  - Guards for missing/empty CSVs: missing music/transcript files, zero rows,
    single CLAP window (F6)
"""
from __future__ import annotations
import argparse, html as html_lib, json, os, re, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
from _paths import disp
PROCESSED = REPO_ROOT / "data" / "processed"
if "PROCESSED_DIR" in os.environ:
    PROCESSED = Path(os.environ["PROCESSED_DIR"])
# Default REPORTS is per-video (UI audit U7): a standalone run with no
# REPORTS_DIR used to share reports/frames across videos (stale thumbnails).
REPORTS = REPO_ROOT / "reports" / PROCESSED.name
if "REPORTS_DIR" in os.environ:
    REPORTS = Path(os.environ["REPORTS_DIR"])

from _clap_tags import MOOD_TAGS
from _content_profile import CONTENT_TYPES, detect_content_type, explain, profile_line

EMOTION_COLORS = {
    # The 16 canonical emotions (from _normalize_emotion.py) — AUDIT_R4 F5:
    # sensual/energetic/intense/fearful/surprised/disgusted were missing and
    # rendered as gray.
    "joyful": "#2ca02c",
    "sad": "#1f77b4",
    "angry": "#d62728",
    "fearful": "#8c564b",
    "surprised": "#ffd700",
    "disgusted": "#bcbd22",
    "neutral": "#999999",
    "contemplative": "#9467bd",
    "sensual": "#e75480",
    "energetic": "#ff7f0e",
    "melancholic": "#6495ed",
    "anxious": "#ff9896",
    "playful": "#f5c518",
    "romantic": "#e377c2",
    "intense": "#c0392b",
    "confident": "#ffbb78",
    # Aliases the model (or the old vocabulary) may emit, mapped to the same
    # color family as their nearest canonical emotion.
    "happy": "#2ca02c",
    "excited": "#ff7f0e",
    "lonely": "#1f77b4",
    "aggressive": "#d62728",
    "tense": "#ff9896",
    "calm": "#999999",
    "introspective": "#9467bd",
    "dreamy": "#9467bd",
    "tender": "#e377c2",
    "intimate": "#e377c2",
    "epic": "#ffbb78",
    "powerful": "#ffbb78",
    "triumphant": "#ffbb78",
    "whimsical": "#f5c518",
    "dark": "#c0392b",
    "ominous": "#c0392b",
    "peaceful": "#17becf",
}


def color_for_emotion(emotion_text: str) -> str:
    """Color for an emotion string, matched on whole words only.

    Word-boundary matching matters: 'intense' must not match the 'tense' key
    (it has its own color). Keys are tried longest-first so multi-word labels
    like 'dark and ominous' resolve before their components.
    """
    e = (emotion_text or "").lower()
    keys = sorted(EMOTION_COLORS, key=len, reverse=True)
    for k in keys:
        if re.search(rf"\b{re.escape(k)}\b", e):
            return EMOTION_COLORS[k]
    return "#999999"



# Client-side per-shot table: search + dropdown filters + cut-on-beat filter +
# click-a-header sort (UI audit U4). Uses the data-* attributes written onto
# each row by main(); self-contained, no external JS.
TABLE_JS = """
<script>
(function () {
  var table = document.getElementById("shotTable");
  if (!table) return;
  var rows = Array.prototype.slice.call(table.querySelectorAll("tr")).slice(1);
  var search = document.getElementById("tblSearch");
  var emo = document.getElementById("tblEmotion");
  var cam = document.getElementById("tblCamera");
  var beat = document.getElementById("tblBeat");
  var count = document.getElementById("tblCount");
  function apply() {
    var q = (search && search.value || "").toLowerCase();
    var visible = 0;
    rows.forEach(function (r) {
      var blocked = (emo && emo.value && r.dataset.emotion !== emo.value) ||
                    (cam && cam.value && r.dataset.camera !== cam.value) ||
                    (beat && beat.checked && r.dataset.beat !== "1");
      var hit = !q || r.textContent.toLowerCase().indexOf(q) !== -1;
      var show = !blocked && hit;
      r.style.display = show ? "" : "none";
      if (show) visible++;
    });
    if (count) count.textContent = "showing " + visible + " / " + rows.length + " shots";
  }
  [search, emo, cam].forEach(function (el) { if (el) { el.addEventListener("input", apply); el.addEventListener("change", apply); } });
  if (beat) beat.addEventListener("change", apply);
  var dir = 1, lastCol = -1;
  Array.prototype.forEach.call(table.querySelectorAll("th"), function (th, ci) {
    th.style.cursor = "pointer";
    th.addEventListener("click", function () {
      var first = rows[0] && rows[0].children[ci];
      var numeric = !!(first && first.dataset && first.dataset.v && !isNaN(parseFloat(first.dataset.v)));
      dir = (ci === lastCol) ? -dir : 1;
      lastCol = ci;
      rows.sort(function (a, b) {
        var av = numeric ? parseFloat(a.children[ci].dataset.v) : 0;
        var bv = numeric ? parseFloat(b.children[ci].dataset.v) : 0;
        if (numeric) return (av < bv ? -1 : av > bv ? 1 : 0) * dir;
        var at = a.children[ci].textContent, bt = b.children[ci].textContent;
        return at < bt ? -dir : at > bt ? dir : 0;
      });
      rows.forEach(function (r) { table.appendChild(r); });
    });
  });
  apply();
})();
</script>
"""


# UI audit U5: click a shot bar on the timeline -> highlight + jump to the
# table row (runs after Plotly has registered the chart div).
TIMELINE_JS = """
<script>
(function () {
  var tl = document.getElementById("chart_timeline");
  var table = document.getElementById("shotTable");
  if (!tl || !table || typeof tl.on !== "function") return;
  tl.on("plotly_click", function (d) {
    try {
      var p = d.points && d.points[0];
      var cd = p && p.customdata;
      if (!cd) return;
      var idx = Array.isArray(cd) ? cd[0] : (cd.shot_idx !== undefined ? cd.shot_idx : null);
      if (idx === null || idx === undefined) return;
      var row = table.querySelector("tr[data-i='" + idx + "']");
      if (!row) return;
      row.scrollIntoView({ behavior: "smooth", block: "center" });
      row.style.background = "#ffe28a";
      setTimeout(function () { row.style.background = ""; }, 1600);
    } catch (e) { /* timeline click highlight is best-effort */ }
  });
})();
</script>
"""


def load_csv_rows(path: Path, label: str) -> list:
    """Read a CSV if it exists; warn (not crash) if missing (AUDIT_R4 F6)."""
    if not path.exists():
        print(f"[warn] {label} not found at {path} — chart will be empty")
        return []
    import csv
    return list(csv.DictReader(path.open(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--content-type", default="auto",
                        choices=list(CONTENT_TYPES),
                        help="Content type for dashboard emphasis (UI audit round 6). "
                             "Default: auto-detect from tempo/lyrics/talk-coverage/shots.")
    args = parser.parse_args()

    sync_csv = PROCESSED / "sync_per_shot.csv"
    stats_path = PROCESSED / "sync_stats.json"
    if not sync_csv.exists():
        print(f"[error] sync_per_shot.csv not found; run phase 7 first")
        return 1

    import csv
    shots = load_csv_rows(sync_csv, "sync_per_shot.csv")
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
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

    # Frame rate from metadata.json (for the data-quality banner)
    metadata = {}
    meta_path = PROCESSED / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    extracted_fps = metadata.get("frame_fps", "?")
    video_fps = metadata.get("video", {}).get("fps", "?")
    n_frames_extracted = metadata.get("frames_extracted", "?")

    # Vision model provenance — what ACTUALLY ran (AUDIT_R4 F4; never hardcode)
    vision_model_name = stats.get("vision_model", "unknown (phase 2 stats missing)")

    # ===== Video identity & source deep-links (UI audit U1/U2) =====
    video_id = PROCESSED.name
    source_url = metadata.get("source_url")
    if not source_url and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        # Fallback: video ID looks like a YouTube ID (pre-round-5 metadata lacks
        # source_url; local files never match the 11-char pattern)
        source_url = f"https://youtu.be/{video_id}"
    total_sec = max((float(s["end_sec"]) for s in shots), default=0.0)

    def yt_link(sec: float) -> str:
        """Anchor to the source video at `sec`, or '' when no source URL is known."""
        if not source_url:
            return ""
        t = max(0, int(sec))
        sep = "&" if "?" in source_url else "?"
        return (f"<a href='{html_lib.escape(source_url)}{sep}t={t}' target='_blank' rel='noopener' "
                f"style='color:#2c7fb8;text-decoration:none;' title='Open in video at {t}s'>&#9654;</a>")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots


    # ===== Chart 1: Synchronized timeline =====
    # Tracks adapt to the data (UI audit round 6): a speech has no CLAP/RMS
    # rows, a film may have no transcript — a fixed 4-row layout used to
    # render empty frames for content types the pipeline once assumed away.
    clap = load_csv_rows(PROCESSED / "audio_clap.csv", "audio_clap.csv")
    music = load_csv_rows(PROCESSED / "music_features.csv", "music_features.csv")
    transcript = load_csv_rows(PROCESSED / "transcript.csv", "transcript.csv")
    beats = music_summary.get("beat_times", [])

    # Content profile (UI audit round 6): type drives emphasis + caveats
    profile = detect_content_type(
        shots=shots, music_summary=music_summary, transcript=transcript,
        video_duration=float(metadata.get("duration_sec", 0) or 0), clap_rows=clap,
        override=args.content_type)
    content_type = profile["type"]

    starts = [float(s["start_sec"]) for s in shots]
    durations = [float(s["duration_sec"]) for s in shots]
    shot_colors = [color_for_emotion(s.get("vision_emotion", "")) for s in shots]
    emotion_texts = [s.get("vision_emotion", "") for s in shots]
    captions = [s.get("vision_caption", "") for s in shots]

    # Which tracks exist?
    top_vars = []
    has_clap_track = len(clap) >= 2
    if has_clap_track:
        import statistics
        tag_var = {tag: statistics.variance([float(r[tag]) for r in clap])
                   for tag in MOOD_TAGS if tag in clap[0]}
        top_vars = sorted(tag_var, key=tag_var.get, reverse=True)[:4]
    has_clap_track = has_clap_track and bool(top_vars)
    has_music_track = bool(music) or bool(beats)
    has_lyrics_track = bool(transcript)

    track_titles = ["Shots timeline (color = visual emotion)"]
    if has_clap_track:
        track_titles.append("Audio mood (CLAP, 5s windows) — top-4 most variable tags")
    if has_music_track:
        track_titles.append("Music energy (RMS per second)" + (" + detected beats" if beats else ""))
    if has_lyrics_track:
        track_titles.append("Lyrics / transcripts")
    n_tracks = len(track_titles)

    heights = {"clap": 0.30, "music": 0.18, "lyrics": 0.18}
    row_h = []
    if has_clap_track: row_h.append(heights["clap"])
    if has_music_track: row_h.append(heights["music"])
    if has_lyrics_track: row_h.append(heights["lyrics"])
    row_h = [0.3] + row_h
    row_sum = sum(row_h)
    row_h = [h / row_sum for h in row_h]

    fig = make_subplots(
        rows=n_tracks, cols=1,
        subplot_titles=tuple(track_titles),
        shared_xaxes=True, vertical_spacing=0.06, row_heights=row_h,
    )

    # Row: shot bars with emotion color (+ click data for timeline->table U5)
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

    # Emotion legend (dedup by color, label = first emotion with that color)
    seen_emotions: dict[str, str] = {}
    for emotion in sorted(set(emotion_texts)):
        c = color_for_emotion(emotion)
        if c not in seen_emotions:
            seen_emotions[c] = emotion
            fig.add_trace(go.Bar(
                x=[None], y=[None], marker_color=c, name=emotion,
                showlegend=True, hoverinfo="skip",
            ), row=1, col=1)

    row_i = 1
    if has_clap_track:
        row_i += 1
        for tag in top_vars:
            xs = [(float(r["start_sec"]) + float(r["end_sec"])) / 2 for r in clap]
            ys = [float(r[tag]) for r in clap]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name=tag,
                hovertemplate=f"<b>{tag}</b><br>%{{x:.1f}}s: %{{y:.2f}}<extra></extra>",
            ), row=row_i, col=1)
        fig.update_yaxes(title_text="CLAP similarity", row=row_i, col=1, range=[0, 1])

    if has_music_track:
        row_i += 1
        if music:
            xs = [float(r["start_sec"]) for r in music]
            ys = [float(r["rms_energy"]) for r in music]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", line=dict(color="#ff7f0e"),
                name="RMS energy", showlegend=False,
                hovertemplate="%{x:.1f}s<br>RMS: %{y:.3f}<extra></extra>",
            ), row=row_i, col=1)
        if beats:
            fig.add_trace(go.Scatter(
                x=beats, y=[0] * len(beats),
                mode="markers", marker=dict(symbol="line-ns-open", size=5, color="#aaa"),
                name="Beats", showlegend=False,
                hovertemplate="Beat at %{x:.2f}s<extra></extra>",
            ), row=row_i, col=1)
        fig.update_yaxes(title_text="RMS", row=row_i, col=1)

    if has_lyrics_track:
        row_i += 1
        for t in transcript:
            s, e = float(t["start_sec"]), float(t["end_sec"])
            text = t.get("text", "")
            fig.add_trace(go.Bar(
                x=[e - s], y=[1], base=[s],
                marker_color="#17becf", marker_line_width=0,
                name="Lyrics", showlegend=False,
                hovertemplate=f"<b>Lyric</b><br>{s:.1f}-{e:.1f}s<br>{html_lib.escape(text[:60])}<extra></extra>",
            ), row=row_i, col=1)
        fig.update_yaxes(visible=False, row=row_i, col=1)

    fig.update_layout(
        height=min(1000, 240 + 190 * n_tracks), width=None,
        title_text=f"<b>{html_lib.escape(video_id)} ({content_type}) — synchronized timeline</b>",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Time (sec)", row=n_tracks, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)
    # UI audit U6: embed Plotly inline so the dashboard is fully self-contained
    # (offline / file:// no longer blanked half the page)
    timeline_html = fig.to_html(include_plotlyjs="inline", full_html=False, div_id="chart_timeline")

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

    # ===== Chart 3: Camera motion distribution + VLM-vs-OpenCV comparison =====
    cam_counts = Counter(s.get("camera_motion", "unknown") for s in shots)
    fig_cam = go.Figure(data=[go.Pie(
        labels=list(cam_counts.keys()), values=list(cam_counts.values()),
        hole=0.4, textinfo="label+percent",
    )])
    fig_cam.update_layout(
        title="Camera motion distribution (OpenCV optical flow)",
        template="plotly_white",
    )
    cam_dist_html = fig_cam.to_html(include_plotlyjs=False, full_html=False, div_id="chart_camera")

    # VLM camera vs OpenCV camera comparison table
    vlm_counts = Counter(s.get("vision_camera_from_vlm", "") or "(none)" for s in shots)
    agreement_count = sum(1 for s in shots
                           if (s.get("camera_motion", "") or "")
                           and (s.get("vision_camera_from_vlm", "") or "")
                           and s.get("camera_motion") == s.get("vision_camera_from_vlm"))
    agreement_pct = round(100 * agreement_count / max(1, len([s for s in shots if s.get("vision_camera_from_vlm")])), 1)

    n_shots = len(shots)
    vlm_cam_html = f"<table border='1' style='border-collapse:collapse;font-family:monospace;font-size:12px;'>"
    vlm_cam_html += "<tr><th style='padding:6px 12px;background:#eee;'>Source</th><th style='padding:6px 12px;background:#eee;'>Method</th><th style='padding:6px 12px;background:#eee;'>Top motion</th><th style='padding:6px 12px;background:#eee;'>Shot count</th></tr>"
    # OpenCV row
    top_opencv = cam_counts.most_common(1)[0] if cam_counts else ("?", 0)
    vlm_cam_html += f"<tr><td style='padding:6px 12px;'><b>OpenCV</b></td><td style='padding:6px 12px;'>optical flow over shot frames</td><td style='padding:6px 12px;'>{html_lib.escape(str(top_opencv[0]))}</td><td style='padding:6px 12px;'>{top_opencv[1]} ({100*top_opencv[1]/max(1, n_shots):.0f}%)</td></tr>"
    # VLM row
    top_vlm = vlm_counts.most_common(1)[0] if vlm_counts else ("?", 0)
    vlm_cam_html += f"<tr><td style='padding:6px 12px;'>{html_lib.escape(vision_model_name)}</td><td style='padding:6px 12px;'>single mid-frame per shot</td><td style='padding:6px 12px;'>{html_lib.escape(str(top_vlm[0]))}</td><td style='padding:6px 12px;'>{top_vlm[1]} ({100*top_vlm[1]/max(1, n_shots):.0f}%)</td></tr>"
    # Agreement row
    vlm_cam_html += f"<tr><td style='padding:6px 12px;'><b>Agreement</b></td><td style='padding:6px 12px;'>shot-level exact-match</td><td style='padding:6px 12px;'>-</td><td style='padding:6px 12px;'>{agreement_count}/{n_shots} = <b>{agreement_pct}%</b></td></tr>"
    vlm_cam_html += "</table>"
    vlm_cam_html += "<p style='font-size:11px;color:#666;margin-top:4px;'>The mid-frame alone cannot show camera motion (the VLM sees 1 instant). OpenCV optical flow sees every extracted frame in the shot and is genuinely better at detecting pan/tilt/zoom. The VLM still helps with semantic understanding (what the subject is doing). Both signals are kept separately in <code>sync_per_shot.csv</code> as <code>camera_motion</code> and <code>vision_camera_from_vlm</code>.</p>"

    # ===== Chart 4: Audio mood averages =====
    clap_tags_present = [t for t in MOOD_TAGS if clap and t in clap[0]]
    if clap_tags_present:
        avg_per_mood = [(tag, sum(float(r[tag]) for r in clap) / len(clap)) for tag in clap_tags_present]
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

    # ===== Chart 5: Per-shot detail table (all shots with thumbnails) =====
    # Columns (UI audit U3): the vision model answers 8 questions — previously
    # only 3 were shown; colors/location/lighting/composition/entities and the
    # cut-on-beat flag are surfaced here, camera scores as cell tooltips (U3).
    report_frames_dir = REPORTS / "frames"
    report_frames_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    n_copied = 0
    n_skipped = 0
    for s in shots:
        mid_rel = s.get("mid_frame", "")
        if mid_rel:
            src = REPO_ROOT / mid_rel
            if src.exists():
                dst = report_frames_dir / src.name
                # Skip ONLY if existing file matches source size (avoids stale thumbnails
                # from a previous run with a different video, where the new file has a
                # different resolution/size). Bug discovered when running on 4 Blocks
                # after Tyla — both had frame_00009.jpg with different sizes.
                if dst.exists() and dst.stat().st_size == src.stat().st_size:
                    n_skipped += 1
                    continue
                shutil.copy2(src, dst)
                n_copied += 1
    print(f"[info] copied {n_copied} mid-frames to {disp(report_frames_dir)} (skipped {n_skipped} that already match)")

    headers = ["Thumb", "#", "Time", "Dur", "Emotion", "Camera", "Caption", "Audio",
               "Colors", "Location", "Lighting", "Composition", "Entities", "Lyrics", "Beat"]
    col_i = {name: i for i, name in enumerate(headers)}
    header_cells = []
    for name in headers:
        header_cells.append(
            f"<th style='padding:6px 8px;background:#eee;text-align:left;"
            f"position:sticky;top:0;'>{name}</th>")
    table_rows = ["".join(header_cells)]
    for i, s in enumerate(shots):
        lyrics = html_lib.escape(s.get("lyric_text", "") or "—")
        if len(lyrics) > 60:
            lyrics = lyrics[:60] + "…"
        mid_rel = s.get("mid_frame", "")
        thumb_path = ""
        if mid_rel:
            # Use the reports/frames/ copy for self-contained HTML
            thumb_path = f"frames/{Path(mid_rel).name}"

        # Camera-score tooltip (U3): the continuous scores behind the class
        score_parts = []
        for axis in ("pan", "tilt", "zoom"):
            v = s.get(f"camera_{axis}_score", "")
            if v not in ("", None):
                try:
                    score_parts.append(f"{axis} {float(v):+.2f}")
                except (TypeError, ValueError):
                    pass
        cam_tooltip = " · ".join(score_parts)

        cut_on_beat = 1 if str(s.get("cut_on_beat", "")).lower() == "true" else 0
        start_s = float(s["start_sec"])
        time_text = f"{start_s:.1f}-{float(s['end_sec']):.1f}s"

        vals = {
            "Thumb": thumb_path,
            "#": str(i),
            "Time": time_text,
            "Dur": f"{float(s['duration_sec']):.1f}s",
            "Emotion": html_lib.escape(s.get("vision_emotion", "") or "—"),
            "Camera": html_lib.escape(s.get("camera_motion", "") or "—"),
            "Caption": html_lib.escape((s.get("vision_caption", "") or "—")[:80]),
            "Audio": html_lib.escape(s.get("audio_top_mood", "") or "—"),
            "Colors": html_lib.escape(s.get("vision_colors", "") or "—"),
            "Location": html_lib.escape(s.get("vision_location", "") or "—"),
            "Lighting": html_lib.escape(s.get("vision_lighting", "") or "—"),
            "Composition": html_lib.escape(s.get("vision_composition", "") or "—"),
            "Entities": html_lib.escape((s.get("vision_entities", "") or "—")[:40]),
            "Lyrics": lyrics,
            "Beat": "♩" if cut_on_beat else "",
        }
        row_attrs = (f" data-i='{i}' data-start='{start_s:.3f}'"
                     f" data-emotion='{html_lib.escape(vals['Emotion']).replace(chr(39), '')}'"
                     f" data-camera='{html_lib.escape(vals['Camera']).replace(chr(39), '')}'"
                     f" data-beat='{cut_on_beat}'")
        cells = [f"<tr{row_attrs}>"]
        for name in headers:
            tag, v = "td", vals[name]
            sort_key = ""
            if name == "#" or name == "Dur":
                sort_key = f" data-v='{v.rstrip('s')}'"
            elif name == "Time":
                sort_key = f" data-v='{start_s:.3f}'"
            if name == "Thumb":
                if v:
                    cells.append(f"<{tag} style='padding:2px;background:#fff;text-align:center;'>"
                                 f"<img src='{v}' loading='lazy' alt='shot {i} mid-frame' "
                                 f"style='width:100px;height:auto;border:2px solid #555;cursor:pointer;' "
                                 f"onclick='window.open(this.src,\"_blank\")'/>"
                                 f"</{tag}>")
                else:
                    cells.append(f"<{tag} style='padding:4px 8px;text-align:left;'>—</{tag}>")
            elif name == "Emotion":
                bg = color_for_emotion(str(v))
                text_color = "white" if bg in ["#1f77b4", "#9467bd", "#d62728", "#c0392b",
                                               "#8c564b", "#6495ed"] else "black"
                cells.append(f"<{tag} style='padding:4px 8px;background:{bg};color:{text_color};text-align:left;'>{v}</{tag}>")
            elif name == "Camera" and cam_tooltip:
                cells.append(f"<{tag} title='{html_lib.escape(cam_tooltip)}' "
                             f"style='padding:4px 8px;text-align:left;'>{v}</{tag}>")
            elif name == "Time":
                cells.append(f"<{tag}{sort_key} style='padding:4px 8px;text-align:left;"
                             f"white-space:nowrap;'>{v} {yt_link(start_s)}</{tag}>")
            else:
                cells.append(f"<{tag}{sort_key} style='padding:4px 8px;text-align:left;'>{v}</{tag}>")
        cells.append("</tr>")
        table_rows.append("".join(cells))

    table_html = ("<table id='shotTable' border='1' "
                  "style='border-collapse:collapse;font-family:monospace;font-size:11px;width:100%;'>")
    table_html += "".join(table_rows)
    table_html += "</table>"
    table_html += "<p style='font-size:11px;color:#666;margin-top:4px;'>Click any thumbnail to view full-size; ▶ opens the source video at that shot's start; click a column header to sort.</p>"

    # Per-shot filter bar + sort (UI audit U4)
    n_emotions = sorted({s.get("vision_emotion", "") for s in shots} - {""})
    n_cameras = sorted({s.get("camera_motion", "") for s in shots} - {""})
    emo_opts = "".join(f"<option value='{html_lib.escape(e)}'>{html_lib.escape(e)}</option>" for e in n_emotions)
    cam_opts = "".join(f"<option value='{html_lib.escape(c)}'>{html_lib.escape(c)}</option>" for c in n_cameras)
    table_controls_html = f"""
<div style="margin:8px 0;display:flex;flex-wrap:wrap;gap:8px;align-items:center;">
  <input id="tblSearch" type="text" placeholder="Search text (caption, lyrics…)" style="padding:6px 8px;width:240px;">
  <select id="tblEmotion" style="padding:6px;"><option value="">Emotion: all</option>{emo_opts}</select>
  <select id="tblCamera" style="padding:6px;"><option value="">Camera: all</option>{cam_opts}</select>
  <label style="font-size:12px;"><input type="checkbox" id="tblBeat" style="margin-right:4px;">only cuts on beat</label>
  <span id="tblCount" style="font-size:11px;color:#666;"></span>
</div>"""
    table_html += table_controls_html + TABLE_JS + TIMELINE_JS

    # ===== Honest findings =====
    findings = []
    safe_model = html_lib.escape(vision_model_name)
    findings.append(f"<li>Detected <b>{n_shots} shots</b> using <b>{shot_stats.get('detector', 'PySceneDetect-ContentDetector')}</b> "
                   f"(threshold={shot_stats.get('threshold', '?')}, min_scene_len={shot_stats.get('min_scene_len_frames', '?')} frames).</li>")
    if shot_stats:
        findings.append(f"<li>Shot duration: avg <b>{shot_stats.get('avg_shot_duration_sec', '?'):.1f}s</b>, "
                       f"min <b>{shot_stats.get('min_shot_duration_sec', '?'):.1f}s</b>, "
                       f"max <b>{shot_stats.get('max_shot_duration_sec', '?'):.1f}s</b>.</li>")
    has_music_sig = bool(music_summary.get("tempo_bpm")) and bool(music_summary.get("n_beats"))
    if stats.get("cuts_on_beat") is not None and n_shots > 0 and has_music_sig:
        findings.append(f"<li><b>{stats['cuts_on_beat']}/{stats['total_shots']} cuts</b> on a beat "
                       f"({stats['cuts_on_beat_pct']}%) — within 100ms tolerance.</li>")
    if content_type in ("speech", "vlog") and transcript:
        talk_chars = sum(len(t.get("text", "")) for t in transcript)
        talk_dur = sum(max(0.0, float(t["end_sec"]) - float(t["start_sec"])) for t in transcript)
        dur_all = max(1.0, float(metadata.get("duration_sec", 0) or 0))
        findings.append(f"<li><b>{len(transcript)} transcript segments</b>, ~{talk_chars / dur_all * 60:.0f} chars/min, "
                       f"talk occupies {min(100, talk_dur / dur_all * 100):.0f}% of the video.</li>")
    how_set = ("explicitly set with --content-type" if profile.get("confidence") is None
               else f"auto-detected (confidence {profile.get('confidence', '?')})")
    findings.append(f"<li>Content type: <b>{content_type}</b> ({how_set}). "
                    f"{html_lib.escape(explain(profile))}.</li>")
    if stats.get("shots_with_lyrics"):
        findings.append(f"<li><b>{stats['shots_with_lyrics']}/{stats['total_shots']} shots</b> contain spoken/sung lyrics "
                       f"({stats['shots_with_lyrics_pct']}%).</li>")
    if music_summary.get("tempo_bpm"):
        findings.append(f"<li>Music: <b>{music_summary['tempo_bpm']} BPM</b>, key <b>{music_summary.get('key', '?')}</b> "
                       f"({music_summary.get('n_beats', '?')} beats across {music_summary.get('duration_sec', '?')}s).</li>")
    if clap_tags_present:
        top_mood = max(clap_tags_present, key=lambda t: sum(float(r[t]) for r in clap) / len(clap))
        findings.append(f"<li>Dominant audio mood (CLAP): <b>{top_mood}</b></li>")
    if emotion_counts:
        top_emotion = emotion_counts.most_common(1)[0]
        findings.append(f"<li>Most common visual emotion: <b>{html_lib.escape(str(top_emotion[0]))}</b> ({top_emotion[1]} shots, "
                        f"{top_emotion[1]/max(1, n_shots)*100:.0f}%) — vision model: {safe_model} "
                        f"(<a href='#shots-table'>see per-shot table</a> — filter by emotion or search text).</li>")
    if source_url:
        findings.append(f"<li>Source: <a href='{html_lib.escape(source_url)}' target='_blank' rel='noopener'>{html_lib.escape(source_url)}</a> "
                        f"— every &#9654; glyph in the per-shot table deep-links to the shot's start time.</li>")
    findings_html = "<ul>" + "".join(findings) + "</ul>"

    # ===== Data quality section =====
    dq_items = []
    dq_items.append(("Frame extraction", f"{n_frames_extracted} frames at {extracted_fps} fps (video native: {round(float(video_fps), 1) if video_fps != '?' else '?'} fps)"))
    dq_items.append(("Shot detection", f"✓ {n_shots} shots from PySceneDetect-ContentDetector (threshold={shot_stats.get('threshold', '?')}, min_scene_len={shot_stats.get('min_scene_len_frames', '?')} frames)"))
    dq_items.append(("Vision captions", f"✓ {len([s for s in shots if s.get('vision_caption') and '[error' not in s.get('vision_caption', '')])}/{n_shots} shots have captions (vision model: {safe_model})"))
    dq_items.append(("Camera motion", f"✓ {len([s for s in shots if s.get('camera_motion')])}/{n_shots} shots classified"))
    dq_items.append(("Transcription", f"{'✓' if transcript else '⚠'} {len(transcript)} segments ({stats.get('total_lyric_chars', 0)} chars)"))
    dq_items.append(("CLAP audio", f"{'✓' if clap else '⚠'} {len(clap)} 5s windows × {len(MOOD_TAGS)} mood tags"))
    dq_items.append(("Music structure", f"{'✓' if music_summary else '⚠'} {music_summary.get('n_beats', 0)} beats, {music_summary.get('tempo_bpm', '?')} BPM"))
    dq_html = "<table border='1' style='border-collapse:collapse;font-family:monospace;'>"
    for label, status in dq_items:
        dq_html += f"<tr><td style='padding:6px 12px;font-weight:bold;'>{label}</td><td style='padding:6px 12px;'>{status}</td></tr>"
    dq_html += "</table>"

    # ===== Methodology caveat =====
    # Caveats follow the content profile (UI audit round 6): music-specific
    # items (CLAP / cut-on-beat / key) only when a music signal exists.
    caveat_lines = [
        f"<li><b>Visual analysis</b> uses <code>{safe_model}</code>. The model \"sees\" one mid-frame per shot and answers 8 questions. Quality depends on the chosen mid-frame.</li>",
        "<li><b>Shot detection</b> uses PySceneDetect's ContentDetector (HSV color delta + edge detection). Detects both hard cuts and gradual transitions. False positives possible in compression artifacts.</li>",
        f"<li><b>Camera motion</b> is computed via OpenCV optical flow between consecutive frames within each shot (at {extracted_fps} fps → {round(1 / extracted_fps, 2) if isinstance(extracted_fps, (int, float)) and extracted_fps else '?'}s time resolution). Coarser than a human labeler but consistent. At lower fps, the optical flow algorithm tends to over-classify zoom-in because frames 1s+ apart often have apparent radial divergence.</li>",
        f"<li><b>Content type: {content_type}</b> ({'explicitly set' if profile.get('confidence') is None else 'auto-detected'}). Rules are ordered and heuristic; a wrong auto-detection here is visible in its reasons, and can be overridden with <code>--content-type</code>.</li>",
    ]
    if transcript:
        caveat_lines.append("<li><b>Transcription</b> uses faster-whisper. Trained on speech, not music. On heavily reverbed or whispered vocals, expect gaps or mistakes.</li>")
    if clap:
        caveat_lines.append("<li><b>CLAP similarity</b>: 0-1 probability per tag. High score = audio is <i>similar to</i> the tag, not that it <i>is</i> the tag.</li>")
    if has_music_sig:
        caveat_lines.append(f"<li><b>\"Cut on beat\"</b>: shot start is within ±100ms of a detected beat. {stats.get('cuts_on_beat_pct', 0)}% for this video; compare across videos for genre-level patterns.</li>")
        caveat_lines.append("<li><b>Key detection</b> uses Krumhansl-Schmuckler template matching on chroma. Works for most popular music; fails on atonal tracks.</li>")
    caveats = "<ul>" + "".join(caveat_lines) + "</ul>"

    # ===== Build full HTML =====
    # Dynamic timeline description — matches the tracks actually drawn
    timeline_tracks = [f"<b>Shots</b> = bars colored by visual emotion (from {safe_model} captions), positioned at each shot's start time; click a shot bar to jump to its table row"]
    if has_clap_track:
        timeline_tracks.append("<b>Audio mood</b> = top-4 CLAP tags by variance, plotted as curves over the 5s windows")
    if has_music_track:
        timeline_tracks.append("<b>Energy + beats</b> = librosa RMS per second + beat tracker ticks")
    if has_lyrics_track:
        timeline_tracks.append("<b>Lyrics</b> = faster-whisper segments")
    timeline_method_line = (f"<b>Method:</b> {n_tracks} track{'s' if n_tracks != 1 else ''} sharing the x-axis "
                            f"(adapted to this {content_type}: tracks with no data are omitted). "
                            + " ".join(timeline_tracks) + ".")
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    full_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{video_id} — Multimodal Video Analysis</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; color: #222; line-height: 1.5; }}
    h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
    h2 {{ margin-top: 32px; color: #444; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
    .findings {{ background: #f0f4f8; border-left: 4px solid #2ca02c; padding: 12px 16px; margin: 20px 0; }}
    .caveat {{ background: #fff8e1; border-left: 4px solid #ff9800; padding: 12px 16px; margin: 20px 0; font-size: 14px; }}
    .section-subtitle {{ background: #f5f5f5; padding: 10px 14px; border-radius: 4px; font-size: 13px; line-height: 1.6; margin: 8px 0 16px; color: #333; }}
    .quality {{ background: #f5f5f5; padding: 12px 16px; margin: 20px 0; border-radius: 4px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    .panel {{ background: #fafafa; border: 1px solid #e0e0e0; padding: 12px; border-radius: 4px; }}
    code {{ background: #eee; padding: 1px 4px; border-radius: 3px; font-size: 13px; }}
    table#shotTable tr:hover {{ background: #ffffe0 !important; }}
  </style>
</head>
<body>

<h1>{html_lib.escape(video_id)} — multimodal video analysis</h1>
<p style="background:#e8f4f8; padding:10px 16px; border-left:4px solid #2c7fb8; margin: 16px 0;">
  <b>Video:</b> {html_lib.escape(video_id)}
  {f'<a href="{html_lib.escape(source_url)}" target="_blank" rel="noopener">&#9654; open source video</a>' if source_url else ''}
  &middot; duration ~{total_sec:.0f}s &middot; {n_shots} shots &middot; generated {generated_at}
  <br><b>Content type:</b> {content_type}
  ({'you set --content-type' if profile.get('confidence') is None else 'auto-detected from tempo / talk coverage / shot grammar'})
  <br><b>Frame extraction:</b> {n_frames_extracted} frames at <b>{extracted_fps} fps</b>
  (video native: {round(float(video_fps), 1) if video_fps != '?' else '?'} fps).
  Camera motion classification depends on this rate — at 2 fps we get 0.5s
  time resolution. See README.md "Sampling-rate decision" for the trade-off
  table and <a href="https://github.com/zaheralkaei/multimodal_analysis/blob/main/docs/COMPARISON_1FPS_VS_2FPS.md">
  docs/COMPARISON_1FPS_VS_2FPS.md</a> for what changes between 1 fps and 2 fps.
</p>
<p>Single video analyzed across 7 streams. All numbers auto-computed from CSV/JSON files in <code>data/&lt;video_id&gt;/</code>. Generated by <code>scripts/phase8_dashboard.py</code>.</p>

<div class="findings">
  <h2>Honest findings (computed dynamically)</h2>
  {findings_html}
</div>

<div class="quality">
  <h2>Data quality</h2>
<p class="section-subtitle"><b>Method:</b> Auto-computed table from <code>data/&lt;video_id&gt;/*.json</code>. Each row shows ✓/⚠ + the actual numbers. Frame count, frame rate, detector params, and model names all come from the upstream files (metadata.json, shot_detection_stats.json, sync_stats.json, shot_vision_stats.json). If a model is wrong, regenerate it and re-run this script.</p>
  {dq_html}
</div>

<h2>1. Synchronized timeline</h2>
<p class="section-subtitle">{timeline_method_line}</p>
{timeline_html}

<h2>2. Per-modality breakdowns</h2>
<p class="section-subtitle"><b>Method:</b> Aggregations across all {n_shots} shots. <b>Emotion</b> = {safe_model} caption emotion word, color-coded by sentiment family. <b>Camera motion</b> = OpenCV optical flow classification (pan/tilt/zoom/static), see Phase 3 docstring for the per-class decision boundaries. <b>Audio mood</b> = mean CLAP probability for each of 12 mood tags across all 5s windows.</p>
<div class="grid">
  <div class="panel">{emotion_dist_html}</div>
  <div class="panel">{cam_dist_html}</div>
</div>
<div class="panel">{vlm_cam_html}</div>
<div class="panel">{mood_avg_html}</div>

<h2 id="shots-table">3. Per-shot detail ({n_shots} shots total)</h2>
<p class="section-subtitle"><b>Method:</b> One row per shot from PySceneDetect ContentDetector. <b>Thumbnail</b> = mid-frame of the shot (the same image sent to the vision model). <b>Emotion</b> = {safe_model} answer to "What is the dominant emotion shown?". <b>Camera</b> = OpenCV optical flow dominant motion class for the shot. <b>Caption</b> = first 80 chars of {safe_model} caption. <b>Audio</b> = highest-probability CLAP mood tag averaged across the shot's duration. <b>Lyrics</b> = faster-whisper text overlapping the shot (truncated to 60 chars).</p>
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
    print(f"[ok] wrote {disp(out)} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())