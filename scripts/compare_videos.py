"""
Compare every processed video side by side.

Reads:  data/<video_id>/{sync_per_shot.csv, sync_stats.json, shot_detection_stats.json,
        music_summary.json, metadata.json, run_info.json, validation.json}
Writes: reports/comparison.html — self-contained report
        reports/comparison.csv  — one row per video

Answers questions a single-video dashboard cannot: does one video cut faster,
follow the beat more tightly, move the camera more, or skew to different
emotions than another? Videos are only comparable if they were processed the
same way, so the report flags differences in shot detector, frame rate,
vision model or code version between them.

Usage:
  python scripts/compare_videos.py                 # all videos under data/
  python scripts/compare_videos.py rtwpk9rb1Dc Z2ki180nHCI
"""
from __future__ import annotations
import argparse, csv, html, json, statistics, sys
from collections import Counter
from pathlib import Path

from common import DATA_DIR, REPO_ROOT, display_path
from phase8_dashboard import CSS, camera_family, color_for_emotion, html_table


def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_video(d: Path) -> dict | None:
    stats = read_json(d / "sync_stats.json")
    if not stats or not (d / "sync_per_shot.csv").exists():
        return None
    with (d / "sync_per_shot.csv").open(encoding="utf-8") as f:
        shots = list(csv.DictReader(f))
    run_info = read_json(d / "run_info.json")
    meta = read_json(d / "metadata.json")
    det = read_json(d / "shot_detection_stats.json")
    music = read_json(d / "music_summary.json")
    durations = [float(s["duration_sec"]) for s in shots]
    total = float(stats.get("duration_sec") or (shots[-1]["end_sec"] if shots else 0) or 0)
    head = (stats.get("beat_alignment") or [{}])[0]
    cams = Counter(camera_family(s.get("camera_motion", "")) or "(none)" for s in shots)
    emos = Counter(s.get("vision_emotion") or "(none)" for s in shots)
    moods = Counter(s.get("audio_top_mood") or "(none)" for s in shots)
    corr = {f"{c['x']}~{c['y']}": c for c in stats.get("correlations", [])}
    validation = read_json(d / "validation.json")
    return {
        "video_id": d.name,
        "title": Path(str(meta.get("source_file", d.name))).stem,
        "duration_sec": round(total, 1),
        "n_shots": len(shots),
        "median_shot_sec": round(statistics.median(durations), 2) if durations else None,
        "cuts_per_min": round(stats.get("total_cuts", 0) / (total / 60), 1) if total else None,
        "tempo_bpm": music.get("tempo_bpm"),
        "key": music.get("key"),
        "on_beat_pct": head.get("observed_pct"),
        "on_beat_chance_pct": head.get("chance_pct"),
        "on_beat_p": head.get("p_value"),
        "moving_camera_pct": round(100 * sum(v for k, v in cams.items() if k not in ("static", "unknown", "(none)"))
                                   / max(1, len(shots)), 1),
        "top_emotion": emos.most_common(1)[0][0] if emos else None,
        "top_audio_mood": moods.most_common(1)[0][0] if moods else None,
        "rho_loudness_vs_shot_length": corr.get("music_avg_rms~duration_sec", {}).get("rho"),
        "rho_loudness_vs_camera_motion": corr.get("music_avg_rms~camera_motion_magnitude", {}).get("rho"),
        "human_validated": bool(validation.get("results")),
        # settings that must match for numbers to be comparable
        "_settings": {
            "detector": det.get("detector"),
            "frame_fps": meta.get("frame_fps"),
            "vision_model": run_info.get("phase2", {}).get("params", {}).get("model"),
            "code_version": sorted({v.get("git_commit", "?").replace("-dirty", "") for v in run_info.values()}),
        },
        "_durations": durations,
        "_cams": cams,
        "_emos": emos,
    }


def comparability_warnings(videos: list[dict]) -> list[str]:
    warnings = []
    for key in ("detector", "frame_fps", "vision_model"):
        values = {str(v["_settings"][key]) for v in videos}
        if len(values) > 1:
            warnings.append(f"Videos used different <b>{key}</b> settings ({', '.join(sorted(map(html.escape, values)))}).")
    versions = {tuple(v["_settings"]["code_version"]) for v in videos}
    if len(versions) > 1 or any(len(v) > 1 for v in versions):
        warnings.append("Videos were processed with different code versions; re-run the pipeline on all of them "
                        "(<code>run_pipeline.py --force</code>) before drawing conclusions.")
    if not all(v["human_validated"] for v in videos):
        warnings.append("Emotion/camera labels are not human-validated for every video (see scripts/label_shots.py).")
    return warnings


def stacked_shares(videos, key, title, colors=None):
    import plotly.graph_objects as go
    cats = sorted({c for v in videos for c in v[key]})
    fig = go.Figure()
    for c in cats:
        fig.add_trace(go.Bar(
            name=c, x=[v["video_id"] for v in videos],
            y=[100 * v[key][c] / max(1, sum(v[key].values())) for v in videos],
            marker_color=colors(c) if colors else None, hovertemplate=f"{html.escape(c)}: %{{y:.0f}}%<extra></extra>"))
    fig.update_layout(barmode="stack", title=title, yaxis_title="% of shots", template="plotly_white", height=420)
    return fig


def build_report(videos: list[dict]) -> str:
    import plotly.graph_objects as go
    ids = [v["video_id"] for v in videos]

    fig_len = go.Figure([go.Box(y=v["_durations"], name=v["video_id"], boxpoints="all", jitter=0.4) for v in videos])
    fig_len.update_layout(title="Shot length distribution", yaxis_title="seconds", template="plotly_white",
                          height=420, showlegend=False)

    fig_beat = go.Figure([
        go.Bar(name="observed", x=ids, y=[v["on_beat_pct"] for v in videos], marker_color="#4a6fa5",
               text=[f"p={v['on_beat_p']}" for v in videos], textposition="outside"),
        go.Bar(name="chance", x=ids, y=[v["on_beat_chance_pct"] for v in videos], marker_color="#bbbbbb"),
    ])
    fig_beat.update_layout(barmode="group", title="Cuts on a beat (±100 ms): observed vs chance",
                           yaxis_title="% of cuts", template="plotly_white", height=420)

    charts = [fig_len.to_html(include_plotlyjs="cdn", full_html=False),
              fig_beat.to_html(include_plotlyjs=False, full_html=False),
              stacked_shares(videos, "_emos", "Visual emotion mix", color_for_emotion).to_html(
                  include_plotlyjs=False, full_html=False),
              stacked_shares(videos, "_cams", "Camera motion mix").to_html(include_plotlyjs=False, full_html=False)]

    cols = [k for k in videos[0] if not k.startswith("_")]
    table = html_table(cols, [[html.escape(str(v[k])) for k in cols] for v in videos])
    warnings = comparability_warnings(videos)
    warn_html = ("<div class='box warn'><b>Comparability:</b><ul>" + "".join(f"<li>{w}</li>" for w in warnings)
                 + "</ul></div>") if warnings else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Video Comparison</title><style>{CSS}</style></head><body>
<h1>Comparison of {len(videos)} videos</h1>
{warn_html}
<div class="panel" style="overflow-x:auto">{table}</div>
<p class="method"><b>Reading the table:</b> <code>on_beat_p</code> is the permutation p-value for cuts landing on beats
more often than chance; <code>rho_*</code> are Spearman correlations across shots (negative loudness vs shot length = shorter
shots in louder passages). Per-video confidence intervals are in each video's dashboard.</p>
<div class="grid"><div class="panel">{charts[0]}</div><div class="panel">{charts[1]}</div></div>
<div class="grid"><div class="panel">{charts[2]}</div><div class="panel">{charts[3]}</div></div>
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video_ids", nargs="*", help="video IDs under data/ (default: all processed videos)")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out", default=str(REPO_ROOT / "reports" / "comparison.html"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dirs = [data_dir / v for v in args.video_ids] if args.video_ids else sorted(
        d for d in data_dir.iterdir() if d.is_dir() and d.name not in ("raw", "processed"))
    videos = [v for v in (load_video(d) for d in dirs) if v]
    if len(videos) < 2:
        print(f"[error] need at least 2 processed videos (phase 7 done); found {len(videos)} in {data_dir}")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(videos), encoding="utf-8")
    cols = [k for k in videos[0] if not k.startswith("_")]
    with out.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(videos)
    for warning in comparability_warnings(videos):
        print("[warn] " + warning.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", ""))
    print(f"[ok] wrote {display_path(out)} and {display_path(out.with_suffix('.csv'))} ({len(videos)} videos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
