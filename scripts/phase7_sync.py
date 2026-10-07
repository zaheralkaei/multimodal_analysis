"""
Phase 7 — Temporal synchronization + cross-modal analysis.

Reads:
  <PROCESSED>/shots.json (Phase 1)
  <PROCESSED>/shot_vision.csv + shot_vision_meta.json (Phase 2, optional)
  <PROCESSED>/shot_camera.csv (Phase 3, optional)
  <PROCESSED>/transcript.csv (Phase 4, optional)
  <PROCESSED>/audio_clap.csv (Phase 5, optional)
  <PROCESSED>/music_features.csv + music_summary.json (Phase 6)

Writes:
  <PROCESSED>/sync_per_shot.csv — joined table per shot
  <PROCESSED>/sync_stats.json — derived cross-modal statistics

Per shot (audio/music values are averaged over the shot, weighted by overlap):
  - CLAP mood probabilities and the top mood
  - music RMS energy, onset strength, beat count
  - camera motion label and magnitude (|pan| + |tilt| + |zoom| speed)
  - lyrics overlapping the shot
  - cut_on_beat: shot start within ±100 ms of a beat (never for shot 0)

Video-level statistics (see crossmodal.py for the methods):
  - beat_alignment: on-beat cut rate at several tolerances vs chance, with a
    circular-shift permutation p-value
  - correlations: Spearman rho + block-bootstrap 95% CI between visual
    (shot length, camera motion) and audio (energy, onsets) features
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path

from common import BEAT_TOLERANCES_SEC, MOOD_TAGS, PROCESSED, display_path, record_run
from crossmodal import correlate, cut_on_beat_test, nearest_distance, overlap_mean


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _round(v: float, nd: int):
    return "" if math.isnan(v) else round(v, nd)


def join_data(shots: list[dict], vision: list[dict], camera: list[dict],
              transcript: list[dict], clap: list[dict], music: list[dict],
              beats: list[float], mood_tags: list[str] = MOOD_TAGS) -> list[dict]:
    """Join all streams by time into one row per shot."""
    vision_by_shot = {int(r["shot_idx"]): r for r in vision}
    camera_by_shot = {int(r["shot_idx"]): r for r in camera}
    head_tol = BEAT_TOLERANCES_SEC[0]
    beat_dist = nearest_distance([float(s["start_sec"]) for s in shots], beats)

    rows = []
    for i, shot in enumerate(shots):
        s, e = float(shot["start_sec"]), float(shot["end_sec"])
        v = vision_by_shot.get(i, {})
        c = camera_by_shot.get(i, {})

        mood = {t: overlap_mean(clap, t, s, e) for t in mood_tags}
        valid = {t: p for t, p in mood.items() if not math.isnan(p)}
        top_mood = max(valid, key=valid.get) if valid else "n/a"

        pan, tilt, zoom = (_num(c.get(k)) for k in ("pan_score_mean", "tilt_score_mean", "zoom_score_mean"))
        motion = abs(pan) + abs(tilt) + abs(zoom)

        lyrics = [r for r in transcript if float(r["start_sec"]) < e and float(r["end_sec"]) > s
                  and (r.get("text") or "").strip()]

        rows.append({
            "shot_idx": i,
            "start_sec": s,
            "end_sec": e,
            "duration_sec": round(e - s, 3),
            # Visual (Phase 2)
            "vision_caption": v.get("caption", ""),
            "vision_camera_from_vlm": v.get("camera", ""),
            "vision_emotion": v.get("emotion", ""),
            "vision_colors": v.get("colors", ""),
            "vision_entities": v.get("entities", ""),
            "vision_location": v.get("location", ""),
            "vision_lighting": v.get("lighting", ""),
            "vision_composition": v.get("composition", ""),
            "mid_frame": v.get("mid_frame") or shot.get("mid_frame_path", ""),
            # Camera (Phase 3)
            "camera_motion": c.get("camera_motion", ""),
            "camera_pan_score": c.get("pan_score_mean", ""),
            "camera_tilt_score": c.get("tilt_score_mean", ""),
            "camera_zoom_score": c.get("zoom_score_mean", ""),
            "camera_motion_magnitude": _round(motion, 4),
            # Audio (Phase 5)
            "audio_top_mood": top_mood,
            **{f"audio_mood_{t}": _round(mood[t], 4) for t in mood_tags},
            # Music (Phase 6)
            "music_avg_rms": _round(overlap_mean(music, "rms_energy", s, e), 5),
            "music_avg_onset": _round(overlap_mean(music, "onset_strength", s, e), 3),
            "music_n_beats": sum(1 for b in beats if s <= b < e),
            # Cross-modal
            "cut_to_beat_sec": "" if i == 0 or math.isinf(beat_dist[i]) else round(float(beat_dist[i]), 3),
            "cut_on_beat": bool(i > 0 and beat_dist[i] <= head_tol),
            "n_lyric_segments": len(lyrics),
            "lyric_text": " | ".join(r["text"] for r in lyrics)[:300],
        })
    return rows


CORRELATION_PAIRS = [
    # (x column, y column, plain-language question, sign of rho that answers "yes")
    ("music_avg_rms", "duration_sec", "Are shots shorter when the music is louder?", -1),
    ("music_avg_onset", "duration_sec", "Are shots shorter when the music is busier (more onsets)?", -1),
    ("music_avg_rms", "camera_motion_magnitude", "Does the camera move more when the music is louder?", +1),
    ("music_avg_onset", "camera_motion_magnitude", "Does the camera move more when the music is busier?", +1),
]


def answer(res: dict, expected_sign: int) -> str:
    """'yes' / 'opposite' / 'no evidence' / 'insufficient data' from a correlate() result."""
    if res["rho"] is None or res["ci_low"] is None:
        return "insufficient data"
    if not res["significant"]:
        return "no evidence"
    return "yes" if (res["rho"] > 0) == (expected_sign > 0) else "opposite"


def compute_stats(rows: list[dict], transcript: list[dict], beats: list[float],
                  duration: float, n_perm: int = 5000) -> dict:
    cuts = [r["start_sec"] for r in rows[1:]]  # shot 0 starts at t=0: not a cut
    alignment = [cut_on_beat_test(cuts, beats, tol, duration, n_perm=n_perm) for tol in BEAT_TOLERANCES_SEC]
    head = alignment[0]

    correlations = []
    for x, y, question, sign in CORRELATION_PAIRS:
        res = correlate(x, [_num(r[x]) for r in rows], y, [_num(r[y]) for r in rows])
        res["question"] = question
        res["answer"] = answer(res, sign)
        correlations.append(res)

    with_lyrics = sum(1 for r in rows if r["n_lyric_segments"] > 0)
    return {
        "total_shots": len(rows),
        "total_cuts": len(cuts),
        "duration_sec": round(duration, 3),
        # Headline numbers (first tolerance); full table in beat_alignment
        "cuts_on_beat": head["hits"],
        "cuts_on_beat_pct": head["observed_pct"],
        "cuts_on_beat_chance_pct": head["chance_pct"],
        "cuts_on_beat_p_value": head["p_value"],
        "beat_alignment": alignment,
        "correlations": correlations,
        "shots_with_lyrics": with_lyrics,
        "shots_with_lyrics_pct": round(with_lyrics / max(1, len(rows)) * 100, 1),
        "total_lyric_segments": len(transcript),
        "total_lyric_chars": sum(len(r.get("text", "")) for r in transcript),
        "clap_mood_tags": MOOD_TAGS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-perm", type=int, default=5000, help="permutations for the cut-on-beat test")
    args = parser.parse_args()

    shots_path = PROCESSED / "shots.json"
    summary_path = PROCESSED / "music_summary.json"
    for p, phase in ((shots_path, 1), (summary_path, 6)):
        if not p.exists():
            print(f"[error] {p.name} not found; run phase {phase} first")
            return 1
    shots = json.loads(shots_path.read_text(encoding="utf-8"))
    music_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    beats = music_summary.get("beat_times", [])

    inputs = {name: PROCESSED / name for name in
              ("shot_vision.csv", "shot_camera.csv", "transcript.csv", "audio_clap.csv", "music_features.csv")}
    data = {name: load_csv(p) for name, p in inputs.items()}
    print("[info] " + ", ".join(f"{n.split('.')[0]}: {len(v)}" for n, v in data.items())
          + f", shots: {len(shots)}, beats: {len(beats)}")

    rows = join_data(shots, data["shot_vision.csv"], data["shot_camera.csv"], data["transcript.csv"],
                     data["audio_clap.csv"], data["music_features.csv"], beats)
    duration = float(shots[-1]["end_sec"]) if shots else 0.0
    stats = compute_stats(rows, data["transcript.csv"], beats, duration, args.n_perm)

    vision_meta_path = PROCESSED / "shot_vision_meta.json"
    if vision_meta_path.exists():
        stats["vision_model"] = json.loads(vision_meta_path.read_text(encoding="utf-8")).get("model")

    out_csv = PROCESSED / "sync_per_shot.csv"
    if rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"[ok] wrote {display_path(out_csv)} ({len(rows)} rows)")
    out_json = PROCESSED / "sync_stats.json"
    out_json.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {display_path(out_json)}")

    print("\n[stats] cuts on beat (observed vs chance, permutation p):")
    for a in stats["beat_alignment"]:
        print(f"  ±{a['tolerance_sec'] * 1000:.0f} ms: {a['hits']}/{a['n_cuts']} = {a['observed_pct']}% "
              f"vs {a['chance_pct']}% chance, p = {a['p_value']}")
    print("[stats] correlations (Spearman rho, 95% block-bootstrap CI):")
    for c in stats["correlations"]:
        print(f"  {c['x']} ~ {c['y']}: rho={c['rho']} [{c['ci_low']}, {c['ci_high']}] n={c['n']}"
              + ("  *" if c["significant"] else ""))
    print(f"  {stats['shots_with_lyrics']}/{stats['total_shots']} shots contain lyrics")

    record_run(7, inputs=[shots_path, summary_path, vision_meta_path, *inputs.values()],
               outputs=[out_csv, out_json], params={"n_perm": args.n_perm,
                                                    "beat_tolerances_sec": BEAT_TOLERANCES_SEC})
    print("\n[next] Phase 8: python scripts/phase8_dashboard.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
