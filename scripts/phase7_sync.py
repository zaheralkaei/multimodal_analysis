"""
Phase 7 — Temporal synchronization + cross-modal analysis.

Reads:
  data/<video_id>/shots.json (Phase 1)
  data/<video_id>/shot_vision.csv (Phase 2)
  data/<video_id>/shot_vision_stats.json (Phase 2 — model/provider provenance)
  data/<video_id>/shot_camera.csv (Phase 3)
  data/<video_id>/transcript.csv (Phase 4)
  data/<video_id>/audio_clap.csv (Phase 5)
  data/<video_id>/music_features.csv (Phase 6)
  data/<video_id>/music_summary.json (Phase 6)

Writes:
  data/<video_id>/sync_per_shot.csv — joined table per shot
  data/<video_id>/sync_stats.json — derived cross-modal signals + vision model

Round 8 (analysis validity):
  - Shot 0 starts at t=0 — that is not an edit; it is excluded from cut
    statistics (cut_on_beat is blank for it).
  - cut-on-beat is reported WITH its chance level: circular-shift null
    model + Rayleigh test on beat phase (scripts/_stats.py). Before, the bare
    percentage (~45 %) was indistinguishable from random cutting.
  - Vision/camera rows are joined only when their start_sec matches the
    shot (stale CSVs from an earlier phase-1 run no longer attach to the
    wrong shot). Mismatches are counted in sync_stats.json.
  - Transcript segments flagged `suspect` by phase 4 are excluded from
    lyric counts/text.

Cross-modal signals computed:
  - cut_on_beat: did this shot's start coincide with a beat (within 100ms)?
  - avg_audio_mood_for_shot: mean CLAP mood tag probabilities across shot duration
  - avg_music_energy_for_shot: mean RMS energy across shot duration
  - lyric_count_in_shot: how many transcript segments overlap the shot
  - combined_emotion_score: rough alignment of visual emotion with audio mood
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
from _paths import disp
PROCESSED = REPO_ROOT / "data" / "processed"
if "PROCESSED_DIR" in os.environ:
    PROCESSED = Path(os.environ["PROCESSED_DIR"])


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))



def within_tolerance(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


BEAT_TOLERANCE_SEC = 0.1


def _aligned(row: dict, start: float) -> bool:
    """A per-shot row belongs to this shot if its start time matches (rows
    without a start_sec column are accepted for backward compatibility)."""
    v = row.get("start_sec", "")
    if v in ("", None):
        return True
    try:
        return abs(float(v) - start) < 0.02
    except ValueError:
        return False


def _is_suspect(seg: dict) -> bool:
    return str(seg.get("suspect", "")).strip().lower() in ("true", "1", "yes")


def join_data(shots: list[dict], vision: list[dict], camera: list[dict],
              transcript: list[dict], clap: list[dict],
              music: list[dict], beats: list[float],
              mood_tags: list[str]) -> tuple[list[dict], dict]:
    """Join all streams by time. Returns (per_shot_rows, stats_dict)."""
    rows = []
    cut_on_beat_count = 0
    total_cuts = 0
    misaligned = {"vision": 0, "camera": 0}
    usable_transcript = [r for r in transcript if not _is_suspect(r)]

    # Index by shot_idx for fast lookup
    vision_by_shot = {int(r["shot_idx"]): r for r in vision}
    camera_by_shot = {int(r["shot_idx"]): r for r in camera}

    for i, shot in enumerate(shots):
        s = float(shot["start_sec"])
        e = float(shot["end_sec"])
        v = vision_by_shot.get(i, {})
        c = camera_by_shot.get(i, {})
        if v and not _aligned(v, s):
            misaligned["vision"] += 1
            v = {}
        if c and not _aligned(c, s):
            misaligned["camera"] += 1
            c = {}

        # CLAP: mean probability across windows in [s, e]
        clap_in_shot = [r for r in clap if float(r["start_sec"]) < e and float(r["end_sec"]) > s]
        avg_clap = {tag: 0.0 for tag in mood_tags}
        if clap_in_shot:
            for tag in mood_tags:
                vals = [float(r[tag]) for r in clap_in_shot if tag in r]
                if vals:
                    avg_clap[tag] = round(sum(vals) / len(vals), 4)
        # Top mood
        top_mood = max(avg_clap, key=avg_clap.get) if any(avg_clap.values()) else "n/a"

        # Music features: mean RMS in [s, e]
        music_in_shot = [r for r in music if float(r["start_sec"]) < e and float(r["end_sec"]) > s]
        rms_vals = [float(r["rms_energy"]) for r in music_in_shot if r.get("rms_energy")]
        avg_rms = round(sum(rms_vals) / len(rms_vals), 5) if rms_vals else 0.0
        # beats whose time falls inside the shot (audit R9: summing the
        # per-second counts of every overlapping second credited a 0.7 s shot
        # with up to 2 s of beats)
        beats_in_shot = sum(1 for b in beats if s <= float(b) < e)

        # Transcript: count segments overlapping
        lyrics_in_shot = [r for r in usable_transcript if float(r["start_sec"]) < e and float(r["end_sec"]) > s and r.get("text", "").strip()]
        lyric_text = " | ".join(r["text"] for r in lyrics_in_shot)[:300]

        # Cut on beat? (shot 0 starts the video — not a cut)
        if i == 0:
            is_cut_on_beat = ""
        else:
            total_cuts += 1
            is_cut_on_beat = any(within_tolerance(s, b, BEAT_TOLERANCE_SEC) for b in beats)
            if is_cut_on_beat:
                cut_on_beat_count += 1

        rows.append({
            "shot_idx": i,
            "start_sec": s,
            "end_sec": e,
            "duration_sec": round(e - s, 3),
            # Visual (from Phase 2)
            "vision_caption": v.get("caption", ""),
            "vision_camera_from_vlm": v.get("camera", ""),
            "vision_emotion": v.get("emotion", ""),
            "vision_colors": v.get("colors", ""),
            "vision_entities": v.get("entities", ""),
            "vision_location": v.get("location", ""),
            "vision_lighting": v.get("lighting", ""),
            "vision_composition": v.get("composition", ""),
            "mid_frame": v.get("mid_frame", ""),  # propagate for dashboard thumbnails
            # Camera (from Phase 3)
            "camera_motion": c.get("camera_motion", ""),
            "camera_pan_score": c.get("pan_rate", c.get("pan_score_mean", "")),
            "camera_tilt_score": c.get("tilt_rate", c.get("tilt_score_mean", "")),
            "camera_zoom_score": c.get("zoom_rate", c.get("zoom_score_mean", "")),
            "camera_pan_rate": c.get("pan_rate", ""),
            "camera_tilt_rate": c.get("tilt_rate", ""),
            "camera_zoom_rate": c.get("zoom_rate", ""),
            "camera_shake": c.get("shake", ""),
            "camera_confidence": c.get("camera_confidence", ""),
            "camera_components": c.get("camera_components", ""),
            "camera_subject_motion_share": c.get("subject_motion_share", ""),
            # Audio (from Phase 5)
            "audio_top_mood": top_mood,
            **{f"audio_mood_{k}": avg_clap[k] for k in mood_tags},
            # Music (from Phase 6)
            "music_avg_rms": avg_rms,
            "music_n_beats": beats_in_shot,
            # Cross-modal
            "cut_on_beat": is_cut_on_beat,
            "n_lyric_segments": len(lyrics_in_shot),
            "lyric_text": lyric_text,
        })

    # Stats
    from _stats import beat_alignment
    cut_times = [float(sh["start_sec"]) for sh in shots[1:]]
    duration = float(shots[-1]["end_sec"]) if shots else 0.0
    alignment = beat_alignment(cut_times, beats, duration, tol=BEAT_TOLERANCE_SEC)
    stats = {
        "total_shots": len(shots),
        "total_cuts": total_cuts,
        "cuts_on_beat": cut_on_beat_count,
        "cuts_on_beat_pct": round(cut_on_beat_count / max(1, total_cuts) * 100, 1),
        "cut_beat_alignment": alignment,
        "misaligned_rows_dropped": misaligned,
        "suspect_transcript_segments_excluded": len(transcript) - len(usable_transcript),
        "shots_with_lyrics": sum(1 for r in rows if r["n_lyric_segments"] > 0),
        "shots_with_lyrics_pct": round(
            sum(1 for r in rows if r["n_lyric_segments"] > 0) / max(1, len(rows)) * 100, 1
        ),
        "total_lyric_segments": len(usable_transcript),
        "total_lyric_chars": sum(len(r.get("text", "")) for r in usable_transcript),
        "avg_music_rms": round(sum(float(r.get("music_avg_rms", 0)) for r in rows) / max(1, len(rows)), 5),
        "clap_mood_tags": mood_tags,
    }
    return rows, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args = parser.parse_args()

    # Load everything
    shots_path = PROCESSED / "shots.json"
    if not shots_path.exists():
        print(f"[error] shots.json not found")
        return 1
    shots = json.loads(shots_path.read_text(encoding="utf-8"))

    vision = load_csv(PROCESSED / "shot_vision.csv")
    camera = load_csv(PROCESSED / "shot_camera.csv")
    transcript = load_csv(PROCESSED / "transcript.csv")
    clap = load_csv(PROCESSED / "audio_clap.csv")
    music = load_csv(PROCESSED / "music_features.csv")

    music_summary = json.loads((PROCESSED / "music_summary.json").read_text(encoding="utf-8"))
    beats = music_summary.get("beat_times", [])

    print(f"[info] shots: {len(shots)}, vision: {len(vision)}, camera: {len(camera)}, "
          f"transcript: {len(transcript)}, clap: {len(clap)}, music: {len(music)}, "
          f"beats: {len(beats)}")

    # Mood tag list from the shared vocabulary module (round-4 audit fix:
    # was hand-copied here and could drift from phase 5's list)
    from _clap_tags import MOOD_TAGS

    rows, stats = join_data(shots, vision, camera, transcript, clap, music, beats, MOOD_TAGS)

    # Vision model provenance (AUDIT_R4 F4) — phase 8 renders what actually ran
    vision_stats_path = PROCESSED / "shot_vision_stats.json"
    if vision_stats_path.exists():
        vstats = json.loads(vision_stats_path.read_text(encoding="utf-8"))
        stats["vision_model"] = f"{vstats.get('model', '?')}"
        stats["vision_provider"] = vstats.get("provider", "?")
        stats["vision_endpoint"] = vstats.get("endpoint", "?")

    # Write per-shot CSV — ALWAYS, so a stale sync CSV from a previous video
    # can't survive an empty run (round-4 audit low-severity fix)
    out_csv = PROCESSED / "sync_per_shot.csv"
    cols = list(rows[0].keys()) if rows else ["shot_idx"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[ok] wrote {disp(out_csv)} ({len(rows)} rows)")

    # Write stats
    out_json = PROCESSED / "sync_stats.json"
    out_json.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {disp(out_json)}")

    print(f"\n[stats]")
    al = stats["cut_beat_alignment"]
    if al.get("chance_rate") is None:
        print(f"  {stats['cuts_on_beat']}/{stats['total_cuts']} cuts on a beat "
              f"({stats['cuts_on_beat_pct']}%); no chance test ({al.get('note', 'no beats')})")
    else:
        print(f"  {stats['cuts_on_beat']}/{stats['total_cuts']} cuts on a beat ({stats['cuts_on_beat_pct']}%); "
              f"chance {al['chance_rate'] * 100:.1f}% ± {al['chance_sd'] * 100:.1f}, "
              f"p = {al['p_value']:.4g}"
              f"{'  (significant)' if al.get('significant') else '  (not distinguishable from chance)'}")
    if any(stats["misaligned_rows_dropped"].values()):
        print(f"  [warn] dropped stale per-shot rows: {stats['misaligned_rows_dropped']} "
              "— re-run the phase that produced them")
    print(f"  {stats['shots_with_lyrics']}/{stats['total_shots']} shots contain lyrics ({stats['shots_with_lyrics_pct']}%)")
    print(f"  total lyric segments: {stats['total_lyric_segments']} "
          f"({stats['total_lyric_chars']} chars)")

    print(f"\n[next] Phase 8: python scripts/phase8_dashboard.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
