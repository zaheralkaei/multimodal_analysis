"""
Phase 7 — Temporal synchronization + cross-modal analysis.

Reads:
  data/processed/shots.json (Phase 1)
  data/processed/shot_vision.csv (Phase 2)
  data/processed/shot_camera.csv (Phase 3)
  data/processed/transcript.csv (Phase 4)
  data/processed/audio_clap.csv (Phase 5)
  data/processed/music_features.csv (Phase 6)
  data/processed/music_summary.json (Phase 6)

Writes:
  data/processed/sync_per_shot.csv — joined table per shot
  data/processed/sync_stats.json — derived cross-modal signals

Cross-modal signals computed:
  - cut_on_beat: did this shot's start coincide with a beat (within 100ms)?
  - avg_audio_mood_for_shot: mean CLAP mood tag probabilities across shot duration
  - avg_music_energy_for_shot: mean RMS energy across shot duration
  - lyric_count_in_shot: how many transcript segments overlap the shot
  - combined_emotion_score: rough alignment of visual emotion with audio mood
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def within_tolerance(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def join_data(shots: list[dict], vision: list[dict], camera: list[dict],
              transcript: list[dict], clap: list[dict],
              music: list[dict], beats: list[float],
              mood_tags: list[str]) -> tuple[list[dict], dict]:
    """Join all streams by time. Returns (per_shot_rows, stats_dict)."""
    rows = []
    cut_on_beat_count = 0
    total_cuts = 0

    # Index by shot_idx for fast lookup
    vision_by_shot = {int(r["shot_idx"]): r for r in vision}
    camera_by_shot = {int(r["shot_idx"]): r for r in camera}

    for i, shot in enumerate(shots):
        s = float(shot["start_sec"])
        e = float(shot["end_sec"])
        v = vision_by_shot.get(i, {})
        c = camera_by_shot.get(i, {})

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
        beats_in_shot = sum(int(r.get("n_beats", 0)) for r in music_in_shot)

        # Transcript: count segments overlapping
        lyrics_in_shot = [r for r in transcript if float(r["start_sec"]) < e and float(r["end_sec"]) > s and r.get("text", "").strip()]
        lyric_text = " | ".join(r["text"] for r in lyrics_in_shot)[:300]

        # Cut on beat?
        total_cuts += 1
        is_cut_on_beat = any(within_tolerance(s, b, 0.1) for b in beats)
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
            # Camera (from Phase 3)
            "camera_motion": c.get("camera_motion", ""),
            "camera_pan_score": c.get("pan_score_mean", ""),
            "camera_tilt_score": c.get("tilt_score_mean", ""),
            "camera_zoom_score": c.get("zoom_score_mean", ""),
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
    stats = {
        "total_shots": total_cuts,
        "cuts_on_beat": cut_on_beat_count,
        "cuts_on_beat_pct": round(cut_on_beat_count / max(1, total_cuts) * 100, 1),
        "shots_with_lyrics": sum(1 for r in rows if r["n_lyric_segments"] > 0),
        "shots_with_lyrics_pct": round(
            sum(1 for r in rows if r["n_lyric_segments"] > 0) / max(1, len(rows)) * 100, 1
        ),
        "total_lyric_segments": len(transcript),
        "total_lyric_chars": sum(len(r.get("text", "")) for r in transcript),
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

    # Mood tag list (must match phase5)
    mood_tags = [
        "happy and bright", "sad and melancholic", "aggressive and intense",
        "romantic and tender", "triumphant and epic", "calm and peaceful",
        "tense and anxious", "dreamy and ethereal", "dark and ominous",
        "playful and whimsical", "lonely and introspective", "powerful and confident",
    ]

    rows, stats = join_data(shots, vision, camera, transcript, clap, music, beats, mood_tags)

    # Write per-shot CSV
    out_csv = PROCESSED / "sync_per_shot.csv"
    if rows:
        cols = list(rows[0].keys())
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"[ok] wrote {out_csv.relative_to(REPO_ROOT)} ({len(rows)} rows)")

    # Write stats
    out_json = PROCESSED / "sync_stats.json"
    out_json.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_json.relative_to(REPO_ROOT)}")

    print(f"\n[stats]")
    print(f"  {stats['cuts_on_beat']}/{stats['total_shots']} shots cut on a beat ({stats['cuts_on_beat_pct']}%)")
    print(f"  {stats['shots_with_lyrics']}/{stats['total_shots']} shots contain lyrics ({stats['shots_with_lyrics_pct']}%)")
    print(f"  total lyric segments: {stats['total_lyric_segments']} "
          f"({stats['total_lyric_chars']} chars)")

    print(f"\n[next] Phase 8: python scripts/phase8_dashboard.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
