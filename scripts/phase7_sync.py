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
  data/<video_id>/content_route.json (route step, round 10)
  data/<video_id>/lineage.json (round 10)

Writes:
  data/<video_id>/sync_per_shot.csv — joined table per shot (+ validity flags)
  data/<video_id>/sync_stats.json — derived cross-modal signals, coverage
  data/<video_id>/qa_report.json — cross-module consistency checks

Round 8 (analysis validity):
  - Shot 0 starts at t=0 — that is not an edit; it is excluded from cut
    statistics (cut_on_beat is blank for it).
  - cut-on-beat is reported WITH its chance level (scripts/_stats.py).
  - Vision/camera rows are joined only when their start_sec matches the shot.
  - Transcript segments flagged `suspect` by phase 4 are excluded.

Round 10 (does the assembled pipeline hold together?):
  - Lineage: a stream whose step is stale (its input changed after it ran)
    or overwritten is NOT joined; the QA report says which and why.
  - Route: without music (content_route.json) the cut-on-beat test is
    skipped and per-shot audio mood is marked invalid.
  - Audio values (CLAP mood, RMS) are averaged weighted by how much of each
    window overlaps the shot — a window touching the shot for 0.1 s used to
    count as much as one covering it entirely.
  - Validity flags per shot (valid_vision, valid_camera, valid_camera_high,
    valid_audio_mood, audio_window_shared) and a coverage table: statistics
    and the dashboard use only valid values and say how many there were.
  - qa_report.json: cross-module checks (scripts/_qa.py).

Round 11 (audio stack):
  - Beats come from music_structure.json (Beat This! / All-In-One, beats
    outside music removed) when present, else from phase 6 (librosa).
  - Cut-on-DOWNBEAT test (chance = 2*tol / bar length) and section-boundary
    vs cut test (scripts/_stats.boundary_alignment).
  - sections.csv: one row per song section (cut rate, shot length, energy,
    camera, vocals, emotion) — the unit of analysis for music questions,
    since most shots are shorter than the audio windows.
  - Per-shot `section`, `cut_on_downbeat`, `vocals_active_share`.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from _paths import disp
PROCESSED = REPO_ROOT / "data" / "processed"
if "PROCESSED_DIR" in os.environ:
    PROCESSED = Path(os.environ["PROCESSED_DIR"])

BEAT_TOLERANCE_SEC = 0.1
AUDIO_MOOD_MIN_MUSIC_SHARE = 0.5   # a shot's audio mood counts only over music
BAD_VISION_STATUS = {"error", "parse_error", "budget", "mock"}

# lineage step -> stream it produced
STEP_STREAM = {"phase2": "vision", "phase3": "camera", "phase4": "transcript",
               "phase5": "clap", "phase6": "music", "phase1": "shots",
               "structure": "structure", "separate": "separation"}


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def within_tolerance(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


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


def _vision_ok(v: dict) -> bool:
    if not v:
        return False
    if v.get("status", "") in BAD_VISION_STATUS:
        return False
    return not v.get("caption", "").startswith("[")


def _audio_type(row: dict) -> str | None:
    from _clap_tags import window_audio_type
    return window_audio_type(row)


def _pct(valid: int, total: int) -> dict:
    return {"valid": valid, "total": total,
            "pct": round(100 * valid / total, 1) if total else None}


def join_data(shots: list[dict], vision: list[dict], camera: list[dict],
              transcript: list[dict], clap: list[dict],
              music: list[dict], beats: list[float],
              mood_tags: list[str], music_present: bool | None = None,
              clap_window_sec: float = 5.0, downbeats: list | None = None,
              segments: list | None = None, vocal_activity: list | None = None
              ) -> tuple[list[dict], dict]:
    """Join all streams by time. Returns (per_shot_rows, stats_dict).

    music_present: False = route found no music (no beat test, no audio mood);
    None = unknown (treated as present, as before round 10).
    """
    rows = []
    cut_on_beat_count = 0
    total_cuts = 0
    misaligned = {"vision": 0, "camera": 0}
    usable_transcript = [r for r in transcript if not _is_suspect(r)]
    music_ok = music_present is not False
    test_beats = music_ok and bool(beats)

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

        # CLAP: overlap-weighted mean over windows touching the shot
        w_clap = [(overlap(s, e, float(r["start_sec"]), float(r["end_sec"])), r) for r in clap]
        w_clap = [(w, r) for w, r in w_clap if w > 0]
        wsum = sum(w for w, _ in w_clap)
        avg_clap = {tag: 0.0 for tag in mood_tags}
        if wsum > 0:
            for tag in mood_tags:
                avg_clap[tag] = round(sum(w * float(r[tag]) for w, r in w_clap if tag in r) / wsum, 4)
        top_mood = max(avg_clap, key=avg_clap.get) if any(avg_clap.values()) else "n/a"
        typed = [(w, _audio_type(r)) for w, r in w_clap]
        typed = [(w, t) for w, t in typed if t is not None]
        music_share_shot = (sum(w for w, t in typed if t == "music") / sum(w for w, _ in typed)
                            if typed else None)
        valid_mood = bool(wsum > 0 and music_ok
                          and (music_share_shot is None
                               or music_share_shot >= AUDIO_MOOD_MIN_MUSIC_SHARE))

        # Music features: overlap-weighted mean RMS
        w_rms = [(overlap(s, e, float(r["start_sec"]), float(r["end_sec"])), r) for r in music
                 if r.get("rms_energy")]
        w_rms = [(w, r) for w, r in w_rms if w > 0]
        rsum = sum(w for w, _ in w_rms)
        avg_rms = round(sum(w * float(r["rms_energy"]) for w, r in w_rms) / rsum, 5) if rsum else 0.0
        # beats whose time falls inside the shot (audit R9)
        beats_in_shot = sum(1 for b in beats if s <= float(b) < e)

        lyrics_in_shot = [r for r in usable_transcript if float(r["start_sec"]) < e
                          and float(r["end_sec"]) > s and r.get("text", "").strip()]
        lyric_text = " | ".join(r["text"] for r in lyrics_in_shot)[:300]

        # Cut on beat? (shot 0 starts the video — not a cut; no music — no test)
        if i == 0 or not test_beats:
            is_cut_on_beat = ""
        else:
            total_cuts += 1
            is_cut_on_beat = any(within_tolerance(s, b, BEAT_TOLERANCE_SEC) for b in beats)
            if is_cut_on_beat:
                cut_on_beat_count += 1

        camera_label = c.get("camera_motion", "")
        section = next((sg.get("label", "") for sg in (segments or [])
                        if float(sg["start"]) <= s < float(sg["end"])), "")
        on_downbeat = ("" if (i == 0 or not test_beats or not downbeats) else
                       any(within_tolerance(s, d, BEAT_TOLERANCE_SEC) for d in downbeats))
        va = [(overlap(s, e, float(r["start_sec"]), float(r["end_sec"])), str(r["active"]).lower() == "true")
              for r in (vocal_activity or [])]
        va = [(w, a) for w, a in va if w > 0]
        vocals_share = round(sum(w for w, a in va if a) / sum(w for w, _ in va), 3) if va else ""
        rows.append({
            "shot_idx": i,
            "start_sec": s,
            "end_sec": e,
            "duration_sec": round(e - s, 3),
            # Visual (from Phase 2)
            "vision_caption": v.get("caption", ""),
            "vision_camera_from_vlm": v.get("camera", ""),
            "vision_emotion": v.get("emotion", ""),
            "vision_emotion_raw": v.get("emotion_raw", ""),
            "vision_colors": v.get("colors", ""),
            "vision_entities": v.get("entities", ""),
            "vision_location": v.get("location", ""),
            "vision_lighting": v.get("lighting", ""),
            "vision_composition": v.get("composition", ""),
            "mid_frame": v.get("mid_frame", "") or shot.get("mid_frame_path", ""),
            # Camera (from Phase 3)
            "camera_motion": camera_label,
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
            "audio_top_mood": top_mood if valid_mood else "",
            **{f"audio_mood_{k}": avg_clap[k] for k in mood_tags},
            "audio_music_share": "" if music_share_shot is None else round(music_share_shot, 3),
            # Music (from Phase 6)
            "music_avg_rms": avg_rms,
            "music_n_beats": beats_in_shot if music_ok else "",
            # Cross-modal
            "cut_on_beat": is_cut_on_beat,
            "cut_on_downbeat": on_downbeat,
            "section": section,
            "vocals_active_share": vocals_share,
            "n_lyric_segments": len(lyrics_in_shot),
            "lyric_text": lyric_text,
            # Validity (round 10): use a value only where its flag is True
            "valid_vision": _vision_ok(v),
            "valid_camera": camera_label not in ("", "unknown"),
            "valid_camera_high": c.get("camera_confidence", "") == "high",
            "valid_audio_mood": valid_mood,
            "audio_window_shared": (e - s) < clap_window_sec,
        })

    from _stats import beat_alignment
    cut_times = [float(sh["start_sec"]) for sh in shots[1:]]
    duration = float(shots[-1]["end_sec"]) if shots else 0.0
    if test_beats:
        alignment = beat_alignment(cut_times, beats, duration, tol=BEAT_TOLERANCE_SEC)
    else:
        alignment = {"skipped": True, "significant": False,
                     "chance_rate": None, "p_value": None,
                     "note": ("no music detected — cut-on-beat is not meaningful"
                              if not music_ok else "no beats")}
    if test_beats and downbeats:
        downbeat_alignment = beat_alignment(cut_times, downbeats, duration, tol=BEAT_TOLERANCE_SEC)
    else:
        downbeat_alignment = {"skipped": True, "significant": False, "chance_rate": None,
                              "p_value": None, "note": "no downbeats"}
    from _stats import boundary_alignment
    inner = [float(sg["start"]) for sg in (segments or [])[1:]]
    section_alignment = (boundary_alignment(inner, cut_times, duration) if segments
                         else {"skipped": True, "note": "no sections"})
    n = len(rows)
    coverage = {
        "vision": _pct(sum(r["valid_vision"] for r in rows), n),
        "camera": _pct(sum(r["valid_camera"] for r in rows), n),
        "camera_high_confidence": _pct(sum(r["valid_camera_high"] for r in rows), n),
        "audio_mood": _pct(sum(r["valid_audio_mood"] for r in rows), n),
        "transcript_segments_usable": _pct(len(usable_transcript), len(transcript)),
        "shots_shorter_than_audio_window": _pct(sum(r["audio_window_shared"] for r in rows), n),
    }
    stats = {
        "total_shots": n,
        "total_cuts": total_cuts,
        "cuts_on_beat": cut_on_beat_count,
        "cuts_on_beat_pct": round(cut_on_beat_count / max(1, total_cuts) * 100, 1),
        "cut_beat_alignment": alignment,
        "cut_downbeat_alignment": downbeat_alignment,
        "section_boundary_alignment": section_alignment,
        "music_present": music_present,
        "coverage": coverage,
        "misaligned_rows_dropped": misaligned,
        "suspect_transcript_segments_excluded": len(transcript) - len(usable_transcript),
        "shots_with_lyrics": sum(1 for r in rows if r["n_lyric_segments"] > 0),
        "shots_with_lyrics_pct": round(
            sum(1 for r in rows if r["n_lyric_segments"] > 0) / max(1, n) * 100, 1),
        "total_lyric_segments": len(usable_transcript),
        "total_lyric_chars": sum(len(r.get("text", "")) for r in usable_transcript),
        "avg_music_rms": round(sum(float(r.get("music_avg_rms", 0)) for r in rows) / max(1, n), 5),
        "clap_mood_tags": mood_tags,
    }
    return rows, stats


def section_table(rows: list, segments: list, music: list) -> list:
    """One row per section: the unit of analysis for music-related questions."""
    from collections import Counter
    out = []
    for sg in segments or []:
        a, b = float(sg["start"]), float(sg["end"])
        dur = max(1e-6, b - a)
        shots = [r for r in rows if a <= (float(r["start_sec"]) + float(r["end_sec"])) / 2 < b]
        cuts = [r for r in rows if r["shot_idx"] != 0 and a <= float(r["start_sec"]) < b]
        rms = [(overlap(a, b, float(m["start_sec"]), float(m["end_sec"])), float(m["rms_energy"]))
               for m in music if m.get("rms_energy")]
        rms = [(w, v) for w, v in rms if w > 0]
        cams = [r["camera_motion"] for r in shots if r["valid_camera"]]
        moving = [c for c in cams if c not in ("static", "handheld")]
        emos = [r["vision_emotion"] for r in shots if r["valid_vision"]
                and r["vision_emotion"] not in ("", "no people", "other")]
        vocal = [float(r["vocals_active_share"]) for r in shots if r["vocals_active_share"] != ""]
        out.append({
            "label": sg.get("label", ""), "start_sec": round(a, 3), "end_sec": round(b, 3),
            "duration_sec": round(dur, 3), "n_shots": len(shots),
            "cuts_per_min": round(len(cuts) / dur * 60, 2),
            "mean_shot_sec": round(sum(float(r["duration_sec"]) for r in shots) / len(shots), 3) if shots else "",
            "mean_rms": round(sum(w * v for w, v in rms) / sum(w for w, _ in rms), 5) if rms else "",
            "camera_moving_share": round(len(moving) / len(cams), 3) if cams else "",
            "dominant_camera": Counter(cams).most_common(1)[0][0] if cams else "",
            "vocals_active_share": round(sum(vocal) / len(vocal), 3) if vocal else "",
            "dominant_emotion": Counter(emos).most_common(1)[0][0] if emos else "",
            "n_valid_vision": sum(1 for r in shots if r["valid_vision"]),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    shots_path = PROCESSED / "shots.json"
    if not shots_path.exists():
        print(f"[error] shots.json not found")
        return 1
    shots = json.loads(shots_path.read_text(encoding="utf-8"))

    # Lineage: refuse streams whose producing step no longer matches its inputs
    from _lineage import check as lineage_check, record
    lin = lineage_check(PROCESSED, steps=list(STEP_STREAM))
    dropped = sorted({STEP_STREAM[p["step"]] for p in lin["problems"]})
    if "shots" in dropped:
        print("[error] shots.json is stale or overwritten (lineage) — re-run phase 1 and later")
        for p in lin["problems"]:
            print(f"   {p['step']}: {p['kind']} {p['file']}")
        return 1
    for p in lin["problems"]:
        print(f"[warn] lineage: {p['step']} {p['kind']} ({p['file']}) — "
              f"{STEP_STREAM[p['step']]} stream not joined")

    def stream(name, loader):
        return loader() if name not in dropped else ([] if name != "music" else {})

    vision = stream("vision", lambda: load_csv(PROCESSED / "shot_vision.csv"))
    camera = stream("camera", lambda: load_csv(PROCESSED / "shot_camera.csv"))
    transcript = stream("transcript", lambda: load_csv(PROCESSED / "transcript.csv"))
    clap = stream("clap", lambda: load_csv(PROCESSED / "audio_clap.csv"))
    music = stream("music", lambda: load_csv(PROCESSED / "music_features.csv")) or []
    music_summary = stream("music", lambda: load_json(PROCESSED / "music_summary.json")) or {}
    structure = stream("structure", lambda: load_json(PROCESSED / "music_structure.json")) or {}
    if structure.get("beats"):
        beats, beats_source = structure["beats"], structure.get("beats_source", "structure")
    else:
        beats, beats_source = music_summary.get("beat_times", []), "librosa (phase 6)"
    downbeats = structure.get("downbeats") or []
    segments = structure.get("segments") or []
    vocal_activity = stream("separation", lambda: load_csv(PROCESSED / "vocal_activity.csv")) or []

    from route_content import load_route
    route = load_route(PROCESSED)
    if route is not None:
        music_present = bool(route.get("has_music"))
    elif music_summary.get("beats_valid") is False:
        music_present = False
    else:
        music_present = None

    print(f"[info] shots: {len(shots)}, vision: {len(vision)}, camera: {len(camera)}, "
          f"transcript: {len(transcript)}, clap: {len(clap)}, music: {len(music)}, "
          f"beats: {len(beats)}, music present: {music_present}")

    from _clap_tags import MOOD_TAGS
    clap_stats = load_json(PROCESSED / "audio_clap_stats.json")
    rows, stats = join_data(shots, vision, camera, transcript, clap, music, beats, MOOD_TAGS,
                            music_present=music_present,
                            clap_window_sec=float(clap_stats.get("window_sec", 5.0)),
                            downbeats=downbeats, segments=segments, vocal_activity=vocal_activity)
    stats["beats_source"] = beats_source
    stats["n_downbeats"] = len(downbeats)
    stats["sections"] = {"n": len(segments), "source": structure.get("sections_source"),
                         "labelled": structure.get("sections_labelled")}
    sep = load_json(PROCESSED / "separation_stats.json")
    tstats = load_json(PROCESSED / "transcript_stats.json")
    stats["audio_stack"] = {"separation": sep.get("provider"), "separation_passthrough": sep.get("is_passthrough"),
                            "asr_backend": tstats.get("backend"), "asr_model": tstats.get("model"),
                            "asr_input": tstats.get("input")}
    from _cloud import spent
    stats["cloud_audio_est_usd"] = spent(PROCESSED)

    vstats = load_json(PROCESSED / "shot_vision_stats.json") if "vision" not in dropped else {}
    if vstats:
        stats["vision_model"] = f"{vstats.get('model', '?')}"
        stats["vision_provider"] = vstats.get("provider", "?")
        stats["vision_endpoint"] = vstats.get("endpoint", "?")
    stats["streams_not_joined"] = dropped
    stats["route"] = None if route is None else {
        k: route.get(k) for k in ("audio_class", "audio_shares", "has_music", "has_music_basis",
                                  "content_type_override", "reasons")}

    # Write per-shot CSV — ALWAYS (round-4 audit low-severity fix)
    out_csv = PROCESSED / "sync_per_shot.csv"
    cols = list(rows[0].keys()) if rows else ["shot_idx"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[ok] wrote {disp(out_csv)} ({len(rows)} rows)")
    sec_rows = section_table(rows, segments, music)
    out_sec = PROCESSED / "sections.csv"
    if sec_rows:
        with out_sec.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sec_rows[0].keys()))
            w.writeheader()
            w.writerows(sec_rows)
        print(f"[ok] wrote {disp(out_sec)} ({len(sec_rows)} sections)")
    else:
        out_sec.unlink(missing_ok=True)

    # QA report across modules
    from _qa import build_report
    meta = load_json(PROCESSED / "metadata.json")
    qa = build_report(rows=rows, coverage=stats["coverage"], route=route, lineage=lin,
                      music_summary=music_summary, transcript=transcript, clap=clap,
                      transcript_stats=load_json(PROCESSED / "transcript_stats.json"),
                      vision_stats=vstats, repo_root=REPO_ROOT,
                      frames_dir=PROCESSED / "frames", frame_fps=int(meta.get("frame_fps") or 0),
                      audio_stack=stats.get("audio_stack"), structure=structure)
    stats["qa_summary"] = qa["summary"]
    out_qa = PROCESSED / "qa_report.json"
    out_qa.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")

    out_json = PROCESSED / "sync_stats.json"
    out_json.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {disp(out_json)}, {disp(out_qa)}")

    inputs = [shots_path] + [PROCESSED / f for f in (
        "shot_vision.csv", "shot_camera.csv", "transcript.csv", "audio_clap.csv",
        "music_features.csv", "music_summary.json", "content_route.json",
        "music_structure.json", "vocal_activity.csv")
        if (PROCESSED / f).exists()]
    record(PROCESSED, "phase7", inputs, [out_csv, out_json, out_qa],
           {"beat_tolerance_sec": BEAT_TOLERANCE_SEC,
            "audio_mood_min_music_share": AUDIO_MOOD_MIN_MUSIC_SHARE,
            "streams_not_joined": dropped}, __file__)

    print(f"\n[stats]")
    al = stats["cut_beat_alignment"]
    if al.get("skipped") or al.get("chance_rate") is None:
        print(f"  cut-on-beat: not tested ({al.get('note', 'no beats')})")
    else:
        print(f"  {stats['cuts_on_beat']}/{stats['total_cuts']} cuts on a beat ({stats['cuts_on_beat_pct']}%); "
              f"chance {al['chance_rate'] * 100:.1f}% ± {al['chance_sd'] * 100:.1f}, "
              f"p = {al['p_value']:.4g}"
              f"{'  (significant)' if al.get('significant') else '  (not distinguishable from chance)'}")
    for key, label in (("cut_downbeat_alignment", "cut-on-downbeat"),
                       ("section_boundary_alignment", "section boundaries on cuts")):
        a = stats[key]
        if a.get("chance_rate") is not None:
            print(f"  {label}: {a['observed_rate'] * 100:.0f}% vs chance {a['chance_rate'] * 100:.0f}%, "
                  f"p = {a['p_value']:.3g}{'  (significant)' if a.get('significant') else ''}")
    cov = stats["coverage"]
    print("  coverage: " + ", ".join(f"{k} {v['valid']}/{v['total']}" for k, v in cov.items()))
    if any(stats["misaligned_rows_dropped"].values()):
        print(f"  [warn] dropped stale per-shot rows: {stats['misaligned_rows_dropped']}")
    print(f"  {stats['shots_with_lyrics']}/{stats['total_shots']} shots contain lyrics "
          f"({stats['shots_with_lyrics_pct']}%)")
    qs = qa["summary"]
    print(f"  QA: {qs['critical']} critical, {qs['warning']} warnings, {qs['info']} info "
          f"-> {disp(out_qa)}")
    for it in qa["issues"][:8]:
        where = f" @ {it['time_sec']}s" if it["time_sec"] is not None else ""
        print(f"    [{it['severity']}] {it['check']}{where}: {it['message'][:110]}")

    print(f"\n[next] Phase 8: python scripts/phase8_dashboard.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
