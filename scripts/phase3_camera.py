"""
Phase 3 — Camera movement estimation (global-motion model, round 8).

Reads:  data/<video_id>/shots.json (from Phase 1)
        data/<video_id>/metadata.json — source video path + extraction rate
        the source video itself (decoded directly at --flow-fps)
Writes: data/<video_id>/shot_camera.csv — per-shot camera motion
        data/<video_id>/shot_camera_stats.json — method, thresholds, provenance

Method (replaces the round-1..7 dense-Farneback mean-flow classifier):
  1. Decode each shot's own frames at --flow-fps (default 8) straight from the
     video, skipping a guard band at both ends so no frame pair spans a cut.
     (The old code used the 2 fps JPEGs — 0.5 s apart — and its first pair
     usually straddled the cut.)
  2. Track corner features (Shi-Tomasi + pyramidal Lucas-Kanade) between
     consecutive samples and fit a 4-DoF similarity transform with RANSAC.
     RANSAC inliers = the dominant (background) motion = the camera;
     outliers = independently moving subjects. The old global mean flow
     counted a dancer walking right as a camera pan.
  3. Per pair, convert the transform into CAMERA-convention rates:
       pan_rate   > 0  camera pans right  (content moves left), frame-widths/s
       tilt_rate  > 0  camera tilts down  (content moves up),  frame-heights/s
       zoom_rate  > 0  zoom-in or dolly-in (content expands),  log-scale/s
       roll_rate       degrees/s, positive = counter-clockwise content rotation
     The old classifier had all three signs inverted (pan right was reported
     as pan-left, tilt down as tilt-up, zoom in as zoom-out) — see
     tests/test_camera.py for the synthetic-motion regression tests.
  4. Shot = median of the pair rates; shake = median frame-to-frame change of
     the translation rates (high-frequency jitter = handheld).

Labels: static, pan-left/right, tilt-up/down, zoom-in/out, handheld,
unknown (too few trackable features or no consistent global motion).
`camera_confidence` = high when the label is clearly past its threshold
(>= 2x; static: every component <= 0.5x) AND >= 60 % of tracks agree on the
background motion; otherwise low. Filter on it for analyses until the
thresholds are calibrated.
`camera_components` lists every component above threshold (compound moves).
Zoom vs dolly cannot be separated by a 2-D transform; "zoom" means either.
A large subject filling most of the frame can still be mistaken for the
background (RANSAC follows the majority of tracked points).

THRESHOLDS ARE PROVISIONAL (physically motivated, not yet fitted). They will
be calibrated against the hand-labelled gold set; the raw rates are always
written so labels can be recomputed without re-running.
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from _paths import disp
PROCESSED = REPO_ROOT / "data" / "processed"
if "PROCESSED_DIR" in os.environ:
    PROCESSED = Path(os.environ["PROCESSED_DIR"])
FRAMES_DIR = PROCESSED / "frames"

METHOD = "klt-ransac-similarity-v1"
ANALYSIS_WIDTH = 320          # frames are downscaled to this width for tracking

# Provisional thresholds (see module docstring)
TH_TRANSLATION = 0.04         # frame-widths (or heights) per second
TH_ZOOM = 0.04                # log-scale per second (≈4 %/s)
TH_SHAKE = 0.08               # median |Δ translation rate| between pairs, frames/s
MIN_TRACKS = 25               # minimum tracked features for a usable pair
MIN_INLIER_RATIO = 0.35       # RANSAC consensus below this → no global motion
# RANSAC inlier tolerance: 1 % of the analysis width (3.2 px). 1 px rejected
# most tracks on motion-blurred fast pans/tilts (consensus ~25 %) in the
# round-8 dry run; moving subjects (typically > 5 px/sample) stay outliers.
RANSAC_TOL_PX = 0.01 * ANALYSIS_WIDTH
HIGH_CONF_INLIERS = 0.6       # min RANSAC consensus for a "high" confidence label


def shot_frame_range(start_sec: float, end_sec: float, fps: int) -> tuple[int, int]:
    """Inclusive range of 1-indexed extracted-frame files INSIDE the shot.

    Frame N (1-indexed) samples t = (N-1)/fps. Keep frames with
    start_sec <= t < end_sec:
      first = ceil(start_sec * fps) + 1
      last  = ceil(end_sec * fps)          ((last-1)/fps < end_sec)
    (Round 8: the old `floor(start*fps)+1` admitted a frame from BEFORE the
    cut, so the first flow pair straddled two shots.) Ultra-short shots
    still get one frame.
    """
    first = int(math.ceil(start_sec * fps - 1e-9)) + 1
    last = int(math.ceil(end_sec * fps - 1e-9))
    if last < first:
        last = first
    return first, last


def get_metadata() -> dict:
    meta_path = PROCESSED / "metadata.json"
    if not meta_path.exists():
        print(f"[error] metadata.json not found at {meta_path}")
        print("  run phase 0 first: python scripts/phase0_input.py <source>")
        sys.exit(1)
    return json.loads(meta_path.read_text(encoding="utf-8"))


def get_extraction_fps() -> int:
    """Frame extraction rate from phase-0 metadata.json (audit F1b)."""
    return int(get_metadata().get("frame_fps") or 1)


def sample_times(start_sec: float, end_sec: float, flow_fps: float, guard: float) -> list[float]:
    """Sample timestamps strictly inside [start+guard, end-guard]."""
    a, b = start_sec + guard, end_sec - guard
    if b <= a:
        return []
    step = 1.0 / flow_fps
    n = int(math.floor((b - a) / step + 1e-9)) + 1
    return [a + k * step for k in range(n) if a + k * step <= b + 1e-9]


def sample_frame_indices(start_sec: float, end_sec: float, native_fps: float,
                         flow_fps: float, guard_frames: int = 2) -> list[int]:
    """0-based native frame indices inside the shot, every ~native/flow_fps
    frames, skipping `guard_frames` at each end (no pair may straddle a cut;
    decoder timestamps can be off by a frame). Integer maths — no float
    tolerance matching."""
    first = int(math.ceil(start_sec * native_fps - 1e-6)) + guard_frames
    last = int(math.ceil(end_sec * native_fps - 1e-6)) - 1 - guard_frames
    step = max(1, int(round(native_fps / flow_fps)))
    if last <= first:
        return []
    return list(range(first, last + 1, step))


def estimate_pair(g1, g2):
    """Global similarity motion between two grayscale frames.

    Returns dict(dx, dy, scale, rot_deg, n_tracks, inlier_ratio) in pixels of
    the analysis frame, or None when there are too few trackable features.
    dx/dy are the displacement of the image CENTRE (so a zoom about the
    centre does not leak into pan/tilt).
    """
    import cv2
    import numpy as np
    pts = cv2.goodFeaturesToTrack(g1, maxCorners=500, qualityLevel=0.01,
                                  minDistance=6, blockSize=7)
    if pts is None or len(pts) < MIN_TRACKS:
        return None
    nxt, st, _err = cv2.calcOpticalFlowPyrLK(g1, g2, pts, None, winSize=(21, 21), maxLevel=3)
    # forward-backward check removes unreliable tracks
    back, st2, _ = cv2.calcOpticalFlowPyrLK(g2, g1, nxt, None, winSize=(21, 21), maxLevel=3)
    fb = np.linalg.norm((pts - back).reshape(-1, 2), axis=1)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
    p1, p2 = pts.reshape(-1, 2)[ok], nxt.reshape(-1, 2)[ok]
    if len(p1) < MIN_TRACKS:
        return None
    M, inliers = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC,
                                             ransacReprojThreshold=RANSAC_TOL_PX,
                                             maxIters=2000, confidence=0.99)
    if M is None:
        return None
    h, w = g1.shape[:2]
    a, b = float(M[0, 0]), float(M[1, 0])
    scale = math.hypot(a, b)
    rot = math.degrees(math.atan2(b, a))
    cx, cy = w / 2.0, h / 2.0
    ncx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
    ncy = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
    return {
        "dx": float(ncx - cx), "dy": float(ncy - cy),
        "scale": scale, "rot_deg": rot,
        "n_tracks": int(len(p1)),
        "inlier_ratio": float(inliers.sum()) / len(p1) if inliers is not None else 0.0,
        "w": w, "h": h,
    }


def pair_rates(est: dict, dt: float) -> dict:
    """Convert a pair estimate into camera-convention rates per second."""
    return {
        "pan_rate": -est["dx"] / est["w"] / dt,
        "tilt_rate": -est["dy"] / est["h"] / dt,
        "zoom_rate": math.log(max(est["scale"], 1e-6)) / dt,
        "roll_rate": est["rot_deg"] / dt,
    }


def classify_shot(pairs: list[dict]) -> dict:
    """Aggregate per-pair estimates (each with rates + quality) into a label."""
    import numpy as np
    usable = [p for p in pairs if p is not None and p["inlier_ratio"] >= MIN_INLIER_RATIO]
    out = {"n_pairs": len(pairs), "n_usable_pairs": len(usable)}
    if not usable or len(usable) < max(1, len(pairs) // 2):
        out.update({"camera_motion": "unknown", "camera_confidence": "", "camera_components": "",
                    "pan_rate": "", "tilt_rate": "", "zoom_rate": "", "roll_rate": "",
                    "shake": "", "inlier_ratio": "",
                    "subject_motion_share": ""})
        return out
    pan = float(np.median([p["pan_rate"] for p in usable]))
    tilt = float(np.median([p["tilt_rate"] for p in usable]))
    zoom = float(np.median([p["zoom_rate"] for p in usable]))
    roll = float(np.median([p["roll_rate"] for p in usable]))
    inl = float(np.median([p["inlier_ratio"] for p in usable]))
    if len(usable) >= 3:
        d = [math.hypot(b["pan_rate"] - a["pan_rate"], b["tilt_rate"] - a["tilt_rate"])
             for a, b in zip(usable, usable[1:])]
        shake = float(np.median(d))
    else:
        shake = 0.0

    comps = []
    if abs(pan) >= TH_TRANSLATION:
        comps.append(("pan-right" if pan > 0 else "pan-left", abs(pan) / TH_TRANSLATION))
    if abs(tilt) >= TH_TRANSLATION:
        comps.append(("tilt-down" if tilt > 0 else "tilt-up", abs(tilt) / TH_TRANSLATION))
    if abs(zoom) >= TH_ZOOM:
        comps.append(("zoom-in" if zoom > 0 else "zoom-out", abs(zoom) / TH_ZOOM))
    comps.sort(key=lambda c: -c[1])
    if comps:
        label = comps[0][0]
        strength = comps[0][1]               # multiples of its threshold
    elif shake >= TH_SHAKE:
        label = "handheld"
        strength = shake / TH_SHAKE
    else:
        label = "static"
        # how far below every threshold the shot stays
        strength = 1.0 / max(abs(pan) / TH_TRANSLATION, abs(tilt) / TH_TRANSLATION,
                             abs(zoom) / TH_ZOOM, 0.5)
    # "high" = clearly past the decision boundary AND most tracks agree on
    # one background motion (a frame-filling subject lowers consensus)
    confidence = "high" if strength >= 2.0 and inl >= HIGH_CONF_INLIERS else "low"
    out.update({
        "camera_motion": label,
        "camera_confidence": confidence,
        "camera_components": "+".join(c[0] for c in comps),
        "pan_rate": round(pan, 4), "tilt_rate": round(tilt, 4),
        "zoom_rate": round(zoom, 4), "roll_rate": round(roll, 3),
        "shake": round(shake, 4), "inlier_ratio": round(inl, 3),
        "subject_motion_share": round(1.0 - inl, 3),
    })
    return out


def analyze_video(video_path: Path, shots: list[dict], flow_fps: float) -> list[dict]:
    """Single sequential decode; frames are routed to their shot by timestamp."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # native frame index -> shot index, for the frames we want to keep
    wanted: dict[int, int] = {}
    for i, s in enumerate(shots):
        for fi in sample_frame_indices(float(s["start_sec"]), float(s["end_sec"]),
                                       native_fps, flow_fps):
            wanted[fi] = i
    taken: dict[int, list[tuple[float, object]]] = {i: [] for i in range(len(shots))}
    last_wanted = max(wanted) if wanted else -1

    frame_idx = -1
    while frame_idx < last_wanted:
        if not cap.grab():
            break
        frame_idx += 1
        shot_i = wanted.get(frame_idx)
        if shot_i is None:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        h, w = frame.shape[:2]
        sc = ANALYSIS_WIDTH / float(w)
        small = cv2.resize(frame, (ANALYSIS_WIDTH, max(1, int(round(h * sc)))),
                           interpolation=cv2.INTER_AREA)
        taken[shot_i].append((frame_idx / native_fps,
                              cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)))
    cap.release()

    rows = []
    for i, shot in enumerate(shots):
        samples = taken[i]
        base = {
            "shot_idx": i,
            "start_sec": shot["start_sec"],
            "end_sec": shot["end_sec"],
            "duration_sec": shot["duration_sec"],
            "n_frames": len(samples),
        }
        if len(samples) < 2:
            base.update(classify_shot([]))
            base["camera_motion"] = "unknown"
            rows.append(base)
            continue
        pairs = []
        for (t1, g1), (t2, g2) in zip(samples, samples[1:]):
            est = estimate_pair(g1, g2)
            if est is None:
                pairs.append(None)
                continue
            est.update(pair_rates(est, t2 - t1))
            pairs.append(est)
        base.update(classify_shot(pairs))
        rows.append(base)
        if (i + 1) % 10 == 0 or i == len(shots) - 1:
            print(f"  [{i+1}/{len(shots)}] shots estimated")
    return rows


CSV_COLS = ["shot_idx", "start_sec", "end_sec", "duration_sec", "n_frames",
            "n_pairs", "n_usable_pairs", "camera_motion", "camera_confidence", "camera_components",
            "pan_rate", "tilt_rate", "zoom_rate", "roll_rate", "shake",
            "inlier_ratio", "subject_motion_share"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flow-fps", type=float, default=8.0,
                        help="sampling rate inside each shot for motion estimation (default 8)")
    parser.add_argument("--video", default=None,
                        help="source video (default: metadata.json source_file)")
    args = parser.parse_args()

    shots_path = PROCESSED / "shots.json"
    if not shots_path.exists():
        print(f"[error] shots.json not found at {shots_path}")
        print("  run phase 1 first")
        return 1
    shots = json.loads(shots_path.read_text(encoding="utf-8"))
    print(f"[info] loaded {len(shots)} shots")

    meta = get_metadata()
    video = Path(args.video or meta.get("source_file", ""))
    if not video.is_absolute():
        video = REPO_ROOT / video
    if not video.exists():
        print(f"[error] source video not found: {video}")
        return 1
    print(f"[info] method: {METHOD}, sampling {args.flow_fps:g} fps inside each shot")
    print(f"[info] video: {disp(video)}")

    rows = analyze_video(video, shots, args.flow_fps)

    out_csv = PROCESSED / "shot_camera.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    stats = {
        "method": METHOD,
        "flow_fps": args.flow_fps,
        "analysis_width": ANALYSIS_WIDTH,
        "thresholds": {
            "translation_frames_per_sec": TH_TRANSLATION,
            "zoom_log_scale_per_sec": TH_ZOOM,
            "shake": TH_SHAKE,
            "min_tracks": MIN_TRACKS,
            "min_inlier_ratio": MIN_INLIER_RATIO,
            "ransac_tolerance_px": RANSAC_TOL_PX,
            "high_confidence": "dominant component >= 2x its threshold (static: all "
                               "components <= 0.5x) and inlier ratio >= 0.6",
            "status": "provisional — not yet calibrated on gold labels",
        },
        "sign_convention": "camera: pan_rate>0 = pans right, tilt_rate>0 = tilts down, "
                           "zoom_rate>0 = zoom/dolly in",
        "video": disp(video),
        "n_shots": len(shots),
    }
    (PROCESSED / "shot_camera_stats.json").write_text(json.dumps(stats, indent=2) + "\n",
                                                      encoding="utf-8")
    from _lineage import record
    record(PROCESSED, "phase3", [shots_path, video],
           [out_csv, PROCESSED / "shot_camera_stats.json"],
           {"flow_fps": args.flow_fps, "thresholds": stats["thresholds"]}, __file__)

    from collections import Counter
    counts = Counter(r["camera_motion"] for r in rows)
    print(f"\n[ok] wrote {disp(out_csv)} ({len(rows)} rows)")
    print(f"[stats] camera motion distribution:")
    for direction, count in counts.most_common():
        print(f"   {direction:<14} {count:>4} shots ({count/max(1, len(rows))*100:>5.1f}%)")

    print(f"\n[next] Phase 4: python scripts/phase4_transcribe.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
