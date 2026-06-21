"""
Phase 1 — Shot boundary detection with TransNetV2.

Reads:  data/processed/frames/frame_*.jpg (from Phase 0)
Writes: data/processed/shots.json — list of {start_sec, end_sec, mid_sec, mid_frame_path}
        data/processed/shot_predictions.csv — per-frame TransNetV2 scores (debug)

TransNetV2 expects frames at 27x48 RGB at the original video's frame rate.
We extracted at 1fps, so we tell TransNetV2 to use fps=1.

Model weights live in /tmp/transnetv2-pytorch-weights.pth (downloaded from HF).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"
FRAMES_DIR = PROCESSED / "frames"
WEIGHTS_PATH = Path("/tmp/transnetv2-pytorch-weights.pth")
TRANSNET_DIR = Path("/c/Users/zaher/AppData/Local/Temp/TransNetV2/inference-pytorch")


def load_transnetv2():
    """Load TransNetV2 model. We vendor the PyTorch implementation."""
    sys.path.insert(0, str(TRANSNET_DIR))
    try:
        from transnetv2_pytorch import TransNetV2
        import torch
    except ImportError as e:
        print(f"[error] could not import transnetv2_pytorch: {e}")
        print(f"  ensure {TRANSNET_DIR} is cloned and has transnetv2_pytorch.py")
        sys.exit(1)
    if not WEIGHTS_PATH.exists():
        print(f"[error] weights not found at {WEIGHTS_PATH}")
        print(f"  download from https://huggingface.co/MiaoshouAI/transnetv2-pytorch-weights")
        sys.exit(1)
    model = TransNetV2()
    state_dict = torch.load(str(WEIGHTS_PATH), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def detect_shots(model, frames_dir: Path, frame_fps: int = 1) -> list[dict]:
    """Run TransNetV2 on extracted frames. Returns list of shot dicts.

    TransNetV2 returns per-frame predictions. A shot boundary is where the
    single-frame-prediction exceeds 0.5.
    """
    import numpy as np
    import torch
    from PIL import Image

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        print(f"[error] no frames in {frames_dir}")
        sys.exit(1)
    print(f"[info] loading {len(frames)} frames ...")
    arrs = []
    for f in frames:
        img = Image.open(f).convert("RGB").resize((48, 27))
        arrs.append(np.array(img))
    video = np.stack(arrs)[None, ...]  # (1, T, 27, 48, 3)
    print(f"[info] running TransNetV2 on shape {video.shape} ...")
    with torch.no_grad():
        single_frame_pred, all_frame_pred = model(torch.from_numpy(video))
        single_frame_pred = torch.sigmoid(single_frame_pred[0]).cpu().numpy()

    # A shot boundary is where single_frame_pred > 0.5
    boundaries = [i for i, p in enumerate(single_frame_pred) if p > 0.5]
    # Convert frame indices to (start, end) pairs in seconds
    # Frame i is at time i / frame_fps
    starts = [0] + [b for b in boundaries]
    ends = [b for b in boundaries] + [len(frames) - 1]

    shots = []
    for s, e in zip(starts, ends):
        start_sec = s / frame_fps
        end_sec = e / frame_fps
        mid_idx = (s + e) // 2
        shots.append({
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "duration_sec": round(end_sec - start_sec, 3),
            "mid_sec": round(mid_idx / frame_fps, 3),
            "mid_frame_path": str((frames_dir / f"frame_{mid_idx+1:05d}.jpg").relative_to(REPO_ROOT)),
            "n_frames": e - s + 1,
            "boundary_score": float(single_frame_pred[s]) if s == 0 else float(single_frame_pred[s]),
        })
    print(f"[ok] detected {len(shots)} shots")
    return shots, single_frame_pred.tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fps", type=int, default=1, help="fps of extracted frames (must match phase0)")
    args = parser.parse_args()

    if not FRAMES_DIR.exists():
        print(f"[error] frames dir not found: {FRAMES_DIR}")
        print("  run phase 0 first: python scripts/phase0_input.py <source>")
        return 1

    # Load metadata for sanity
    meta_path = PROCESSED / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"[info] video duration: {meta['duration_sec']:.1f}s")
        print(f"[info] frames: {meta['frames_extracted']} at {meta['frame_fps']}fps")

    model = load_transnetv2()
    shots, per_frame_scores = detect_shots(model, FRAMES_DIR, args.fps)

    # Write shots.json
    out_path = PROCESSED / "shots.json"
    out_path.write_text(json.dumps(shots, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_path.relative_to(REPO_ROOT)} ({len(shots)} shots)")

    # Write per-frame predictions (debug)
    pred_path = PROCESSED / "shot_predictions.csv"
    import csv
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "time_sec", "boundary_score"])
        for i, s in enumerate(per_frame_scores):
            w.writerow([i, round(i / args.fps, 3), round(float(s), 4)])
    print(f"[ok] wrote {pred_path.relative_to(REPO_ROOT)} ({len(per_frame_scores)} rows)")

    # Stats
    durations = [s["duration_sec"] for s in shots]
    avg = sum(durations) / len(durations) if durations else 0
    print(f"\n[stats] {len(shots)} shots, avg duration {avg:.1f}s, "
          f"min {min(durations):.1f}s, max {max(durations):.1f}s")

    print(f"\n[next] Phase 2: python scripts/phase2_qwen_vision.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
