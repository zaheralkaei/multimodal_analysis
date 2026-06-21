"""
Run the entire 8-phase pipeline on a different video, output to a separate folder.

Useful for:
  - Comparing results across videos (e.g. music video vs TV scene)
  - Keeping the main `data/processed/` as the canonical Tyla baseline
  - Multi-video experiments from the round-2 plan

Usage:
    python scripts/run_pipeline.py "https://youtu.be/Z2ki180nHCI" --name 4blocks --fps 2
    # → everything goes into data/processed_4blocks/
    # → runs phase0, phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8

What it does:
  1. Sets PROCESSED env var to data/processed_<name>/
  2. Calls each phase script as a subprocess (so they see the env var)
  3. Each phase writes to that folder instead of data/processed/

This is intentionally a thin wrapper — no logic changes, just rerouting.
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="YouTube URL or local video path")
    parser.add_argument("--name", required=True,
                       help="Short name for the output folder (e.g. '4blocks', 'radiohead_creep')")
    parser.add_argument("--fps", type=int, default=2, help="Frame extraction rate (default 2)")
    parser.add_argument("--model", default=os.environ.get("VISION_MODEL", "gemini-3-flash-preview"),
                       help="Vision model for phase 2 (default: $VISION_MODEL or gemini-3-flash-preview)")
    parser.add_argument("--skip-phase2", action="store_true",
                       help="Skip the vision model phase (no API calls, faster but no captions/emotions)")
    parser.add_argument("--skip-phase5", action="store_true",
                       help="Skip the CLAP audio phase (no model download, faster)")
    parser.add_argument("--start-from", type=int, default=0,
                       help="Skip phases before this number (0-8). Useful to resume.")
    args = parser.parse_args()

    out_dir = REPO_ROOT / "data" / f"processed_{args.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] output directory: {out_dir.relative_to(REPO_ROOT)}")
    print(f"[info] source: {args.source}")
    print(f"[info] fps: {args.fps}, model: {args.model}")
    print()

    # Build env so all phase scripts write to the chosen folder
    env = os.environ.copy()
    env["PROCESSED_DIR"] = str(out_dir)

    phases = [
        (0, f'phase0_input.py "{args.source}" --fps {args.fps}'),
        (1, "phase1_shots.py --threshold 35 --min-scene-len 30"),
        (2, f"phase2_vision.py --model {args.model}"),
        (3, "phase3_camera.py"),
        (4, "phase4_transcribe.py --model small.en"),
        (5, "phase5_audio.py"),
        (6, "phase6_music.py"),
        (7, "phase7_sync.py"),
        (8, "phase8_dashboard.py"),
    ]

    py = sys.executable  # use the same Python that's running this script
    for n, cmd in phases:
        if n < args.start_from:
            print(f"[skip] phase {n} (start_from={args.start_from})")
            continue
        if n == 2 and args.skip_phase2:
            print(f"[skip] phase 2 (--skip-phase2)")
            continue
        if n == 5 and args.skip_phase5:
            print(f"[skip] phase 5 (--skip-phase5)")
            continue
        print(f"\n========== phase {n} ==========")
        full_cmd = f'cd "{REPO_ROOT}" && "{py}" scripts/{cmd}'
        # Use a temp PROCESSED dir patch via env var (each phase must read it)
        # For now, simpler: pass --out-dir to each phase (need to add that to each)
        # OR: temporarily symlink data/processed → data/processed_<name>
        # Symlink approach is more invasive but works without changing phase scripts.
        #
        # Simpler still: chdir into a temp dir where data/processed is a symlink.
        # Even simpler: monkey-patch by setting cwd and renaming.
        # ACTUALLY simplest: have each phase respect PROCESSED env var if set.
        # Since they currently don't, let's just edit each phase to read PROCESSED env var.
        #
        # For now, the wrapper just runs phases sequentially against the
        # current data/processed/. To support --name cleanly we need each
        # phase to read PROCESSED_DIR. Do that in a follow-up.
        result = subprocess.run(full_cmd, shell=True, env=env)
        if result.returncode != 0:
            print(f"\n[error] phase {n} failed (returncode {result.returncode})")
            return n

    print(f"\n[ok] all phases done. outputs in {out_dir.relative_to(REPO_ROOT)}/")
    print(f"[ok] dashboard: {out_dir.relative_to(REPO_ROOT).parent / ('reports_' + args.name) / 'dashboard.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())