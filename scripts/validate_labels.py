"""
Measure how well the pipeline agrees with human labels.

Reads:  <PROCESSED>/human_labels.csv   (from the labelling page, label_shots.py)
        <PROCESSED>/shot_vision.csv    (vision-model emotion + camera)
        <PROCESSED>/shot_camera.csv    (optical-flow camera)
Writes: <PROCESSED>/validation.json    (shown in the dashboard by phase 8)

For each comparison it reports accuracy with a 95% Wilson interval, Cohen's
kappa (agreement beyond chance; 0 = chance, 1 = perfect) and the confusion
pairs. Shots the labeller marked "unsure" or left blank are excluded.

Comparisons:
  emotion  vision model vs human (exact label, and coarse valence group)
  camera   optical flow vs human (exact incl. direction, and pan/tilt/zoom family)
  camera   vision model vs human (family; the model has no direction)
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from collections import Counter
from pathlib import Path

from common import PROCESSED, display_path

# Coarse valence groups. "intense" and "surprised" can go either way, so they
# count as neutral rather than forcing a guess.
VALENCE = {
    **dict.fromkeys(["joyful", "playful", "energetic", "confident", "romantic", "sensual"], "positive"),
    **dict.fromkeys(["neutral", "contemplative", "intense", "surprised", "no people"], "neutral"),
    **dict.fromkeys(["sad", "melancholic", "anxious", "fearful", "angry", "disgusted"], "negative"),
}


def camera_family(label: str) -> str:
    label = (label or "").strip().lower()
    if label.startswith(("pan", "tilt", "zoom")):
        return label.split("-")[0]
    return label


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def compare(name: str, human: dict[int, str], pred: dict[int, str], mapper=lambda x: x) -> dict:
    ids = sorted(i for i in human if i in pred and human[i] not in ("", "unsure") and pred[i])
    h = [mapper(human[i]) for i in ids]
    p = [mapper(pred[i]) for i in ids]
    k = sum(x == y for x, y in zip(h, p))
    lo, hi = wilson(k, len(ids))

    def r(v):
        return None if math.isnan(v) else round(v, 3)
    confusions = Counter((x, y) for x, y in zip(h, p) if x != y).most_common(5)
    return {"comparison": name, "n": len(ids), "agree": k,
            "accuracy": r(k / len(ids)) if ids else None, "ci_low": r(lo), "ci_high": r(hi),
            "kappa": r(cohen_kappa(h, p)),
            "top_confusions": [{"human": x, "pipeline": y, "count": c} for (x, y), c in confusions]}


def load(path: Path, key: str) -> dict[int, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {int(r["shot_idx"]): (r.get(key) or "").strip().lower() for r in csv.DictReader(f)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default=str(PROCESSED / "human_labels.csv"))
    args = parser.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"[error] {labels_path} not found — make one with scripts/label_shots.py")
        return 1
    h_emotion, h_camera = load(labels_path, "emotion"), load(labels_path, "camera")
    v_emotion = load(PROCESSED / "shot_vision.csv", "emotion")
    v_camera = load(PROCESSED / "shot_vision.csv", "camera")
    of_camera = load(PROCESSED / "shot_camera.csv", "camera_motion")

    results = [
        compare("emotion: vision model vs human (exact)", h_emotion, v_emotion),
        compare("emotion: vision model vs human (valence)", h_emotion, v_emotion, lambda x: VALENCE.get(x, x)),
        compare("camera: optical flow vs human (exact)", h_camera, of_camera),
        compare("camera: optical flow vs human (family)", h_camera, of_camera, camera_family),
        compare("camera: vision model vs human (family)", h_camera, v_camera, camera_family),
    ]
    out = {"labels_file": display_path(labels_path), "n_labelled_shots": len(h_emotion), "results": results}
    out_path = PROCESSED / "validation.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("| Comparison | n | Accuracy (95% CI) | Cohen's κ |\n|---|---|---|---|")
    for r in results:
        acc = "—" if r["accuracy"] is None else f"{r['accuracy']:.0%} ({r['ci_low']:.0%}–{r['ci_high']:.0%})"
        print(f"| {r['comparison']} | {r['n']} | {acc} | {r['kappa'] if r['kappa'] is not None else '—'} |")
    print(f"\n[ok] wrote {display_path(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
