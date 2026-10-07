"""
Make a labelling page for building a small human-labelled validation set.

Reads:  <PROCESSED>/shots.json (+ key frames)
Writes: <REPORTS>/labeling.html — open it in a browser, label, click "Download CSV"

The page shows a random sample of shots (fixed seed, so it is reproducible),
each as its three key frames, and asks for the same judgements the pipeline
makes automatically: dominant emotion and camera motion. The model's own
answers are deliberately NOT shown, so they cannot anchor the labeller.

The downloaded file (human_labels.csv) goes into data/<video_id>/; then run
scripts/validate_labels.py to measure agreement.

Usage:
  PROCESSED_DIR=data/<video_id> REPORTS_DIR=reports/<video_id> \\
      python scripts/label_shots.py --n 50
"""
from __future__ import annotations
import argparse, html, json, random, sys

from _normalize_emotion import list_canonical
from common import PROCESSED, REPORTS, display_path
from phase8_dashboard import thumbnail_data_uri

CAMERA_CHOICES = ["static", "pan-left", "pan-right", "tilt-up", "tilt-down", "zoom-in", "zoom-out", "handheld"]
EMOTION_CHOICES = list_canonical() + ["no people"]


def select(name: str, choices: list[str]) -> str:
    opts = "".join(f"<option>{html.escape(c)}</option>" for c in choices)
    return f"<select name='{name}'><option value=''>—</option>{opts}<option>unsure</option></select>"


def build_page(shots: list[dict], video_id: str) -> str:
    cards = []
    for s in shots:
        frames = s.get("key_frame_paths") or [s["mid_frame_path"]]
        imgs = "".join(f"<img src='{thumbnail_data_uri(p, 320)}' alt=''>" for p in frames)
        cards.append(
            f"<div class='card' data-shot='{s['shot_idx']}'><div class='head'>Shot {s['shot_idx']} · "
            f"{s['start_sec']:.1f}–{s['end_sec']:.1f}s</div><div class='frames'>{imgs}</div>"
            f"<label>Emotion {select('emotion', EMOTION_CHOICES)}</label>"
            f"<label>Camera {select('camera', CAMERA_CHOICES)}</label>"
            f"<label>Notes <input name='notes' size='40'></label></div>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shot labelling</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", sans-serif; max-width: 1100px; margin: 0 auto; padding: 16px; background: #fff; color: #222; }}
.card {{ border: 1px solid #ccc; border-radius: 6px; padding: 10px; margin: 14px 0; }}
.card.done {{ border-color: #2ca02c; background: #f3fbf3; }}
.head {{ font-weight: 600; margin-bottom: 6px; }}
.frames {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }}
.frames img {{ width: 320px; max-width: 100%; height: auto; }}
label {{ margin-right: 18px; }}
#bar {{ position: sticky; top: 0; background: #fff; padding: 8px 0; border-bottom: 1px solid #ddd; }}
</style></head><body>
<h1>Label {len(shots)} shots — {html.escape(video_id)}</h1>
<p>For each shot, three frames are shown in time order (early, middle, late).
<b>Emotion</b>: the dominant emotion of the people shown ("no people" if none).
<b>Camera</b>: what the camera does across the frames; directions are the camera's
(pan-right = the camera turns right, so the scene slides left). Use "unsure" rather than guessing.
Your answers are saved in this browser as you go.</p>
<div id="bar"><button id="dl">Download CSV</button> <span id="progress"></span></div>
{''.join(cards)}
<script>
const KEY = 'labels:{html.escape(video_id)}';
const cards = [...document.querySelectorAll('.card')];
let saved = {{}};
try {{ saved = JSON.parse(localStorage.getItem(KEY) || '{{}}'); }} catch (e) {{}}
function update() {{
  let done = 0;
  for (const c of cards) {{
    const v = Object.fromEntries([...c.querySelectorAll('select,input')].map(e => [e.name, e.value]));
    saved[c.dataset.shot] = v;
    const complete = v.emotion && v.camera;
    c.classList.toggle('done', !!complete);
    if (complete) done++;
  }}
  document.getElementById('progress').textContent = done + ' / ' + cards.length + ' labelled';
  try {{ localStorage.setItem(KEY, JSON.stringify(saved)); }} catch (e) {{}}
}}
for (const c of cards) {{
  const v = saved[c.dataset.shot] || {{}};
  c.querySelectorAll('select,input').forEach(e => {{ if (v[e.name]) e.value = v[e.name]; e.addEventListener('change', update); }});
}}
update();
document.getElementById('dl').onclick = () => {{
  update();
  const q = s => '"' + String(s || '').replace(/"/g, '""') + '"';
  const lines = ['shot_idx,emotion,camera,notes'];
  for (const c of cards) {{ const v = saved[c.dataset.shot]; lines.push([c.dataset.shot, q(v.emotion), q(v.camera), q(v.notes)].join(',')); }}
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lines.join('\\n') + '\\n'], {{type: 'text/csv'}}));
  a.download = 'human_labels.csv';
  a.click();
}};
</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=50, help="number of shots to sample (default 50)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    shots_path = PROCESSED / "shots.json"
    if not shots_path.exists():
        print(f"[error] {shots_path} not found; run the pipeline first")
        return 1
    shots = json.loads(shots_path.read_text(encoding="utf-8"))
    sample = sorted(random.Random(args.seed).sample(shots, min(args.n, len(shots))), key=lambda s: s["shot_idx"])
    video_id = PROCESSED.name
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "labeling.html"
    out.write_text(build_page(sample, video_id), encoding="utf-8")
    print(f"[ok] wrote {display_path(out)} ({len(sample)} shots)")
    print(f"[next] label in a browser, save human_labels.csv to {display_path(PROCESSED)}/, then run "
          f"PROCESSED_DIR={display_path(PROCESSED)} python scripts/validate_labels.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
