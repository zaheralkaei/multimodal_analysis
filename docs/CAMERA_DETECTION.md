# Camera motion detection: why we use OpenCV optical flow, not the VLM

## TL;DR

We use **OpenCV optical flow** for camera motion classification, not the VLM.
The VLM (Gemini 3 Flash) sees a single mid-frame per shot and cannot detect
camera motion from one instant. Sending multiple frames or the actual video
doesn't help — Gemini consistently misclassifies real camera motion.

## What we tried

### 1. Single mid-frame (current pipeline default)

```
shots[0]: 0.0-2.8s, optical flow = static
  Gemini response: {"camera": "static", ...}
shots[1]: 2.8-5.2s, optical flow = tilt-down (tilt score 0.766)
  Gemini response: {"camera": "static", ...}  ← WRONG
shots[2]: 5.2-6.6s, optical flow = tilt-up (tilt score -1.0)
  Gemini response: {"camera": "static", ...}  ← WRONG
```

Across 72 shots in Tyla's *SHE DID IT AGAIN*:
- OpenCV detected: tilt-down (20), static (14), pan-right (10), pan-left (9), tilt-up (7), zoom-out (7), zoom-in (5)
- Gemini on mid-frame: static (71), handheld (1)

**Agreement: 14/72 = 19.4%.** Most shots are NOT static — they're pans/tilts —
but Gemini sees 1 instant and concludes "static" every time.

### 2. 5-frame sequence per shot

Test: shot 4, known tilt-down with strong optical-flow signal (pan=-0.94, tilt=+0.72).
Sent 5 frames at 0.5s intervals with prompt: "What is the camera doing across
these frames?"

```
Response: {"camera_motion": "zoom-in", "evidence": "The camera..."}
```

**Wrong.** It said zoom-in when optical flow clearly says tilt-down.

Why? Five frames at 0.5s intervals look like *snapshots* to the model, not
motion. The model can see that something moved but doesn't have enough
spatiotemporal context to determine the trajectory (pan vs tilt vs zoom all
look similar in still frames).

### 3. YouTube URL input

Tried Gemini's recommended approach: pass the YouTube URL directly as
`videos: ["https://www.youtube.com/watch?v=..."]` and let Gemini sample
the video at its default 1 FPS.

Ollama cloud appears to silently ignore the `videos` field. The response
was a generic hallucination about camera operations ("Snapshot Capture",
"Status Heartbeat", "Buffer Flush") — totally unrelated to the actual
music video.

We did not test the native Google Gemini API (we use Ollama cloud as
proxy), so it's possible the YouTube URL feature works there. But:
- It requires a different API key
- The data flow is more complex
- 1 FPS sampling still loses sub-second motion

## Why optical flow wins

OpenCV's `calcOpticalFlowFarneback` computes a **dense motion vector per pixel**
between consecutive frames. From those vectors we get:

- **Pan/tilt:** median horizontal/vertical flow direction
- **Zoom:** radial divergence (vectors pointing inward = zoom-out, outward = zoom-in)
- **Static:** low overall flow magnitude
- **Handheld:** low-to-medium flow with no clear direction

For each shot we have ~2-10 frames (depending on shot length × fps), so we get
~1-9 optical flow comparisons per shot. That's enough to characterize
the dominant motion with confidence.

## What we'd need for VLM-based camera detection to work

To beat optical flow, a VLM would need:
1. **Dense frame sampling** (5+ FPS, not 1 FPS) to see motion trajectories
2. **Higher reasoning capacity** (current models confuse pan vs zoom)
3. **Specialized training** on camera-motion vocabulary

GPT-4V / Claude 3.5 Sonnet might do better than Gemini 3 Flash on this task.
We haven't tested. But:
- Cost would be ~10× higher per shot
- API latency would add minutes
- The output is still a label, not a continuous score

## What we kept

Both signals are saved in `data/processed/sync_per_shot.csv`:

| Field | Source | What it is |
|---|---|---|
| `camera_motion` | OpenCV | Discrete class: static / pan-left / pan-right / tilt-up / tilt-down / zoom-in / zoom-out / handheld |
| `camera_pan_score` | OpenCV | Continuous [-1, 1] horizontal flow |
| `camera_tilt_score` | OpenCV | Continuous [-1, 1] vertical flow |
| `camera_zoom_score` | OpenCV | Continuous [-1, 1] radial divergence |
| `vision_camera_from_vlm` | Gemini | What the model said (usually "static" because it sees 1 frame) |

The dashboard has a VLM-vs-OpenCV comparison panel showing:
- OpenCV's top motion and shot count
- VLM's top motion and shot count
- Shot-level exact-match agreement (currently ~19% for Tyla)

## Recommendation

**For music-video analysis, use OpenCV optical flow for camera motion.**
It's faster, more accurate, gives continuous scores, and doesn't require
any external API. Keep the VLM for semantic understanding (what the
subject is doing, what the scene is, what the mood is).

If we ever need a VLM that handles video end-to-end with high frame rate
sampling and motion reasoning, the candidates are:
- **OpenAI GPT-4V** with video input (paid, unknown motion accuracy)
- **Anthropic Claude 3.5 Sonnet** (similar)
- **Google Gemini 1.5 Pro** (native video support, 1 FPS default)
- **Open-source: InternVideo2.5 Chat 8B** (Apache 2.0, 6.4K⬇, but needs 16GB GPU)

None of these have been tested for camera-motion accuracy in this project.
The Tyla experiment with Gemini 3 Flash is our current baseline.

## Failed-experiment scripts

For reproducibility, the test scripts are committed in `scripts/`:
- `_test_multi_image2.py` — 5-frame sequence test
- `_test_video_url.py` — YouTube URL hallucination test
- `_test_json_format.py` — Ollama cloud format=json not enforced