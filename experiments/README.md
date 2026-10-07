# Experiments

One-off scripts behind the findings in `docs/CAMERA_DETECTION.md`. They are not
part of the pipeline and call the Ollama cloud API directly (they need
`OLLAMA_API_KEY` in `.env`). Point them at a processed video with
`PROCESSED_DIR=data/<video_id>`.

| Script | Question | Finding |
|---|---|---|
| `test_multi_image2.py` | Can the vision model read camera motion from a 5-frame sequence? | Unreliable; confuses pan and zoom |
| `test_video_url.py` | Does the cloud API accept a YouTube URL as video input? | No: the model hallucinates content |
| `test_json_format.py` | Does Ollama cloud enforce `format: "json"` / a JSON schema? | Not enforced at the time of testing, so `phase2_vision.py` still validates and normalizes every answer |
