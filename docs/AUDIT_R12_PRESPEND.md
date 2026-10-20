# Audit round 12 — before the first paid run

Date: 2026-10-20 · Goal: make paid re-runs unnecessary. A bug found after
API calls costs a generate → pay → fix → regenerate loop, so every paid code
path was exercised **for free** first.

## How it was tested

| Method | What it proves |
|---|---|
| **Fake cloud** (`scripts/fake_cloud.py`) — a local server that implements fal's queue API, Groq's transcription endpoint, Replicate's models/files/predictions and OpenRouter's chat/models as documented, and **rejects malformed requests with HTTP 400** | The real client code (auth headers, JSON bodies, multipart forms, queue polling, downloads, caching, cost logs) and every phase that consumes the results work end to end. `calibration.py run --fake-cloud`: **20/20 checks**, 0 rejected requests. |
| Idempotence: the fake-cloud calibration run twice | The second run is all cache hits (logged `cached`, $0) and phase 2 makes **0** model calls. The real cache folder holds no fake answers. |
| Fault injection (`tests/test_fake_cloud.py`, 14 tests) | fal job failing as `COMPLETED + error`, Groq 500, Replicate `failed`, OpenRouter 429, OpenRouter non-JSON answers, malformed responses: none yields silent wrong data, none poisons the cache, none is logged as a paid result. |
| Edge-case inputs | no audio track, 3-second clip, silent audio, portrait 360×640, filename with spaces and `ü é` — all complete with sensible routing/QA. |
| Interruption | a re-run interrupted mid-phase-2 keeps every answer paid for earlier. |
| Static analysis (pyflakes) | no undefined names; unused variables removed. |
| Docs re-check | fal queue protocol, Replicate HTTP API, Groq STT — against the official pages. |

## Bugs fixed (each has a regression test)

| # | Severity | Finding | Fix |
|---|---|---|---|
| P1 | **High (money)** | Interrupted phase-2 re-run **deleted not-yet-reached answers** from `shot_vision.csv` (the file was rewritten top to bottom). | All answers kept in memory; the whole file is rewritten atomically after each answer. |
| P2 | **High (money)** | No circuit breaker: an invalid key, removed model, or a model that never answers in JSON was called for **every** shot. | Stop after 5 consecutive errors *or unparseable answers*; exit non-zero; rows retried next run. |
| P3 | **High (data)** | Cache keys and phase-2 row reuse ignored the **endpoint**: answers from the fake/test server could be reused by a real run on the same audio. | Endpoint is part of every cache key and of every reused row; fake cloud has its own cache folder. |
| P4 | High | fal reports a failed job as `COMPLETED` with an `error` field; it was treated as success. | Error field → failure; unknown status → failure. |
| P5 | High | A video with **no cut** got **zero shots** (PySceneDetect returns `[]`), so a single-take speech had no per-shot analysis. | `start_in_scene=True`: the whole video is one shot. |
| P6 | Medium | A video with **no audio track** crashed phase 0. | Detected; silent audio of the right length; `has_audio: false`; route warns, music analyses off. |
| P7 | Medium (money) | Budget check refused **free, fully cached** re-runs once past spend reached the cap; Groq counted cached chunks against the budget. | Zero-cost calls always allowed; only uncached chunks are estimated. |
| P8 | Medium | A Groq answer without `segments` (or a broken fal stem) could be cached and reused forever. | Validated before caching. |
| P9 | Medium | fal data-URI input is not documented for the queue API. | Upload with fal's official client (`fal-client`); data URI only as fallback. |
| P10 | Low (money) | Phase 2 paid a health-check call even when every shot already had an answer. | Skipped when there is nothing to do. |
| P11 | Low | Replicate output format is undocumented; the parser trusted the `.json` extension. | Tries every non-media output file; failure falls back to local sections **with a recorded reason and a QA warning**. |
| P12 | Low | Local section finder was off by one beat. | Column alignment fixed. |
| P13 | Low | "Uncertain language" warning fired when nothing was transcribed. | Only with real words. |
| P14 | Low | PySceneDetect 0.7 deprecations would break on upgrade. | Version-agnostic helpers. |
| P15 | — | Tests could reach real services if `.env` held keys. | Test fixture blanks every key for every test. |
| P16 | — | Web UI could not choose the free audio path. | "Free audio only" checkbox (`--offline`). |

## What cannot be verified without paying

- Exact response shapes of the live fal Demucs output, the Replicate Cog wrapper and Groq's `verbose_json` — handled tolerantly and failing loudly, but only real calls prove them.
- Whether Gemma 4 on OpenRouter answers within `max_tokens` in JSON mode (the breaker caps the loss at 5 calls if not).

## Recommended first paid sequence (≈ $0.20 total)

1. `python scripts/preflight.py --online --id <video_id>` — free.
2. `python scripts/calibration.py run --cloud` — the 36-s synthetic video through
   **all four** real services (OpenRouter vision ≈ $0.01, fal ≈ $0.03, Groq < $0.01,
   Replicate ≈ $0.11) with 20 checks against known answers. Any format surprise
   shows up here, on a cheap, known input.
3. Only then the real videos (`run_pipeline.py URL`) — ≈ $0.30 each.

## Known open (methodology, not bugs)

- Shot detection misses cuts on video 1 (QA flags 7); tempo trackers disagree
  4:3 on video 1 — both need the gold labels / a listening check.
- On Windows, if a results CSV is open in Excel the atomic replace fails and the
  phase stops; close the file and re-run (nothing is lost or re-paid).
