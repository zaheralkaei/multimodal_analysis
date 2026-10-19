# Audit round 9 — general code audit

Date: 2026-10-19 · Scope: every script in `scripts/`, the web UI, tests,
reproducibility. Follows round 8 (`docs/PLAN_ROUND_8.md`, commit f5cfce6).
Result: 11 issues fixed, each with a regression test; 185 tests pass (was 159).
Two identical runs of phases 3–8 now produce byte-identical outputs (18 files).

## Fixed

| # | Severity | Where | Finding | Evidence / fix |
|---|---|---|---|---|
| R9-1 | **Critical** | `run_pipeline.py` | **Command injection.** Phase commands were f-strings run with `shell=True`; a URL containing `"` broke out of the quoting (`https://x/a" & <command> & "` → `<command>` runs). Reachable from the web UI, which only checked `startswith("http")`. | Reproduced with an intercepted `subprocess.run` (nothing executed). Now argv lists, no shell, `cwd=` instead of `cd &&`. Test: hostile URL arrives as one argv element; `shell=True` fails the test. |
| R9-2 | **Critical** | `app.py` | **Cross-site requests.** Binding to 127.0.0.1 does not stop a web page the user has open from POSTing a form to `http://127.0.0.1:5000/start` (a multipart POST needs no CORS preflight) → downloads, paid API calls, and with R9-1 code execution. Also open to DNS rebinding. | `before_request` refuses non-loopback `Host` headers and state-changing requests whose `Origin` / `Sec-Fetch-Site` is another site; URLs must match `https?://` without spaces/quotes. 5 tests. |
| R9-3 | High | `_content_profile.py` | **Speech was always classified as "song".** librosa reports a tempo for any audio (a 2-min TTS speech: 112 BPM, 250 "beats"), so `has_music` was always true and the speech/vlog rules never fired. | New CLAP group `music / speech / silence` (song: 100 % music windows, speech: 0 %, Z2ki180nHCI: 77 %). Music presence needs ≥ 30 % music windows; without phase 5 the dashboard says presence is unverified. README no longer advises `--skip-phase5` for speeches. 4 tests. |
| R9-4 | High | `phase5_audio.py` | **CLAP not reproducible.** For the fused model, transformers randomly flags one clip per batch as "longer" and routes it through the fusion branch; identical runs differed by up to 0.53 in a tag probability. | `is_longer` forced to False (all windows are 5 s < 10 s). Run-to-run diff 0.0, batch-size independent (≤ 3e-6). Test with a fake processor. |
| R9-5 | High | `_normalize_emotion.py` | **Substring matching mislabelled emotions:** "courageous", "dangerous", "stranger" → angry (via "rage"/"anger"), "funeral" → playful ("fun"). | Whole-word matching; answers > 6 words → "other" (a rationale is not a label). 10 parametrized tests. |
| R9-6 | Medium | `_normalize_emotion.py`, phase 2 | Shots **without people** were counted as "neutral" emotion, inflating "neutral". | Separate label `no people`; the prompt asks for it; own grey color. |
| R9-7 | Medium | `phase2_vision.py` | Only the normalized emotion was stored — the model's own answer was lost, so the mapping could not be audited. | New `emotion_raw` column. Test. |
| R9-8 | Medium | `phase7_sync.py` | `music_n_beats` summed the per-second beat counts of every second touching the shot — a 0.7 s shot could be credited with 2 s of beats. | Counts beat times inside `[start, end)`. Test. |
| R9-9 | Medium | `phase0_input.py` | A YouTube link copied from a playlist (`&list=…`) downloaded **the whole playlist**; fixed 300 s ffmpeg / 600 s download timeouts kill long films. | `--no-playlist`; ffmpeg timeout = max(300 s, 2 × duration); download 3600 s. 2 tests. |
| R9-10 | Low | `phase8_dashboard.py` | Stale thumbnails: copy skipped when only the *size* matched; unreferenced old frames shipped in the ZIP; filter `data-*` attributes double-escaped (filter mismatch for values with `&`, quotes); lyric track showed hallucination-flagged segments like real ones; CLAP axis still said "similarity". | Size + mtime check, stale frames removed, single escaping, flagged lyrics grey with a label, axis "CLAP prob. (among mood tags)". |
| R9-11 | Low | `phase1_shots.py`, requirements | Boundary frame in the debug CSV used `int()` on a ms-rounded time (off by one); `imageio-ffmpeg` fallback not in requirements. | `round()`; requirement added. |

## Verified OK

- Camera (phase 3), transcript (phase 4), music (phase 6), join (phase 7)
  outputs are deterministic on CPU (byte-identical across runs; OpenCV's
  RANSAC uses a fixed RNG seed; Whisper's temperature fallback did not
  trigger nondeterminism on these videos).
- ZIP download contains only dashboard, thumbnails and CSV/JSON — no audio,
  no `.env`.
- Phase 2 sends exactly the documented request; Gemma 4 on OpenRouter
  supports `temperature`, `seed`, `response_format`, `max_tokens`.

## Open (not fixed — methodology or low impact)

| Item | Note |
|---|---|
| Thresholds (shots, camera) uncalibrated | Needs the gold set (plan W0). `min_scene_len` 15 vs 30 is the main open choice. |
| Variable-frame-rate video | Phases 1 and 3 map time ↔ frame index with a constant fps (true for YouTube MP4s; screen recordings/phone video can be VFR). |
| Per-shot audio mood | CLAP windows are 5 s, median shot ~1.5 s — per-shot mood is the surrounding 5 s, not the shot. |
| Key detection | Krumhansl on whole-song `chroma_stft` incl. drums; no per-section key. Plan W3. |
| Whisper temperature fallback | Can sample (non-deterministic) on hard segments; set `temperature=0` only if strict reproducibility outweighs recall. |
| `scripts/_test_*.py` | Old Ollama/Gemini experiments referenced by docs; not part of the pipeline or test suite. |
