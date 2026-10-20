# Round 11 — cloud audio stack

Date: 2026-10-20 · Step 2 of the integration plan. Heavy models run on
hosted services; light ones stay local. Every hosted call is optional,
cached, costed and mocked in tests. 237 tests pass; offline calibration
18/18 checks.

## Where each part runs

```
audio_music.wav ──► separate_vocals.py ──► vocals.wav ──► phase 4 ──► transcript
  (48 kHz mix)       fal.ai Demucs           16 kHz        Groq whisper-large-v3-turbo
                     | local demucs          + vocal       | local faster-whisper
                     | passthrough (test)    activity      + vocal gate (stem silent ->
                     | off                   per second      segment flagged suspect)

audio_music.wav ──► phase 6 (librosa features) ──► music_structure.py ──► phase 7
                                                    beats + downbeats: Beat This! (local CPU)
                                                    sections: Replicate All-In-One (labelled)
                                                              | local novelty (A/B/C)
                                                    beats outside music removed
```

| Step | Default (`auto`) | Fallback | Est. cost, 3.6-min song |
|---|---|---|---|
| Vocal separation | fal.ai `fal-ai/demucs` (`htdemucs_ft`, vocals) if `FAL_KEY` and the route found music | `--separation local` (demucs on CPU), `off` | $0.15 |
| Transcription | Groq `whisper-large-v3-turbo` (verbose_json, word+segment timestamps, temperature 0) if `GROQ_API_KEY`; on the vocal stem when it exists | local faster-whisper | < $0.01 |
| Beats + downbeats | **Beat This!** locally (MIT, CPU, ~78 MB checkpoint once) | librosa (no downbeats) | 0 |
| Song sections | Replicate `sakemin/all-in-one-music-structure-analyzer` if `REPLICATE_API_TOKEN` (labels intro/verse/chorus…; version id and file-input field read from the model's schema at run time) | local novelty segmentation (Foote, snapped to downbeats, letters A/B/C) | $0.11 |

`run_pipeline.py --offline` forces the free path (separation off, local
transcription, local sections). `AUDIO_MAX_USD` (default 1.0) caps the
estimated spend per video; `data/<id>/cloud_usage.json` logs every call and
cache hit. The response cache (`data/cache/`) is keyed by audio content and
settings, so a re-run never pays twice.

## New analysis outputs

- `music_structure.json`: beats, downbeats, sections, sources.
- `sections.csv`: one row per section (cuts/min, mean shot length,
  energy, camera moving share, vocal activity, dominant emotion). Most shots
  are shorter than any audio window, so music questions belong at this level.
- `sync_stats.json`: `cut_downbeat_alignment` (chance = 2·tol / bar length)
  and `section_boundary_alignment` (are section boundaries edited as cuts?
  chance = share of the timeline near a cut). Both use exact tests.
- Per shot: `section`, `cut_on_downbeat`, `vocals_active_share`.
- QA: passthrough separation, missing downbeats, unlabelled sections, masked
  beats, and **beat-tracker disagreement** (librosa vs Beat This! differing
  by any ratio other than 1 or 2).

## Calibration (offline, `python scripts/calibration.py run`)

The synthetic song now has two sections (A: Am-F-C-G, 8th-note arpeggio;
B from 10 s: Dm-Bb-F-C, 16th-note bright lead, busier drums) and accented
downbeats. New checks, all passing:

| Check | Result |
|---|---|
| tempo | 120.0 BPM (Beat This!; librosa gave 117.5) |
| beats / downbeats | 40/40 and 10/10 within 70 ms |
| sections | one boundary found at 10.0 s, none extra |
| cut-on-downbeat | 78 % vs 10 % chance, p = 3e-6 (designed: 7/10) |
| separation wiring | stem aligned to 0 ms, used for transcription |

`calibration.py run --cloud` runs the same video through fal/Groq/Replicate
(~$0.15) and adds `separation_isolation`: the vocal stem must be at least
15 dB quieter than the mix in the instrumental part. **It has not been run:
no API keys are configured yet.**

## Real footage (video 1, offline)

Beat This! reports 96.8 BPM, librosa 129.2 BPM. That is a 4:3 ratio, not the
usual half/double ambiguity, so the QA now flags it: beat and downbeat
statistics for this video should not be trusted until someone checks the
tempo by ear. Local sections give the plausible form A B C B C D B C.

## Not verified yet (needs keys)

- fal: the data-URI upload of the MP3 mix and the queue status/response URLs
  follow fal's documented queue protocol but have only run against mocks.
- Replicate: the output format of this Cog wrapper is not documented; the
  parser accepts a dict, a nested dict, or a JSON file URL, and fails loudly
  otherwise.
- Groq: request format follows the published API; mocked only.
Run `python scripts/calibration.py run --cloud` once after adding keys —
it is the first real call for all three and checks the answers.
