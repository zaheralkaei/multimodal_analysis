# Audit round 13 — systematic audit (before any paid run)

Date: 2026-10-20. Previous rounds reviewed code by reading it and running
it. This round added methods that do not depend on the reviewer noticing
something:

| Method | Scope | Result |
|---|---|---|
| **Coverage measurement** (coverage.py) | every script | 68 % → the untested 32 % (phase 6 had 0 %, phase 5 29 %, phase 1 45 %…) was read line by line |
| **Producer/consumer key audit** (script) | every `.get("k")` / `["k"]` in the consumers vs. every column/key in real outputs | no mismatched names between phases |
| **Property-based fuzzing** (Hypothesis, 15 properties × 3 000 random/extreme inputs = 45 000 cases) | statistics, timeline maths, text parsing, joins, routing, audio helpers | **1 crash found** (F4), then all pass |
| **CLI flag matrix** (13 full runs) | skip-phase2/5, fps 1/3, content-type speech/song, librosa beats + no sections, strict shot settings, passthrough separation, all skips, resume from phases 4/7/8 | all complete; outputs semantically checked |
| **Strict JSON scan** | every JSON written by every run | no NaN/Infinity, all parseable |
| **Determinism** | full pipeline run twice | 24/24 output files byte-identical |
| **Dependency audit** | every third-party import vs. requirements.txt | `demucs` documented as optional |
| **Escaping audit** | every data value inserted into dashboard HTML | all escaped |
| **Swallowed-exception audit** | every broad `except` | all intentional fallbacks |

## Bugs found and fixed (each has a regression test)

| # | Severity | Finding | Fix |
|---|---|---|---|
| F1 | Medium (data) | Silent or flat audio gave key **"C major" with correlation `-Infinity`** (invalid JSON, false result) | No key when chroma has no variance; JSON written with `allow_nan=False` |
| F2 | Medium (crash) | Phase 3 kept **every sampled frame of the whole video in memory**: ~3 GB for a 2-hour film | Each shot is estimated as soon as its last frame arrives and its frames are freed |
| F3 | Low (method) | Silence threshold −45 dBFS could class quiet acoustic passages as silence | −50 dBFS (true silence ≪ −60) |
| F4 | Medium (money) | Model answer that is valid JSON but **not an object** (`0`, a list) **crashed** the parser → counted as an API error instead of a parse error (found by fuzzing) | Non-objects are parse errors |
| F5 | **High (money)** | A **paid job submission (fal, Replicate) was retried after a timeout or 500/502/504** — the job may already have been accepted, so the retry could pay twice; a server `Retry-After` could block for an hour | Paid POSTs retried only when the request certainly did not arrive (connection failure, 429); Groq (cheap) and uploads (free) opt in to retries; waits capped at 120 s |
| F6 | **High (data)** | Lineage stored **absolute paths**: moving/cloning the project made every step "stale" and phase 7 refused to join | Paths stored relative to the video folder or the repository |
| F7 | **High (data)** | **Deleting the raw video** (to save disk) was treated as "input changed" → every stream refused | A missing input is reported as info, not as stale |
| F8 | Medium (data) | Regenerating a dashboard after deleting the extracted frames **deleted every thumbnail** of the report | Referenced thumbnails are kept even when their source frame is gone |
| F9 | Low | `--content-type speech` on 57 %-music audio gave no warning (override warning used 70 %, detection 30 %) | Same 30 % threshold |
| — | — | Phase 6 had no unit test; unused variables; key shown as "None" | tests added; cleaned |

## State after this round

- 282 tests pass (1 opt-in end-to-end calibration skipped), plus 45 000 fuzz cases.
- Offline calibration and fake-cloud calibration pass from scratch.
- Every paid code path has run against the fake cloud, including failure modes.

## Still unverifiable without paying

The response formats of the live services — see `AUDIT_R12_PRESPEND.md`.
The first paid step remains `python scripts/calibration.py run --cloud`
(≈ $0.17): a known input with 20 checks, so any surprise shows up there and
not on a real video.

## Known, not bugs

- Reports (`run_manifest.json`, `lineage.json`) contain local file paths
  including the Windows user name, and are part of the ZIP download.
  Remove them before publishing supplementary material.
- Running the CLI and the web UI on the same video at the same time is not
  guarded; run one at a time.
- Shot detection misses cuts and two beat trackers disagree on both test videos — methodology, settled by the gold labels.
