"""Round 13: property-based tests — random and extreme inputs for every pure
function, checking invariants rather than single examples."""
import math
import os

import pytest

hyp = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from conftest import load_phase

SETTINGS = settings(max_examples=int(os.environ.get("HYP_EXAMPLES", "300")), deadline=None,
                    suppress_health_check=[HealthCheck.function_scoped_fixture])
times = st.floats(min_value=0, max_value=600, allow_nan=False, allow_infinity=False)
sorted_times = st.lists(times, min_size=0, max_size=60).map(sorted)


@pytest.fixture(scope="module")
def mods():
    return {n: load_phase(n) for n in ("_stats", "_normalize_emotion", "_content_profile",
                                       "_clap_tags", "_qa", "route_content", "music_structure",
                                       "phase3_camera", "phase1_shots", "phase4_transcribe",
                                       "phase2_vision", "phase5_audio", "phase7_sync", "calibration")}


# ---------------------------------------------------------------- statistics
@SETTINGS
@given(st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), max_size=40),
       st.integers(min_value=0, max_value=45))
def test_poisson_binomial_is_a_probability_and_monotone(mods, probs, k):
    sf = mods["_stats"].poisson_binomial_sf
    p = sf(probs, k)
    assert 0.0 <= p <= 1.0
    assert sf(probs, 0) == pytest.approx(1.0)
    assert sf(probs, k + 1) <= p + 1e-12


@SETTINGS
@given(sorted_times, sorted_times, st.floats(min_value=0.01, max_value=1.0))
def test_beat_alignment_invariants(mods, cuts, beats, tol):
    res = mods["_stats"].beat_alignment(cuts, beats, 600.0, tol=tol, n_perm=20)
    assert res["n_tested"] + res["n_outside_grid"] == len(cuts)
    for k in ("observed_rate", "chance_rate", "p_value"):
        v = res.get(k)
        assert v is None or (0.0 <= v <= 1.0 and not math.isnan(v))
    if res.get("phase_rayleigh") and res["phase_rayleigh"]["R"] is not None:
        assert 0.0 <= res["phase_rayleigh"]["R"] <= 1.0 + 1e-9


@SETTINGS
@given(sorted_times, sorted_times, st.floats(min_value=0.01, max_value=2.0))
def test_boundary_alignment_invariants(mods, bounds, cuts, tol):
    res = mods["_stats"].boundary_alignment(bounds, cuts, 600.0, tol=tol)
    for k in ("observed_rate", "chance_rate", "p_value"):
        v = res.get(k)
        assert v is None or 0.0 <= v <= 1.0


# ---------------------------------------------------------------- text parsing
@SETTINGS
@given(st.text(max_size=400))
def test_parse_json_response_never_raises(mods, text):
    p2 = mods["phase2_vision"]
    out = p2.parse_json_response(text)
    assert "_parse_error" in out or set(out) == {q[0] for q in p2.QUESTIONS}


@SETTINGS
@given(st.dictionaries(st.sampled_from(["caption", "camera", "emotion", "colors", "x"]),
                       st.one_of(st.text(max_size=30), st.integers(), st.lists(st.text(max_size=8)),
                                 st.none(), st.booleans()), max_size=5))
def test_parse_json_response_handles_any_value_types(mods, d):
    import json
    out = mods["phase2_vision"].parse_json_response(json.dumps(d))
    assert all(isinstance(v, str) for v in out.values())


@SETTINGS
@given(st.text(max_size=120))
def test_normalize_emotion_total(mods, text):
    ne = mods["_normalize_emotion"]
    out = ne.normalize_emotion(text)
    assert out in set(ne.list_canonical()) | {"other", ne.NO_PEOPLE}


# ---------------------------------------------------------------- timelines
@SETTINGS
@given(st.floats(min_value=0.0, max_value=20000.0, allow_nan=False),
       st.floats(min_value=30.0, max_value=900.0), st.floats(min_value=0.0, max_value=20.0))
def test_chunk_plan_covers_timeline_once(mods, dur, chunk, overlap):
    assume(overlap < chunk / 2)
    plan = mods["phase4_transcribe"].chunk_plan(dur, chunk=chunk, overlap=overlap)
    owned = [(o0, o1) for _, _, o0, o1 in plan]
    assert owned[0][0] == 0.0 and owned[-1][1] == pytest.approx(dur)
    assert all(abs(a[1] - b[0]) < 1e-6 for a, b in zip(owned, owned[1:]))
    assert all(c1 - c0 <= chunk + 1e-6 and c0 <= o0 + 1e-9 and o1 <= c1 + 1e-9
               for c0, c1, o0, o1 in plan)
    assert len(plan) < dur / max(1.0, chunk - overlap) + 3


@SETTINGS
@given(st.floats(min_value=0, max_value=300), st.floats(min_value=0.01, max_value=60),
       st.sampled_from([23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0]),
       st.floats(min_value=1.0, max_value=12.0))
def test_camera_samples_stay_inside_the_shot(mods, start, dur, fps, flow_fps):
    idx = mods["phase3_camera"].sample_frame_indices(start, start + dur, fps, flow_fps)
    assert idx == sorted(set(idx))
    assert all(start <= i / fps < start + dur for i in idx)


@SETTINGS
@given(st.floats(min_value=0, max_value=300), st.floats(min_value=0.0, max_value=30),
       st.integers(min_value=1, max_value=10))
def test_frame_range_and_mid_frame_bounds(mods, start, dur, fps):
    first, last = mods["phase3_camera"].shot_frame_range(start, start + dur, fps)
    assert first <= last and first >= 1
    n = int(math.ceil(310 * fps))
    mid = mods["phase1_shots"].mid_frame_index(start + dur / 2, fps, n)
    assert 1 <= mid <= n


# ---------------------------------------------------------------- audio helpers
@SETTINGS
@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), min_size=30, max_size=30))
def test_group_softmax_extreme_logits(mods, row):
    import numpy as np
    from _clap_tags import ALL_TAGS, TAG_GROUPS
    p = mods["phase5_audio"].group_softmax(np.array([row]), TAG_GROUPS, ALL_TAGS)
    assert np.all(np.isfinite(p)) and np.all(p >= 0)
    for names in TAG_GROUPS.values():
        assert sum(p[0, ALL_TAGS.index(t)] for t in names) == pytest.approx(1.0)


@SETTINGS
@given(st.lists(st.tuples(st.floats(0, 1), st.floats(0, 1), st.floats(0, 1),
                          st.floats(-120, 0)), max_size=30))
def test_music_regions_disjoint_sorted(mods, vals):
    rows = [{"start_sec": str(5 * i), "end_sec": str(5 * i + 5), "music": str(m), "speech": str(s),
             "silence": str(q), "rms_dbfs": str(db)} for i, (m, s, q, db) in enumerate(vals)]
    reg = mods["music_structure"].music_regions(rows)
    assert all(a < b for a, b in reg)
    assert all(r1[1] < r2[0] for r1, r2 in zip(reg, reg[1:]))


@SETTINGS
@given(st.dictionaries(st.sampled_from(["music", "speech", "silence"]),
                       st.floats(min_value=0, max_value=1), min_size=3, max_size=3),
       st.sampled_from(["auto", "song", "speech", "film", "vlog", "other", "series"]))
def test_route_decision_total(mods, shares, ctype):
    r = mods["route_content"].decide(shares, ctype)
    assert r["audio_class"] in {"music", "speech", "mixed", "silent"}
    assert isinstance(r["has_music"], bool)
    assert r["modules"]["transcription"] is True


# ---------------------------------------------------------------- join
shot_lists = st.lists(st.floats(min_value=0.2, max_value=10.0), min_size=1, max_size=15).map(
    lambda ds: [{"shot_idx": i, "start_sec": round(sum(ds[:i]), 3), "end_sec": round(sum(ds[:i + 1]), 3)}
                for i in range(len(ds))])


@SETTINGS
@given(shot_lists, st.lists(st.floats(min_value=0, max_value=1), max_size=30), sorted_times,
       st.one_of(st.none(), st.booleans()))
def test_join_data_total_and_bounded(mods, shots, moods, beats, music_present):
    clap = [{"start_sec": str(5 * i), "end_sec": str(5 * i + 5), "happy and bright": str(m),
             "music": str(m), "speech": str(1 - m), "silence": "0", "rms_dbfs": "-20"}
            for i, m in enumerate(moods)]
    rows, stats = mods["phase7_sync"].join_data(shots, [], [], [], clap, [], beats, ["happy and bright"],
                                                music_present=music_present)
    assert len(rows) == len(shots)
    for r in rows:
        assert 0.0 <= r["audio_mood_happy and bright"] <= 1.0 + 1e-9
        assert isinstance(r["valid_audio_mood"], bool)
    for k, v in stats["coverage"].items():
        assert v["pct"] is None or 0 <= v["pct"] <= 100


@SETTINGS
@given(shot_lists, sorted_times, st.text(max_size=60))
def test_content_profile_total(mods, shots, beats, text):
    for s in shots:
        s["duration_sec"] = s["end_sec"] - s["start_sec"]
        s["camera_motion"] = "static"
    prof = mods["_content_profile"].detect_content_type(
        shots=shots, music_summary={"tempo_bpm": 120, "n_beats": len(beats)},
        transcript=[{"start_sec": 0, "end_sec": 5, "text": text}], video_duration=0)
    assert prof["type"] in {"song", "speech", "film", "vlog", "other"}


@SETTINGS
@given(sorted_times, sorted_times, st.floats(min_value=0.01, max_value=1))
def test_match_events_bounds(mods, pred, truth, tol):
    m = mods["calibration"].match_events(pred, truth, tol)
    assert m["tp"] <= min(len(pred), len(truth))
    assert 0 <= m["precision"] <= 1 and 0 <= m["recall"] <= 1 and 0 <= m["f1"] <= 1
