"""Fast unit tests for the pure functions (no ffmpeg, no models)."""
from __future__ import annotations
import json

import numpy as np
import pytest

import common
import crossmodal
from _normalize_emotion import list_canonical, normalize_emotion


# --- frame timing (the round-3 critical bug) --------------------------------

@pytest.mark.parametrize("t, fps, expected", [
    (0.0, 2, 1), (0.24, 2, 1), (0.26, 2, 2), (10.0, 2, 21), (206.1, 2, 413), (5.0, 1, 6),
])
def test_frame_for_time(t, fps, expected):
    assert common.frame_for_time(t, fps) == expected


def test_frame_for_time_clamps_to_extracted_frames():
    assert common.frame_for_time(100.0, 2, n_extracted=50) == 50


def test_frames_in_range_is_half_open():
    # frames at t = 0, .5, 1.0, ... ; shot [5, 10) holds t = 5.0 .. 9.5
    r = common.frames_in_range(5.0, 10.0, 2)
    assert list(r) == list(range(11, 21))
    assert len(common.frames_in_range(0.0, 2.5, 2)) == 5


def test_fingerprint_and_up_to_date(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "PROCESSED", tmp_path)
    monkeypatch.setattr(common, "RUN_INFO", tmp_path / "run_info.json")
    src, out = tmp_path / "in.txt", tmp_path / "out.txt"
    src.write_text("a")
    out.write_text("b")
    common.record_run(3, inputs=[src], outputs=[out], argv=["--x", "1"])
    assert common.is_up_to_date(3, ["--x", "1"], tmp_path)[0]
    assert common.is_up_to_date(3, ["--x", "2"], tmp_path) == (False, "arguments changed")
    src.write_text("changed!")
    fresh, why = common.is_up_to_date(3, ["--x", "1"], tmp_path)
    assert not fresh and "input changed" in why
    assert common.is_up_to_date(4, [], tmp_path) == (False, "never ran")
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert "git_commit" in info["phase3"]


def test_code_change_invalidates_run(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "PROCESSED", tmp_path)
    monkeypatch.setattr(common, "RUN_INFO", tmp_path / "run_info.json")
    common.record_run(5, inputs=[], outputs=[], argv=[])
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert "scripts/common.py" in info["phase5"]["code"]
    assert common.is_up_to_date(5, [], tmp_path)[0]
    info["phase5"]["code"]["scripts/common.py"] = "0" * 40  # as if common.py had been edited
    (tmp_path / "run_info.json").write_text(json.dumps(info))
    assert common.is_up_to_date(5, [], tmp_path) == (False, "code changed: scripts/common.py")


# --- cross-modal statistics -------------------------------------------------

def test_nearest_distance():
    d = crossmodal.nearest_distance([0.1, 0.9, 5.0], [0.0, 1.0])
    assert np.allclose(d, [0.1, 0.1, 4.0])
    assert np.isinf(crossmodal.nearest_distance([1.0], [])).all()


def test_beat_window_coverage():
    assert crossmodal.beat_window_coverage([1, 2, 3], 0.1, 0, 10) == pytest.approx(0.06)
    assert crossmodal.beat_window_coverage([1, 1.15], 0.1, 0, 10) == pytest.approx(0.035)  # overlap merged
    assert crossmodal.beat_window_coverage([0], 0.1, 0, 1) == pytest.approx(0.1)  # clipped at t0


def test_cut_on_beat_detects_alignment():
    beats = np.arange(0, 200, 0.5)
    rng = np.random.default_rng(1)
    cuts = np.sort(rng.choice(beats[1:], 60, replace=False)) + rng.normal(0, 0.02, 60)
    res = crossmodal.cut_on_beat_test(cuts, beats, 0.1, 200, n_perm=2000)
    assert res["observed_pct"] > 90 and res["p_value"] < 0.01
    assert res["chance_pct"] == pytest.approx(40, abs=1)


def test_cut_on_beat_is_calibrated_on_random_cuts():
    beats = np.arange(0, 200, 0.5)
    ps = [crossmodal.cut_on_beat_test(np.random.default_rng(k).uniform(1, 199, 60), beats, 0.1, 200,
                                      n_perm=500, seed=k)["p_value"] for k in range(100)]
    assert np.mean(np.array(ps) < 0.05) <= 0.1


def test_cut_on_beat_without_beats():
    assert crossmodal.cut_on_beat_test([1.0, 2.0], [], 0.1, 10)["p_value"] is None


def test_rankdata_matches_average_ranks():
    assert list(crossmodal.rankdata([10, 20, 20, 5])) == [2.0, 3.5, 3.5, 1.0]


def test_spearman_and_ci():
    rng = np.random.default_rng(0)
    x = rng.normal(size=80)
    strong = crossmodal.correlate("x", x, "y", x + 0.5 * rng.normal(size=80))
    assert strong["rho"] > 0.7 and strong["significant"]
    null = crossmodal.correlate("x", x, "z", rng.normal(size=80))
    assert not null["significant"] and null["ci_low"] < 0 < null["ci_high"]
    assert crossmodal.correlate("x", [1, 2], "y", [2, 1])["rho"] is None  # too few points


def test_correlate_skips_missing():
    res = crossmodal.correlate("x", [1, 2, np.nan, 4, 5, 6, 7], "y", [1, 2, 3, np.nan, 5, 6, 7])
    assert res["n"] == 5 and res["rho"] == pytest.approx(1.0)


def test_overlap_mean_weights_by_overlap():
    recs = [{"start_sec": 0, "end_sec": 1, "v": 0}, {"start_sec": 1, "end_sec": 2, "v": 10}]
    assert crossmodal.overlap_mean(recs, "v", 0.5, 2.0) == pytest.approx(10 * 1 / 1.5)
    assert np.isnan(crossmodal.overlap_mean(recs, "v", 5, 6))


# --- emotion normalisation ---------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("sensual intensity", "sensual"), ("Sultry", "sensual"), ("intense or fierce", "intense"),
    ("", "other"), ("joyful", "joyful"), ("hopeful", "contemplative"),
    ("Based on the body language and downcast expressions of the", "other"),
])
def test_normalize_emotion(raw, expected):
    assert normalize_emotion(raw) == expected


def test_canonical_labels_map_to_themselves():
    for label in list_canonical():
        assert normalize_emotion(label) == label


def test_synonyms_are_unambiguous():
    from _normalize_emotion import CANONICAL_EMOTIONS
    seen = {}
    for canonical, syns in CANONICAL_EMOTIONS.items():
        for s in syns:
            assert seen.setdefault(s, canonical) == canonical, f"{s!r} listed under {seen[s]} and {canonical}"


# --- phase 2 parsing (no network) -------------------------------------------

def test_parse_json_response_variants():
    import phase2_vision as p2
    good = json.dumps({"caption": "A man.", "camera": "static", "emotion": "neutral", "colors": ["red", "blue"]})
    assert p2.parse_json_response(good)["colors"] == "red, blue"
    fenced = "```json\n" + good + "\n```"
    assert p2.parse_json_response(fenced)["caption"] == "A man."
    assert p2.parse_json_response("Sure! " + good + " Hope this helps")["camera"] == "static"
    assert "_parse_error" in p2.parse_json_response("not json")
    assert "_parse_error" in p2.parse_json_response("[1, 2]")


def test_clean_answers_maps_vocabularies():
    import phase2_vision as p2
    out = p2.clean_answers({"emotion": "Sultry", "camera": "pan-left"})
    assert out == {"emotion": "sensual", "camera": "pan"}
    assert p2.clean_answers({"camera": "dolly"})["camera"] == "other"


def test_schema_enums_match_vocabularies():
    import phase2_vision as p2
    assert p2.RESPONSE_SCHEMA["properties"]["emotion"]["enum"] == list_canonical()
    assert p2.RESPONSE_SCHEMA["properties"]["camera"]["enum"] == common.CAMERA_LABELS


# --- phase 5 / 6 / pipeline helpers ------------------------------------------

def test_group_softmax_normalizes_each_group():
    import phase5_audio as p5
    logits = np.random.default_rng(0).normal(size=(4, len(common.ALL_TAGS))) * 10
    probs = p5.group_softmax(logits, common.TAG_GROUPS, common.ALL_TAGS)
    for tags in common.TAG_GROUPS.values():
        idx = [common.ALL_TAGS.index(t) for t in tags]
        assert np.allclose(probs[:, idx].sum(axis=1), 1.0)
    # A huge instrument logit must not change the mood probabilities
    bumped = logits.copy()
    bumped[:, common.ALL_TAGS.index("piano")] += 100
    p2 = p5.group_softmax(bumped, common.TAG_GROUPS, common.ALL_TAGS)
    mood = [common.ALL_TAGS.index(t) for t in common.MOOD_TAGS]
    assert np.allclose(probs[:, mood], p2[:, mood])


def _chroma(chords):
    c = np.zeros(12)
    for notes, weight in chords:
        for m in notes:
            c[m % 12] += weight
    return c


@pytest.mark.parametrize("chords, key", [
    ([([60, 64, 67], 2), ([65, 69, 72], 1), ([67, 71, 74], 1), ([60, 64, 67], 2)], "C major"),
    ([([57, 60, 64], 2), ([62, 65, 69], 1), ([64, 68, 71], 1), ([57, 60, 64], 2)], "A minor"),
    ([([62, 66, 69], 2), ([67, 71, 74], 1), ([69, 73, 76], 1), ([62, 66, 69], 2)], "D major"),
])
def test_estimate_key(chords, key):
    import phase6_music as p6
    assert p6.estimate_key(_chroma(chords))[0] == key


@pytest.mark.parametrize("url, vid", [
    ("https://www.youtube.com/watch?v=rtwpk9rb1Dc", "rtwpk9rb1Dc"),
    ("https://youtu.be/rtwpk9rb1Dc?t=3", "rtwpk9rb1Dc"),
    ("https://www.youtube.com/shorts/rtwpk9rb1Dc", "rtwpk9rb1Dc"),
    ("https://example.com/video.mp4", None),
])
def test_extract_youtube_id(url, vid):
    import run_pipeline
    assert run_pipeline.extract_youtube_id(url) == vid


def test_derive_video_id_rejects_unsafe_ids():
    import run_pipeline
    assert run_pipeline.derive_video_id("/x/My Clip (1).mp4", None) == "my_clip_1"
    with pytest.raises(SystemExit):
        run_pipeline.derive_video_id("x.mp4", "../escape")


# --- validation metrics -------------------------------------------------------

def test_cohen_kappa_and_wilson():
    import validate_labels as vl
    assert vl.cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert vl.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
    lo, hi = vl.wilson(8, 10)
    assert 0.44 < lo < 0.5 and 0.94 < hi < 0.97  # textbook values: 0.490, 0.943


def test_compare_excludes_unsure_and_maps_families():
    import validate_labels as vl
    human = {0: "pan-left", 1: "unsure", 2: "static", 3: "zoom-in"}
    pred = {0: "pan-right", 1: "static", 2: "static", 3: "zoom-in"}
    exact = vl.compare("x", human, pred)
    family = vl.compare("x", human, pred, vl.camera_family)
    assert exact["n"] == 3 and exact["agree"] == 2
    assert family["agree"] == 3


def test_no_significance_claims_on_tiny_samples():
    rng = np.random.default_rng(0)
    x = rng.normal(size=8)
    res = crossmodal.correlate("x", x, "y", x + 0.1 * rng.normal(size=8))
    assert res["rho"] is not None and res["ci_low"] is None and not res["significant"]
