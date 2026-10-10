"""Round-5 UI audit fixes U1-U4: video identity, timestamp deep-links,
surfaced vision fields, per-shot table filters."""
import json
import os
from pathlib import Path

import pytest

from conftest import load_phase

YT_ID = "rtwpk9rb1Dc"


def _run_phase8(tmp_path, video_id, sync_header, sync_row, metadata=None):
    proc = tmp_path / "data" / video_id
    proc.mkdir(parents=True)
    reports = tmp_path / "reports" / video_id
    (proc / "sync_per_shot.csv").write_text(sync_header + sync_row, encoding="utf-8")
    (proc / "sync_stats.json").write_text(json.dumps({"vision_model": "google/gemma-4-31b-it"}))
    if metadata is not None:
        (proc / "metadata.json").write_text(json.dumps(metadata))
    os.environ["PROCESSED_DIR"] = str(proc)
    os.environ["REPORTS_DIR"] = str(reports)
    mod = load_phase("phase8_dashboard")
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        rc = mod.main()
    assert rc == 0
    return (reports / "dashboard.html").read_text(encoding="utf-8")


FULL_HEADER = ("shot_idx,start_sec,end_sec,duration_sec,vision_caption,vision_emotion,"
               "camera_motion,camera_pan_score,camera_tilt_score,camera_zoom_score,"
               "vision_colors,vision_location,vision_lighting,vision_composition,"
               "vision_entities,audio_top_mood,lyric_text,mid_frame,vision_camera_from_vlm,cut_on_beat\n")
FULL_ROW = ('0,10.0,13.0,3.0,cap text,sensual,pan-left,-0.62,0.12,0.03,'
            '"yellow, black",indoor room,bright,closeup,'
            'woman,happy and bright,la la,frame_00001.jpg,static,True\n')


class TestIdentityAndLinks:
    def test_video_id_in_title_and_source_links(self, tmp_path, monkeypatch):
        meta = {"source_url": f"https://www.youtube.com/watch?v={YT_ID}",
                "frame_fps": 2, "frames_extracted": 400,
                "video": {"fps": 24.0}}
        monkeypatch.setattr("sys.argv", ["phase8_dashboard.py"])
        html = _run_phase8(tmp_path, YT_ID, FULL_HEADER, FULL_ROW, meta)
        # U1: identity
        assert f"<title>{YT_ID} — Multimodal Video Analysis</title>" in html
        assert f"<h1>{YT_ID} — multimodal video analysis</h1>" in html
        assert "generated" in html
        # U2: deep links from recorded source_url, stamped with the shot's start
        assert f"https://www.youtube.com/watch?v={YT_ID}&t=10" in html

    def test_youtube_id_fallback_without_metadata(self, tmp_path, monkeypatch):
        # pre-round-5 metadata has no source_url: dir name shaped like a YT id
        monkeypatch.setattr("sys.argv", ["phase8_dashboard.py"])
        html = _run_phase8(tmp_path, YT_ID, FULL_HEADER, FULL_ROW)
        assert f"https://youtu.be/{YT_ID}?t=10" in html

    def test_local_video_gets_no_bogus_link(self, tmp_path, monkeypatch):
        # a local-file id ("my_clip") must NOT produce a youtu.be URL
        monkeypatch.setattr("sys.argv", ["phase8_dashboard.py"])
        html = _run_phase8(tmp_path, "my_clip", FULL_HEADER, FULL_ROW)
        assert "youtu.be" not in html
        assert "youtube.com" not in html
        assert "<title>my_clip — Multimodal Video Analysis</title>" in html


class TestTableSurfacesAnalysis:
    def test_new_columns_scores_and_badge(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["phase8_dashboard.py"])
        html = _run_phase8(tmp_path, YT_ID, FULL_HEADER, FULL_ROW,
                           {"frame_fps": 2, "video": {"fps": 24.0}})
        # U3: surfaced columns
        for col in ("Colors", "Location", "Lighting", "Composition", "Entities", "Beat"):
            assert f">{col}</th>" in html
        # continuous camera scores exposed as tooltip
        assert "pan -0.62" in html
        assert "tilt +0.12" in html
        assert "zoom +0.03" in html
        # cut-on-beat badge
        assert "&#9834;" in html or "♩" in html
        assert 'data-beat="1"' in html or "data-beat='1'" in html

    def test_no_cut_on_beat_shows_empty(self, tmp_path, monkeypatch):
        row = FULL_ROW.replace(",True\n", ",False\n")
        monkeypatch.setattr("sys.argv", ["phase8_dashboard.py"])
        html = _run_phase8(tmp_path, YT_ID, FULL_HEADER, row)
        assert 'data-beat="0"' in html or "data-beat='0'" in html


class TestTableControls:
    def test_filter_controls_and_js_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["phase8_dashboard.py"])
        html = _run_phase8(tmp_path, YT_ID, FULL_HEADER, FULL_ROW)
        for ctl_id in ("tblSearch", "tblEmotion", "tblCamera", "tblBeat", "tblCount"):
            assert ctl_id in html
        assert "<script>" in html and "shotTable" in html
        # dropdowns populated from data
        assert "<option value='sensual'>sensual</option>" in html
        assert "<option value='pan-left'>pan-left</option>" in html
        # findings link to the table anchor (both quote styles accepted)
        assert 'href="#shots-table"' in html or "href='#shots-table'" in html
        assert 'id="shots-table"' in html

    def test_minimal_csv_still_renders(self, tmp_path, monkeypatch):
        # old partial CSV without the new columns → dashes, no crash
        header = "shot_idx,start_sec,end_sec,duration_sec,vision_emotion,vision_caption,camera_motion,audio_top_mood,lyric_text,mid_frame\n"
        row = "0,0,5,5,sensual,cap,static,happy,b,frame_00001.jpg\n"
        monkeypatch.setattr("sys.argv", ["phase8_dashboard.py"])
        html = _run_phase8(tmp_path, "vid1", header, row)
        assert "—" in html  # missing columns render as an em-dash, not a crash
        assert 'data-beat="0"' in html or "data-beat='0'" in html