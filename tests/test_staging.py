"""Round-4 audit fixes F2/F3: video staging + run_pipeline id/path wiring."""
import json
import sys
from pathlib import Path

import pytest

from conftest import load_phase

from run_pipeline import derive_video_id, extract_youtube_id, resolve_staged_video, slugify


class TestVideoId:
    def test_youtube_watch_url(self):
        assert extract_youtube_id("https://www.youtube.com/watch?v=rtwpk9rb1Dc") == "rtwpk9rb1Dc"

    def test_youtu_be_and_shorts(self):
        assert extract_youtube_id("https://youtu.be/Z2ki180nHCI") == "Z2ki180nHCI"
        assert extract_youtube_id("https://www.youtube.com/shorts/abcdefghijk") == "abcdefghijk"

    def test_non_youtube_url_is_none(self):
        assert extract_youtube_id("https://example.com/video.mp4") is None

    def test_slugify_local(self):
        assert slugify("My Video (2024).mp4") == "my_video_2024"

    def test_local_source_derives_file_stem(self):
        assert derive_video_id("some/path/My Clip.mp4", None) == "my_clip"

    def test_explicit_id_wins(self):
        assert derive_video_id("anything", "custom") == "custom"


class TestResolveStagedVideo:
    def test_prefers_metadata_source_file(self, tmp_path):
        video = tmp_path / "vid1.mp4"
        video.write_bytes(b"fake")
        meta = {"source_file": str(video)}
        (tmp_path / "metadata.json").write_text(json.dumps(meta))
        got = resolve_staged_video("vid1", tmp_path)
        assert got == str(video.resolve())

    def test_falls_back_to_raw_glob(self, tmp_path, monkeypatch):
        import run_pipeline as rp
        raw = rp.REPO_ROOT / "data" / "raw"
        raw.mkdir(exist_ok=True)
        # metadata missing → fall back to data/raw/<id>.<ext> (webm, not mp4,
        # must still be found — audit F2's extension edge case)
        fake = raw / "vidX.webm"
        fake.write_bytes(b"fake")
        got = resolve_staged_video("vidX", tmp_path)
        assert got == str(fake.resolve())
        fake.unlink()

    def test_none_when_nothing_exists(self, tmp_path):
        assert resolve_staged_video("no_such_video_zz", tmp_path) is None


class TestPhase0Staging:
    @pytest.fixture()
    def phase0(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "data" / "vid1"))
        monkeypatch.setattr("sys.argv", ["phase0_input.py"])  # avoid arg parse issues in imports
        return load_phase("phase0_input", tmp_path / "data" / "vid1")

    def test_stage_local_copies_into_raw(self, phase0, tmp_path):
        raw = tmp_path / "data" / "raw"
        src = tmp_path / "elsewhere" / "my clip.mp4"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"fakevideo")
        staged = phase0.stage_local(str(src), "my_clip", raw)
        assert staged.exists()
        assert staged == raw / "my_clip.mp4"
        assert staged.read_bytes() == b"fakevideo"

    def test_stage_local_no_recopy_when_already_staged(self, phase0, tmp_path):
        raw = tmp_path / "data" / "raw"
        raw.mkdir(parents=True)
        src = raw / "vid1.mp4"
        src.write_bytes(b"fakevideo")
        staged = phase0.stage_local(str(src), "vid1", raw)
        assert staged == src  # returned as-is, no copy

    def test_resolve_downloaded_finds_webm(self, phase0, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        (raw / "vid1.webm").write_bytes(b"x")
        assert phase0.resolve_downloaded("vid1", raw) == raw / "vid1.webm"

    def test_resolve_downloaded_prefers_mp4(self, phase0, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        (raw / "vid1.mp4").write_bytes(b"x")
        (raw / "vid1.webm").write_bytes(b"yy")
        assert phase0.resolve_downloaded("vid1", raw) == raw / "vid1.mp4"

    def test_resolve_downloaded_exits_when_missing(self, phase0, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        with pytest.raises(SystemExit):
            phase0.resolve_downloaded("vid1", raw)

    def test_fraction_fps_not_eval(self, phase0, monkeypatch):
        # r_frame_rate "30000/1001" must be parsed with Fraction, not eval()
        ffprobe_json = json.dumps({
            "streams": [{"codec_type": "video", "codec_name": "h264",
                         "width": 516, "height": 360, "r_frame_rate": "30000/1001"}],
            "format": {"duration": "215.4", "size": "18259071", "bit_rate": "678182"},
        })

        class FakeOut:
            stdout = ffprobe_json

        monkeypatch.setattr("subprocess.run", lambda *a, **k: FakeOut())
        meta = phase0.extract_metadata(Path("fake.mp4"))
        assert abs(meta["video"]["fps"] - 29.97002997002997) < 1e-6