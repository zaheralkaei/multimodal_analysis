"""Round-4 audit F7: _renormalize_existing honors PROCESSED_DIR."""
import json
import sys

import pytest

from conftest import load_phase


def test_renormalize_uses_processed_dir(tmp_processed):
    renorm = load_phase("_renormalize_existing", tmp_processed, module_name="_renormalize_existing")
    (tmp_processed / "shot_vision.csv").write_text(
        "shot_idx,emotion\n"
        "0,sensual intensity\n"
        "1,Joyful!\n"
        "2,totally made-up word\n",
        encoding="utf-8")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = renorm.main()
    assert rc == 0
    out = tmp_processed / "shot_vision_normalized.csv"
    assert out.exists()  # F7 regression: old code looked in data/processed/
    rows = out.read_text(encoding="utf-8").strip().splitlines()
    assert rows[1].endswith(",sensual")
    assert rows[2].endswith(",joyful")
    assert rows[3].endswith(",other")