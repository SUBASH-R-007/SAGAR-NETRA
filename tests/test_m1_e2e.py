"""M1 acceptance: the bundled sample XTF parses and exports a waterfall PNG
inside the time budget, and truth JSON is complete."""

from __future__ import annotations

import json
import time

from PIL import Image

from sonar_core.parsers.base import load
from sonar_core.waterfall import save_waterfall_png


def test_sample_xtf_to_waterfall_under_budget(sample_xtf, tmp_path) -> None:
    start = time.perf_counter()
    pa = load(sample_xtf)
    png = save_waterfall_png(pa, tmp_path / "waterfall.png")
    elapsed = time.perf_counter() - start

    assert elapsed < 30.0, f"parse+export took {elapsed:.1f}s (budget 30s)"
    assert pa.n_pings == 600
    assert pa.meta["format"] == "xtf"
    # Navigation is present and sane for every ping.
    assert (pa.nav["altitude"] > 0).all()
    assert (abs(pa.nav["lat"] - 13.05) < 0.1).all()
    with Image.open(png) as im:
        assert im.size[1] == pa.n_pings

    truth = json.loads((sample_xtf.parent / "survey_alpha.truth.json").read_text())
    assert truth["scene"]["slant_range_m"] == 50.0
    assert len(truth["targets"]) >= 6
    assert all({"cls", "side", "ping", "height"} <= t.keys() for t in truth["targets"])
