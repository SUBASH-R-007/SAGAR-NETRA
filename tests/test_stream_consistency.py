"""Streaming must see the same seabed the batch path sees.

A window is far shorter than the survey, so any preprocessing stage whose
statistics or geometry depend on image extent will normalise it differently —
and the anomaly autoencoder, whose detection threshold is calibrated once at
training time, then fires on the mismatch rather than on debris. This module
pins the invariant that caused a real regression: CLAHE's tile grid must span
a comparable number of pings in a window as it does in a whole survey.
"""

from __future__ import annotations

import pytest

from api.realtime import _window_preprocess_config
from sonar_core.preprocess.pipeline import DEFAULTS, preprocess
from sonar_core.synth.scene import SceneConfig, SynthTarget, make_scene

WINDOW_PINGS = 200
TOTAL_PINGS = 600


@pytest.fixture(scope="module")
def survey():
    cfg = SceneConfig(
        n_pings=TOTAL_PINGS, n_samples=1024, slant_range=45.0, altitude=9.0,
        start_lat=15.50, start_lon=80.90, seed=4242,
    )
    targets = [
        SynthTarget("container", "starboard", 120, 22.0, 6.0, 2.4, height=2.4,
                    reflectivity=6.0),
        SynthTarget("ghost_net", "port", 300, 17.0, 6.5, 3.5, height=1.4,
                    reflectivity=3.2, shape="irregular"),
    ]
    pa, _ = make_scene(cfg, targets)
    return pa


def test_window_config_scales_clahe_rows_not_columns() -> None:
    cols, rows = DEFAULTS["clahe"]["tile_grid"]
    cfg = _window_preprocess_config(None, WINDOW_PINGS, TOTAL_PINGS)
    grid_cols, grid_rows = cfg["clahe"]["tile_grid"]
    assert grid_cols == cols, "swath width does not change window to window"
    assert grid_rows == max(1, round(rows * WINDOW_PINGS / TOTAL_PINGS))
    assert grid_rows < rows, "a short window must use fewer CLAHE row cells"


def test_window_config_preserves_caller_overrides() -> None:
    base = {"despeckle": {"enabled": False}, "clahe": {"clip_limit": 4.0}}
    cfg = _window_preprocess_config(base, WINDOW_PINGS, TOTAL_PINGS)
    assert cfg["despeckle"] == {"enabled": False}
    assert cfg["clahe"]["clip_limit"] == 4.0, "caller tunables must survive"
    assert "tile_grid" in cfg["clahe"]
    assert base["clahe"] == {"clip_limit": 4.0}, "caller dict must not be mutated"


def test_full_survey_config_is_a_no_op() -> None:
    """A window covering the whole survey keeps the stock grid."""
    cfg = _window_preprocess_config(None, TOTAL_PINGS, TOTAL_PINGS)
    assert tuple(cfg["clahe"]["tile_grid"]) == tuple(DEFAULTS["clahe"]["tile_grid"])


def test_pings_per_clahe_cell_matches_the_survey() -> None:
    """The mechanism: a CLAHE cell must span a comparable number of pings.

    This is the quantity that actually drives how hard the equaliser works.
    The stock grid gives a 200-ping window 25 pings per cell against the
    survey's 75 — a 3x more aggressive equaliser on identical seabed.
    """
    rows = DEFAULTS["clahe"]["tile_grid"][1]
    survey_per_cell = TOTAL_PINGS / rows

    naive_per_cell = WINDOW_PINGS / rows
    scaled_rows = _window_preprocess_config(None, WINDOW_PINGS, TOTAL_PINGS)["clahe"][
        "tile_grid"
    ][1]
    scaled_per_cell = WINDOW_PINGS / scaled_rows

    assert naive_per_cell < survey_per_cell / 2, "sanity: the stock grid is the bug"
    assert abs(scaled_per_cell - survey_per_cell) < 0.35 * survey_per_cell, (
        f"scaled window gives {scaled_per_cell:.1f} pings/cell against the "
        f"survey's {survey_per_cell:.1f}"
    )


def test_windowed_anomaly_rate_matches_batch(survey) -> None:
    """The symptom: window normalisation must not flood the open-set brain.

    Measured before the fix on this exact scene: 0.50 anomalies/tile over the
    whole survey against 19.8/tile in stock-grid windows.
    """
    anomaly = pytest.importorskip("tridentnet.anomaly")
    try:
        detector = anomaly.AnomalyDetector(device="cpu")
    except FileNotFoundError:
        pytest.skip("weights/anomaly.pt not trained in this environment")

    full = preprocess(survey)
    window = survey.slice_pings(0, WINDOW_PINGS)
    naive = preprocess(window)
    scaled = preprocess(
        window, config=_window_preprocess_config(None, WINDOW_PINGS, TOTAL_PINGS)
    )

    def per_tile(pre) -> float:
        return len(detector.detect_tiles(pre.tiles)) / max(len(pre.tiles), 1)

    full_rate, naive_rate, scaled_rate = per_tile(full), per_tile(naive), per_tile(scaled)

    assert naive_rate > 4 * max(full_rate, 0.25), (
        f"sanity: the stock grid should flood ({naive_rate:.2f}/tile vs "
        f"{full_rate:.2f}/tile)"
    )
    assert scaled_rate < naive_rate / 4, (
        f"scaling must collapse the flood: {scaled_rate:.2f}/tile vs "
        f"{naive_rate:.2f}/tile (survey {full_rate:.2f}/tile)"
    )
