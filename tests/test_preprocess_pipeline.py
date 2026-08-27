"""M2 pipeline orchestrator: full-chain results, progress contract, config
enable flags, callback robustness, and the end-to-end time budget."""

from __future__ import annotations

import time

import numpy as np

from sonar_core.parsers.base import load
from sonar_core.preprocess.pipeline import preprocess

ALL_STAGES = (
    "track_bottom",
    "egn",
    "blank_water_column",
    "slant_to_ground",
    "despeckle",
    "clahe",
    "tile",
)


def test_full_pipeline_on_small_scene(small_scene) -> None:
    pa, _ = small_scene
    result = preprocess(pa)

    assert result.tiles
    assert {t.side for t in result.tiles} == {"port", "starboard"}
    for side in ("port", "starboard"):
        img = result.ground.side(side)
        finite = img[np.isfinite(img)]
        assert finite.size, f"{side} ground image has no finite pixels"
        assert finite.min() >= 0.0 and finite.max() <= 1.0
        gain = result.egn_gain[side]
        assert gain.shape == (pa.n_samples(side),)
        assert (gain > 0).all()
    for stage in ALL_STAGES:
        assert stage in result.timings, f"missing timing for enabled stage {stage!r}"
        assert result.timings[stage] >= 0.0
    # Enhanced and raw ground images are fully independent arrays.
    assert not np.shares_memory(result.ground.port, result.ground_raw.port)
    assert not np.shares_memory(result.ground.starboard, result.ground_raw.starboard)


def test_progress_fractions_nondecreasing_and_done(small_scene) -> None:
    pa, _ = small_scene
    calls: list[tuple[str, float]] = []
    preprocess(pa, progress=lambda stage, frac: calls.append((stage, frac)))

    assert calls[0] == ("track_bottom", 0.0)
    assert calls[-1] == ("done", 1.0)
    fractions = [frac for _, frac in calls]
    assert all(0.0 <= frac <= 1.0 for frac in fractions)
    assert all(b >= a for a, b in zip(fractions, fractions[1:], strict=False))
    # One boundary call per enabled stage, in pipeline order, plus "done".
    assert [stage for stage, _ in calls] == [*ALL_STAGES, "done"]


def test_disabling_enhancement_yields_ground_equal_raw(small_scene) -> None:
    pa, _ = small_scene
    config = {"despeckle": {"enabled": False}, "clahe": {"enabled": False}}
    result = preprocess(pa, config=config)

    for side in ("port", "starboard"):
        np.testing.assert_allclose(
            result.ground.side(side), result.ground_raw.side(side), equal_nan=True
        )
        assert not np.shares_memory(result.ground.side(side), result.ground_raw.side(side))
    assert "despeckle" not in result.timings
    assert "clahe" not in result.timings


def test_raising_progress_callback_does_not_break_run(small_scene) -> None:
    pa, _ = small_scene

    def exploding(stage: str, frac: float) -> None:
        raise RuntimeError("observer crashed")

    result = preprocess(pa, progress=exploding)
    assert result.tiles
    assert result.timings


def test_e2e_budget_on_sample_xtf(sample_xtf) -> None:
    pa = load(sample_xtf)
    start = time.perf_counter()
    result = preprocess(pa)
    elapsed = time.perf_counter() - start

    assert elapsed < 90.0, f"pipeline took {elapsed:.1f}s (budget 90s)"
    assert len(result.tiles) > 4
