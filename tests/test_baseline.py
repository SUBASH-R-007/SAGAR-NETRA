"""Tests for the classical CAD baseline and the head-to-head comparison.

The baseline exists to be *beaten fairly*, which makes its correctness load
bearing in a way an unused module's would not be: a broken baseline silently
flatters SAGAR-NETRA, and every published comparison number inherits the lie.
So these tests pin the behaviours a strawman would fail — that it finds an
obvious target, that the shadow gate actually gates on shadow, and that
raising the threshold cannot somehow find *more*.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from tridentnet.baseline import (
    UNCLASSIFIED,
    ClassicalCAD,
    ClassicalConfig,
    _background,
    detect_side,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_WEIGHTS = REPO_ROOT / "weights" / "verifier.pkl"


def _load_script(name: str) -> ModuleType:
    """Load a CLI script by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _plant(
    *,
    height: int = 200,
    width: int = 140,
    rows: tuple[int, int] = (50, 62),
    cols: tuple[int, int] = (40, 54),
    shadow: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """A flat noisy seabed with one bright target and an optional shadow.

    The target spans a small minority of pings so it cannot drag the
    per-column median that :func:`_background` relies on — the same condition
    that holds on real survey lines.
    """
    rng = np.random.default_rng(seed)
    img = rng.normal(1.0, 0.05, size=(height, width)).astype(np.float32)
    img[rows[0] : rows[1], cols[0] : cols[1]] = 3.0
    if shadow:
        img[rows[0] : rows[1], cols[1] : cols[1] + 30] = 0.15
    return img


def test_background_is_robust_to_the_target() -> None:
    """A bright target must not move the background it is measured against."""
    med, sigma = _background(_plant())
    target_cols = med[40:54]
    clean_cols = med[100:120]
    assert np.allclose(target_cols.mean(), clean_cols.mean(), atol=0.05)
    assert np.all(sigma > 0)


def test_finds_an_obvious_target() -> None:
    found = detect_side(_plant(), "starboard", ClassicalConfig(k_sigma=5.0))
    assert len(found) >= 1
    hit = max(found, key=lambda d: d.score)
    assert hit.ping0 <= 55 <= hit.ping1
    assert hit.col0 <= 47 <= hit.col1
    assert hit.cls == UNCLASSIFIED
    assert hit.brain == "classical"
    assert 0.0 <= hit.score <= 1.0


def test_shadow_gate_requires_a_shadow() -> None:
    """The gate is the whole difference between the two published rows."""
    cfg = ClassicalConfig(k_sigma=5.0, require_shadow=True)
    with_shadow = detect_side(_plant(shadow=True), "starboard", cfg)
    without = detect_side(_plant(shadow=False), "starboard", cfg)
    assert len(with_shadow) >= 1
    assert len(without) == 0

    # Without the gate the shadowless target is still found, so the rejection
    # above is attributable to the gate and not to the target being invisible.
    ungated = detect_side(
        _plant(shadow=False), "starboard", ClassicalConfig(k_sigma=5.0)
    )
    assert len(ungated) >= 1


def test_raising_the_threshold_never_finds_more() -> None:
    img = _plant()
    counts = [
        len(detect_side(img, "starboard", ClassicalConfig(k_sigma=k)))
        for k in (1.0, 3.0, 8.0, 20.0)
    ]
    assert counts == sorted(counts, reverse=True)


def test_area_filter_rejects_speckle() -> None:
    """Pure noise at a permissive threshold must not survive the area filter."""
    rng = np.random.default_rng(7)
    noise = rng.normal(1.0, 0.05, size=(200, 140)).astype(np.float32)
    strict = detect_side(noise, "port", ClassicalConfig(k_sigma=3.0, min_area_px=50))
    assert strict == []


def test_nan_pixels_are_survivable() -> None:
    """Blanked water column and swath edges arrive as NaN, not as zeros."""
    img = _plant()
    img[:, :8] = np.nan
    found = detect_side(img, "port", ClassicalConfig(k_sigma=5.0))
    assert len(found) >= 1
    assert all(np.isfinite(d.score) for d in found)


def test_boxes_stay_inside_the_image() -> None:
    """The down-range pad must not run a box off the swath."""
    img = _plant(cols=(120, 134), shadow=False)  # target hard against the edge
    for d in detect_side(img, "starboard", ClassicalConfig(k_sigma=5.0)):
        assert 0 <= d.col0 <= d.col1 < img.shape[1]
        assert 0 <= d.ping0 <= d.ping1 < img.shape[0]


def test_cad_defaults_to_uncontrasted_imagery() -> None:
    """Detecting on the CLAHE'd image would understate the baseline badly."""
    assert ClassicalConfig().use_raw_imagery is True

    class FakePre:
        def __init__(self) -> None:
            self.ground = _FakeSide(np.zeros((40, 40), dtype=np.float32))
            self.ground_raw = _FakeSide(_plant(height=200, width=140))

    found = ClassicalCAD().detect(FakePre())
    assert len(found) >= 1  # came from ground_raw; ground is featureless


class _FakeSide:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def side(self, _name: str) -> np.ndarray:
        return self._arr


@pytest.mark.skipif(
    not VERIFIER_WEIGHTS.exists(),
    reason="weights/verifier.pkl not trained yet (run scripts/train_verifier.py)",
)
def test_comparison_harness_runs_and_separates_the_splits(tmp_path) -> None:
    """Smoke test: the harness selects on split A and reports on split B."""
    from tridentnet.detector import Detection

    class StubDetector:
        def detect_tiles(self, tiles: list, progress=None) -> list[Detection]:
            by_side: dict[str, object] = {}
            for tile in tiles:
                by_side.setdefault(tile.side, tile)
            out = []
            for side, tile in by_side.items():
                h, w = tile.image.shape
                r0, c0 = tile.row0 + h // 3, tile.col0 + w // 3
                out.append(
                    Detection(side=side, ping0=r0, ping1=r0 + 10, col0=c0, col1=c0 + 16,
                              cls="container", score=0.8, tile_index=tile.index)
                )
            return out

    module = _load_script("eval_baseline")
    out = tmp_path / "comparison.md"
    rows = module.run_comparison(
        n_scenes=1,
        n_tune_scenes=1,
        out_path=out,
        detector=StubDetector(),
        k_grid=(2.0, 8.0),
        n_pings_range=(220, 260),
        n_samples=512,
    )

    assert set(rows) == {"blob", "blob_shadow", "sagar_raw", "sagar_full", "sagar_tuned"}
    text = out.read_text(encoding="utf-8")
    assert "SYNTHETIC" in text  # the caveat must travel with the table
    assert "Split A (tuning)" in text and "Split B (evaluation)" in text
    assert "tuned on split A" in text
    for row in rows.values():
        m = row.metrics
        for value in (m.precision, m.recall, m.f1, m.pr_auc):
            assert 0.0 <= value <= 1.0
        assert m.fp_per_km2 >= 0.0


class TestEndpointNote:
    """The endpoint diagnostic decides how a reader weighs the tuned row.

    Its job is to separate "the sweep was too narrow" from "the objective is
    flat here", because only the first is a reason to distrust the number and
    only the first is fixable by widening the grid.
    """

    GRID = (1.0, 2.0, 3.0, 4.0)

    def _note(self, k_sel, curve):
        module = _load_script("eval_baseline")
        return module._endpoint_note("blob", k_sel, curve, self.GRID)

    def test_interior_selection_is_silent(self) -> None:
        curve = [(1.0, 0.5), (2.0, 0.9), (3.0, 0.6), (4.0, 0.4)]
        assert self._note(2.0, curve) is None

    def test_flat_endpoint_is_called_a_plateau(self) -> None:
        curve = [(1.0, 0.900), (2.0, 0.895), (3.0, 0.890), (4.0, 0.100)]
        note = self._note(1.0, curve)
        assert note is not None and "plateau" in note
        # It may *mention* widening, but must not prescribe it: widening a
        # plateau chases noise. That imperative belongs to the other branch.
        assert "widen the grid before" not in note

    def test_genuinely_better_endpoint_asks_for_a_wider_grid(self) -> None:
        curve = [(1.0, 0.900), (2.0, 0.500), (3.0, 0.400), (4.0, 0.300)]
        note = self._note(1.0, curve)
        assert note is not None and "widen the grid" in note

    def test_grid_with_no_interior_reports_itself_as_too_coarse(self) -> None:
        module = _load_script("eval_baseline")
        note = module._endpoint_note("blob", 2.0, [(2.0, 0.5), (8.0, 0.4)], (2.0, 8.0))
        assert note is not None and "no interior" in note
