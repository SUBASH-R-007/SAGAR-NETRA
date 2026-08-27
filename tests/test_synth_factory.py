"""Synthetic-data factory: ray-traced shadow geometry, radiometrically
matched copy-paste with NaN-swath safety, and the physics-safe augmentation
group (no horizontal mirror, ever)."""

from __future__ import annotations

import numpy as np
import pytest

from sonar_core.synth.augment import apply_chip, train_augment
from sonar_core.synth.copy_paste import EMPTY_BBOX, paste_target
from sonar_core.synth.shadow_render import render_shadow

GROUND_RES = 0.5  # m per column
ALTITUDE = 8.0  # m
NADIR_COL0 = 10  # tile col 0 sits 10 ground columns from nadir
HEIGHT = 2.0  # m proud height
C_FAR = 40  # far-most footprint column of the seeded box


def _box_scene() -> tuple[np.ndarray, np.ndarray]:
    """Uniform 100-intensity tile with a box footprint, far edge at C_FAR."""
    img = np.full((64, 96), 100.0, dtype=np.float32)
    mask = np.zeros(img.shape, dtype=bool)
    mask[20:31, 30 : C_FAR + 1] = True
    return img, mask


def test_render_shadow_length_matches_ray_geometry() -> None:
    """Seeded box at known ground position: rendered shadow length equals
    x_far*A/(A-H) - x_far within 2 columns; unmasked rows and the input
    array are untouched."""
    img, mask = _box_scene()
    out = render_shadow(
        img, mask, GROUND_RES, ALTITUDE, NADIR_COL0, HEIGHT, feather_px=0.0
    )

    x_far = (NADIR_COL0 + C_FAR + 1.0) * GROUND_RES
    x_end = x_far * ALTITUDE / (ALTITUDE - HEIGHT)
    expected_cols = (x_end - x_far) / GROUND_RES

    row = 25  # a masked row
    darkened = np.flatnonzero(out[row] < 0.5 * img[row])
    assert darkened.size > 0
    assert darkened.min() == C_FAR + 1  # shadow starts right behind the far edge
    assert np.array_equal(darkened, np.arange(darkened.min(), darkened.max() + 1))
    assert abs(darkened.size - expected_cols) <= 2
    # deep shadow sits at the multiplicative floor
    assert out[row, C_FAR + 2] == pytest.approx(100.0 * 0.12, rel=1e-5)

    no_obj = ~mask.any(axis=1)
    np.testing.assert_array_equal(out[no_obj], img[no_obj])
    assert (img == 100.0).all()  # input never modified


def test_shadow_darkens_down_range_only() -> None:
    """Nothing at or before the footprint's far edge may darken — shadows
    exist only toward increasing column (away from nadir)."""
    img, mask = _box_scene()
    out = render_shadow(
        img, mask, GROUND_RES, ALTITUDE, NADIR_COL0, HEIGHT, feather_px=1.0
    )
    np.testing.assert_array_equal(out[:, : C_FAR + 1], img[:, : C_FAR + 1])
    assert (out[:, C_FAR + 1 :] <= img[:, C_FAR + 1 :] + 1e-6).all()


def test_render_shadow_per_row_altitude() -> None:
    """Shadow length L = x_far*H/(A-H) grows as altitude shrinks: a per-row
    altitude array must yield a longer shadow on the low-altitude rows."""
    img, mask = _box_scene()
    alt = np.full(img.shape[0], 10.0)
    alt[:25] = 6.0  # split inside the mask band: rows 20..24 low, 25..30 high
    out = render_shadow(img, mask, GROUND_RES, alt, NADIR_COL0, HEIGHT, feather_px=0.0)
    low = np.count_nonzero(out[22] < 0.5 * img[22])
    high = np.count_nonzero(out[28] < 0.5 * img[28])
    assert low > high > 0


def test_paste_target_position_brightness_shadow_and_nan() -> None:
    """Chip lands at the requested position with the correct bbox, the pasted
    highlight sits near the chip's intended contrast ratio over the local
    background, a shadow is cast down-range, and NaN pixels survive."""
    rng = np.random.default_rng(11)
    bg = np.full((96, 96), 100.0, dtype=np.float32)
    bg[44:46, 34:36] = np.nan  # dropout inside the future footprint
    bg[:8, 80:] = np.nan  # out-of-swath corner

    ratio = 5.0
    chip = np.full((11, 11), 50.0, dtype=np.float32)
    chip_mask = np.zeros((11, 11), dtype=bool)
    chip_mask[3:8, 3:8] = True
    chip[chip_mask] = 50.0 * ratio  # intended highlight ratio over chip context

    res = paste_target(
        bg,
        chip,
        chip_mask,
        position_rc=(40, 30),
        ground_res=GROUND_RES,
        altitude_m=ALTITUDE,
        nadir_col0=0,
        height_m=1.0,
        rng=rng,
        blend_sigma=1.0,
    )

    # footprint lands where asked, minus the NaN holes; bbox wraps it
    expected_mask = np.zeros(bg.shape, dtype=bool)
    expected_mask[43:48, 33:38] = True
    expected_mask[44:46, 34:36] = False  # NaN never pasted over
    np.testing.assert_array_equal(res.mask, expected_mask)
    assert res.bbox == (43, 33, 48, 38)
    assert res.height_m == 1.0

    # highlight brightness: intended level = local_bg * ratio, within 2x
    fg_med = float(np.median(res.image[res.mask]))
    intended = 100.0 * ratio
    assert intended / 2.0 <= fg_med <= intended * 2.0

    # a shadow was cast down-range of the far edge (col 37)
    assert res.image[45, 39] < 0.5 * 100.0

    # NaN pixels preserved exactly — no more, no fewer
    assert np.isnan(res.image[44:46, 34:36]).all()
    assert np.isnan(res.image[:8, 80:]).all()
    assert np.isnan(res.image).sum() == np.isnan(bg).sum()

    # the input background was never modified
    assert np.nanmax(bg) == 100.0 and np.nanmin(bg) == 100.0


def test_paste_target_clips_partial_and_off_tile_chips() -> None:
    """A chip hanging off the tile edge is clipped; one fully off-tile is a
    no-op with the sentinel bbox."""
    rng = np.random.default_rng(3)
    bg = np.full((64, 64), 100.0, dtype=np.float32)
    chip = np.full((9, 9), 300.0, dtype=np.float32)
    chip_mask = np.ones((9, 9), dtype=bool)

    res = paste_target(
        bg, chip, chip_mask, (-4, 60), GROUND_RES, ALTITUDE, 0, 0.5, rng=rng
    )
    assert res.mask.any()
    r0, c0, r1, c1 = res.bbox
    assert (r0, c0) == (0, 60) and (r1, c1) == (5, 64)
    assert res.mask.sum() == (r1 - r0) * (c1 - c0)

    res_off = paste_target(
        bg, chip, chip_mask, (200, 200), GROUND_RES, ALTITUDE, 0, 0.5, rng=rng
    )
    assert not res_off.mask.any()
    assert res_off.bbox == EMPTY_BBOX
    np.testing.assert_array_equal(res_off.image, bg)


def test_augment_preserves_dtype_shape_and_never_mirrors() -> None:
    """20 augmented samples of a left-bright/right-dark chip: dtype and shape
    preserved, and the left half stays brighter in every sample — proof that
    no transform mirrors the across-track (shadow) axis."""
    img = np.full((64, 64), 30, dtype=np.uint8)
    img[:, :32] = 200  # bright near nadir, dark down-range: asymmetric
    aug = train_augment(seed=123)
    for _ in range(20):
        out = apply_chip(aug, img)
        assert out.dtype == np.uint8
        assert out.shape == img.shape
        assert float(out[:, :32].mean()) > float(out[:, 32:].mean())


def test_apply_chip_rejects_float_input() -> None:
    """Float chips would be silently misinterpreted as [0, 1] images."""
    aug = train_augment(seed=1)
    with pytest.raises(TypeError, match="uint8"):
        apply_chip(aug, np.zeros((32, 32), dtype=np.float32))
