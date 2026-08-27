"""CLAHE display enhancement: output range, NaN handling, local-contrast gain,
and that the tunables actually reach the OpenCV backend."""

from __future__ import annotations

import numpy as np

from sonar_core.preprocess.clahe import clahe

DIM_PATCH = (slice(80, 176), slice(32, 144))  # rows, cols of the dim texture


def _dim_bright_image(seed: int = 3) -> np.ndarray:
    """Bright field (5000..9000) with a dim textured patch (80..120) inside —
    the classic shadow-interior situation where global stretch flattens the
    patch to near-zero contrast."""
    rng = np.random.default_rng(seed)
    img = rng.uniform(5000.0, 9000.0, size=(256, 256)).astype(np.float32)
    img[DIM_PATCH] = rng.uniform(80.0, 120.0, size=(96, 112)).astype(np.float32)
    return img


def _linear_norm(img: np.ndarray) -> np.ndarray:
    """Plain global min/max normalization to [0, 1] over finite pixels."""
    lo = np.nanmin(img)
    hi = np.nanmax(img)
    return ((img - lo) / (hi - lo)).astype(np.float32)


def test_range_dtype_and_nan_preserved() -> None:
    img = _dim_bright_image()
    img[:, :7] = np.nan  # out-of-swath stripe, e.g. blanked water column
    img[100:, 120:] = np.nan
    nan_mask = np.isnan(img)

    out = clahe(img)

    assert out.dtype == np.float32
    assert out.shape == img.shape
    assert np.array_equal(np.isnan(out), nan_mask)
    finite = out[~nan_mask]
    assert np.isfinite(finite).all()
    assert finite.min() >= 0.0
    assert finite.max() <= 1.0


def test_local_contrast_in_dim_region_increases() -> None:
    """The dim patch must gain contrast versus what a global linear stretch
    gives — that is the whole point of *adaptive* equalization."""
    img = _dim_bright_image()
    linear_std = float(np.std(_linear_norm(img)[DIM_PATCH]))
    clahe_std = float(np.std(clahe(img)[DIM_PATCH]))
    assert clahe_std > linear_std


def test_constant_image_is_finite() -> None:
    img = np.full((64, 64), 500.0, dtype=np.float32)
    out = clahe(img)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_all_nan_returns_all_nan() -> None:
    img = np.full((32, 32), np.nan, dtype=np.float32)
    out = clahe(img)
    assert out.dtype == np.float32
    assert np.isnan(out).all()


def test_tile_grid_and_clip_limit_are_honored() -> None:
    """Different settings must change the output. The clip comparison uses a
    coarse grid: OpenCV's 16-bit CLAHE floors the per-bin cap at
    ``max(int(clip * tile_area / 65536), 1)``, so tiles must be large enough
    for two clip limits to yield different effective caps."""
    img = _dim_bright_image()
    base = clahe(img, clip_limit=2.5, tile_grid=(8, 8))
    coarse_tiles = clahe(img, clip_limit=2.5, tile_grid=(2, 2))
    high_clip = clahe(img, clip_limit=8.0, tile_grid=(2, 2))
    assert not np.allclose(base, coarse_tiles)
    assert not np.allclose(coarse_tiles, high_clip)
