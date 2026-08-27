"""SAHI tiling: lossless coverage, overlap geometry, and exact coordinate
bookkeeping from tile pixels back to (ping, ground column) and raw slant
samples."""

from __future__ import annotations

import numpy as np
import pytest

from sonar_core.preprocess.bottom_track import track_bottom
from sonar_core.preprocess.slant_range import slant_to_ground
from sonar_core.preprocess.tiler import tile_image, tiles_for_ground_image


def _footprint_union(tiles, shape: tuple[int, int]) -> np.ndarray:
    covered = np.zeros(shape, dtype=bool)
    for t in tiles:
        h, w = t.image.shape
        covered[t.row0 : t.row0 + h, t.col0 : t.col0 + w] = True
    return covered


@pytest.mark.parametrize("shape", [(100, 80), (1024, 1024), (1500, 2100), (700, 3000)])
def test_coverage_is_complete(shape: tuple[int, int]) -> None:
    """Union of tile footprints covers every pixel; no tile leaves the image."""
    img = np.ones(shape, dtype=np.float32)
    tiles = tile_image(img, "port", tile_size=512, overlap=0.25)
    for t in tiles:
        h, w = t.image.shape
        assert t.row0 >= 0 and t.col0 >= 0
        assert t.row0 + h <= shape[0] and t.col0 + w <= shape[1]
    assert _footprint_union(tiles, shape).all()


def test_overlap_geometry() -> None:
    """Consecutive origins differ by stride, except the shifted last tile
    which ends exactly at the image edge."""
    shape = (1500, 2100)
    tile_size = 512
    stride = round(tile_size * (1 - 0.25))  # 384
    img = np.ones(shape, dtype=np.float32)
    tiles = tile_image(img, "port", tile_size=tile_size, overlap=0.25)
    assert [t.index for t in tiles] == list(range(len(tiles)))
    for axis, origins in enumerate(
        (sorted({t.row0 for t in tiles}), sorted({t.col0 for t in tiles}))
    ):
        diffs = np.diff(origins)
        assert (diffs[:-1] == stride).all()
        assert 0 < diffs[-1] <= stride
        assert origins[-1] + tile_size == shape[axis]


def test_to_global_from_global_roundtrip() -> None:
    """Local -> global -> local is exact for random pixels in every tile."""
    rng = np.random.default_rng(1)
    img = rng.random((300, 200)).astype(np.float32)
    tiles = tile_image(img, "starboard", tile_size=128, overlap=0.25)
    assert tiles
    for t in tiles:
        h, w = t.image.shape
        r = rng.integers(0, h, size=20)
        c = rng.integers(0, w, size=20)
        ping, col = t.to_global(r, c)
        assert np.array_equal(ping, r + t.row0)
        assert np.array_equal(col, c + t.col0)
        assert t.contains(ping, col)
        r2, c2 = t.from_global(ping, col)
        assert np.array_equal(r2, r)
        assert np.array_equal(c2, c)
    t0 = tiles[0]
    assert not t0.contains(-1, t0.col0)
    with pytest.raises(ValueError, match="outside tile"):
        t0.from_global(img.shape[0] + 5, t0.col0)


def test_tile_image_is_exact_contiguous_copy() -> None:
    """tile.image equals the source slice (NaN-equal) and shares no memory."""
    rng = np.random.default_rng(3)
    img = rng.random((300, 200)).astype(np.float32)
    img[rng.random((300, 200)) < 0.1] = np.nan
    tiles = tile_image(img, "port", tile_size=128, overlap=0.25)
    for t in tiles:
        h, w = t.image.shape
        np.testing.assert_array_equal(t.image, img[t.row0 : t.row0 + h, t.col0 : t.col0 + w])
        assert t.image.dtype == np.float32
        assert t.image.flags["C_CONTIGUOUS"]
        assert not np.shares_memory(t.image, img)


def test_m2_chain_tile_pixel_to_slant_and_back(small_scene) -> None:
    """M2 acceptance: tile pixel -> (ping, col) -> slant sample -> ground col
    -> tile-local again, error < 1 pixel, on a real synthetic scene."""
    pa, _ = small_scene
    bt = track_bottom(pa)
    gi = slant_to_ground(pa, bt)
    tiles = tiles_for_ground_image(gi, tile_size=64, overlap=0.25)
    assert {t.side for t in tiles} == {"port", "starboard"}
    assert [t.index for t in tiles] == list(range(len(tiles)))  # unique across sides

    rng = np.random.default_rng(2)
    for k in rng.integers(0, len(tiles), size=50):
        t = tiles[k]
        h, w = t.image.shape
        r = int(rng.integers(0, h))
        c = int(rng.integers(0, w))
        ping, col = t.to_global(r, c)
        s = gi.to_slant_sample(t.side, int(ping), int(col))
        back_col = gi.to_ground_col(t.side, int(ping), s)
        back_r, back_c = int(ping) - t.row0, float(back_col) - t.col0
        assert back_r == r
        assert abs(back_c - c) < 1.0


def test_zero_finite_tiles_dropped_kept_tiles_have_content() -> None:
    """Fully-NaN (out-of-swath) tiles are dropped; kept tiles keep every
    finite pixel covered and each contains real content."""
    img = np.ones((256, 256), dtype=np.float32)
    img[:128, :128] = np.nan
    tiles = tile_image(img, "port", tile_size=64, overlap=0.0)
    assert len(tiles) == 12  # 4x4 grid minus the 4 fully-NaN corner tiles
    for t in tiles:
        assert np.isfinite(t.image).any()
    covered = _footprint_union(tiles, img.shape)
    assert covered[np.isfinite(img)].all()
    assert tile_image(np.full((64, 64), np.nan, dtype=np.float32), "port", tile_size=64) == []
