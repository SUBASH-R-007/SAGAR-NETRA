"""Waterfall rendering and exact column bookkeeping."""

from __future__ import annotations

import numpy as np
from PIL import Image

from sonar_core.waterfall import (
    WaterfallLayout,
    combine,
    layout_for,
    normalize_u8,
    save_waterfall_png,
)


def test_column_map_roundtrips_every_pixel() -> None:
    layout = WaterfallLayout(n_port=32, n_starboard=48)
    for col in range(layout.width):
        side, sample = layout.col_to_sample(col)
        assert layout.sample_to_col(side, sample) == col
    assert layout.col_to_sample(0) == ("port", 31)  # far port range at left edge
    assert layout.col_to_sample(31) == ("port", 0)  # port nadir at centreline
    assert layout.col_to_sample(32) == ("starboard", 0)
    assert layout.col_to_sample(79) == ("starboard", 47)


def test_combine_mirrors_port(small_scene) -> None:
    pa, _ = small_scene
    img = combine(pa)
    layout = layout_for(pa)
    assert img.shape == (pa.n_pings, layout.width)
    np.testing.assert_array_equal(img[:, 0], pa.port[:, -1])
    np.testing.assert_array_equal(img[:, layout.n_port], pa.starboard[:, 0])


def test_normalize_u8_stretch() -> None:
    img = np.linspace(0, 1000, 256).reshape(16, 16).astype(np.float32)
    out = normalize_u8(img)
    assert out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255


def test_save_png(small_scene, tmp_path) -> None:
    pa, _ = small_scene
    path = save_waterfall_png(pa, tmp_path / "wf.png")
    with Image.open(path) as im:
        assert im.size == (layout_for(pa).width, pa.n_pings)
        assert im.mode == "L"
