"""GeoTIFF mosaic adapter (optional; requires ``rasterio``, install extra ``[geo]``).

A mosaic is already ground-range and georeferenced, so rows are treated as
pseudo-pings with navigation synthesized from the geotransform, and
``meta["ground_range"] = True`` tells preprocessing to skip bottom tracking,
water-column removal, and slant-range correction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio  # noqa: F401 — import failure keeps this adapter unregistered
from pyproj import Transformer

from sonar_core.parsers.base import NAV_DTYPE, PingArray, SonarParser, register_parser


@register_parser
class GeoTIFFParser(SonarParser):
    suffixes = (".gtif", ".geotiff", ".gtiff")

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        if path.suffix.lower() in cls.suffixes:
            return True
        if path.suffix.lower() in (".tif", ".tiff"):
            try:
                with rasterio.open(path) as src:
                    return src.crs is not None
            except Exception:
                return False
        return False

    def parse(self, path: Path, **kwargs: Any) -> PingArray:
        with rasterio.open(path) as src:
            img = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata

        if nodata is not None:
            img = np.where(img == nodata, np.nan, img)

        n_rows, n_cols = img.shape
        rows = np.arange(n_rows)
        centre_col = n_cols / 2.0
        xs, ys = rasterio.transform.xy(transform, rows, np.full(n_rows, centre_col))
        xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
        if crs is not None and crs.to_epsg() != 4326:
            to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            xs, ys = to_wgs.transform(xs, ys)

        # Heading of increasing-row direction from consecutive row centres.
        dlon = np.gradient(xs) * np.cos(np.deg2rad(ys))
        dlat = np.gradient(ys)
        heading = (np.degrees(np.arctan2(dlon, dlat)) + 360.0) % 360.0

        pixel_size = float(abs(transform.a))  # ground metres if projected CRS
        half_swath = pixel_size * n_cols / 2.0

        nav = np.zeros(n_rows, dtype=NAV_DTYPE)
        nav["time"] = np.nan
        nav["lat"] = ys
        nav["lon"] = xs
        nav["heading"] = heading
        nav["altitude"] = np.nan
        nav["sound_velocity"] = 1500.0
        nav["slant_range"] = half_swath  # ground range here (mosaic is corrected)

        half = n_cols // 2
        img_filled = np.nan_to_num(img, nan=0.0)
        return PingArray(
            port=img_filled[:, :half][:, ::-1],
            starboard=img_filled[:, half:],
            nav=nav,
            source=str(path),
            meta={
                "format": "geotiff",
                "sonar_name": "mosaic",
                "ground_range": True,
                "crs": str(crs),
                "pixel_size": pixel_size,
            },
        )
