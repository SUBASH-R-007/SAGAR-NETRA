"""Pixel -> WGS-84 geotagging.

A detection lives at (ping row, ground-range column) in a :class:`GroundImage`.
Geometry: the towfish is at the ping's nav fix, optionally pushed astern by
the layback along the reciprocal heading; the target sits at the across-track
ground range, perpendicular to the heading — port to the left (heading - 90°),
starboard to the right (heading + 90°). All offsets are solved on the WGS-84
ellipsoid with pyproj's geodesic engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import Geod

from sonar_core.preprocess.slant_range import GroundImage

_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class GeoBox:
    """Georeferenced detection footprint."""

    lat: float  # centre
    lon: float
    corners: tuple[tuple[float, float], ...]  # 4x (lat, lon), along-track x across-track
    along_m: float  # box extent along track
    across_m: float  # box extent across track


def towfish_position(
    lat: float, lon: float, heading_deg: float, layback_m: float = 0.0
) -> tuple[float, float]:
    """Antenna fix -> towfish fix, pushed astern by *layback_m*."""
    if layback_m <= 0:
        return lat, lon
    back = (heading_deg + 180.0) % 360.0
    lon2, lat2, _ = _GEOD.fwd(lon, lat, back, layback_m)
    return lat2, lon2


def offset_position(
    lat: float, lon: float, heading_deg: float, side: str, ground_range_m: float
) -> tuple[float, float]:
    """Across-track offset from the towfish to the imaged seabed point."""
    if side == "port":
        azimuth = (heading_deg - 90.0) % 360.0
    elif side == "starboard":
        azimuth = (heading_deg + 90.0) % 360.0
    else:
        raise KeyError(f"side must be 'port' or 'starboard', got {side!r}")
    lon2, lat2, _ = _GEOD.fwd(lon, lat, azimuth, float(ground_range_m))
    return lat2, lon2


def pixel_to_wgs84(
    gi: GroundImage,
    side: str,
    ping: int,
    col: float,
    layback_m: float | None = None,
) -> tuple[float, float]:
    """One corrected pixel -> (lat, lon). Raises on missing navigation."""
    rec = gi.nav[int(ping)]
    lat, lon = float(rec["lat"]), float(rec["lon"])
    if not (np.isfinite(lat) and np.isfinite(lon)):
        raise ValueError(f"ping {ping} has no navigation fix")
    layback = float(rec["layback"]) if layback_m is None else layback_m
    lat, lon = towfish_position(lat, lon, float(rec["heading"]), layback)
    ground = float(gi.ground_range_of_col(col))
    return offset_position(lat, lon, float(rec["heading"]), side, ground)


def bbox_to_geo(
    gi: GroundImage,
    side: str,
    ping0: int,
    ping1: int,
    col0: float,
    col1: float,
    layback_m: float | None = None,
) -> GeoBox:
    """Detection box (inclusive ping rows, ground columns) -> GeoBox.

    Corners are ordered (ping0,col0), (ping0,col1), (ping1,col1), (ping1,col0)
    so they trace the footprint's perimeter.
    """
    ping_c = (ping0 + ping1) // 2
    col_c = (col0 + col1) / 2.0
    lat, lon = pixel_to_wgs84(gi, side, ping_c, col_c, layback_m)
    corners = tuple(
        pixel_to_wgs84(gi, side, p, c, layback_m)
        for p, c in ((ping0, col0), (ping0, col1), (ping1, col1), (ping1, col0))
    )

    speeds = gi.nav["speed"][ping0 : ping1 + 1]
    times = gi.nav["time"][ping0 : ping1 + 1]
    if len(times) > 1 and np.isfinite(times).all() and times[-1] > times[0]:
        duration = float(times[-1] - times[0])
        along = float(np.nanmean(speeds)) * duration if np.isfinite(speeds).any() else 0.0
    else:
        along = 0.0
    if along <= 0:  # nav-less fallback: geodesic distance between corner rows
        _, _, dist = _GEOD.inv(corners[0][1], corners[0][0], corners[3][1], corners[3][0])
        along = float(dist)
    across = abs(col1 - col0) * gi.ground_res
    return GeoBox(lat=lat, lon=lon, corners=corners, along_m=along, across_m=across)
