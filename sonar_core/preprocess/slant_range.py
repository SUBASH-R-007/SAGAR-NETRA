"""Slant-range to ground-range correction.

Raw pings are sampled uniformly in slant range (two-way travel time). For a
flat seabed and towfish altitude ``A``, a sample at slant range ``r`` images
the seabed at ground range ``g = sqrt(r^2 - A^2)``. This module resamples
each ping onto a uniform ground-range grid, per-ping altitude aware, and
keeps the mapping exactly invertible so any corrected pixel maps back to its
source (ping, slant sample) — and from there to navigation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sonar_core.parsers.base import PingArray
from sonar_core.preprocess.bottom_track import BottomTrack


@dataclass
class GroundImage:
    """Ground-range imagery for both sides plus the exact inverse mapping.

    Columns are uniform ground range: column ``j`` spans ground range
    ``[j, j+1) * ground_res`` (centre at ``(j + 0.5) * ground_res``). Rows are
    ping indices, aligned 1:1 with ``nav``.
    """

    port: np.ndarray  # (n_pings, n_cols_port) float32, NaN beyond swath
    starboard: np.ndarray
    ground_res: float  # metres per column
    altitude_m: np.ndarray  # (n_pings,) altitude used for the correction
    slant_res: dict[str, float]  # side -> metres per source sample
    nav: np.ndarray  # NAV_DTYPE records, one per row
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_pings(self) -> int:
        return self.port.shape[0]

    def side(self, name: str) -> np.ndarray:
        if name not in ("port", "starboard"):
            raise KeyError(f"side must be 'port' or 'starboard', got {name!r}")
        return getattr(self, name)

    def n_cols(self, side: str) -> int:
        return self.side(side).shape[1]

    def ground_range_of_col(self, col: np.ndarray | int) -> np.ndarray:
        """Ground range (m) at column centre."""
        return (np.asarray(col, dtype=np.float64) + 0.5) * self.ground_res

    def col_of_ground_range(self, ground: np.ndarray | float) -> np.ndarray:
        """Fractional column index whose centre sits at *ground* metres."""
        return np.asarray(ground, dtype=np.float64) / self.ground_res - 0.5

    def to_slant_sample(self, side: str, ping: int, col: np.ndarray | int) -> np.ndarray:
        """Corrected pixel -> fractional source sample index in the raw ping."""
        g = self.ground_range_of_col(col)
        a = float(self.altitude_m[ping])
        r = np.hypot(g, a)
        return r / self.slant_res[side] - 0.5

    def to_ground_col(self, side: str, ping: int, sample: np.ndarray | int) -> np.ndarray:
        """Raw (ping, sample) -> fractional corrected column. NaN in the
        water column (slant range shorter than altitude)."""
        r = (np.asarray(sample, dtype=np.float64) + 0.5) * self.slant_res[side]
        a = float(self.altitude_m[ping])
        with np.errstate(invalid="ignore"):
            g = np.sqrt(r**2 - a**2)
        return self.col_of_ground_range(g)


def slant_to_ground(
    pa: PingArray,
    bt: BottomTrack | None = None,
    ground_res: float | None = None,
    fill: float = np.nan,
) -> GroundImage:
    """Resample both sides onto a uniform ground-range grid.

    ``ground_res`` defaults to the finest side's slant resolution, so no
    information is lost at far range (where ground and slant spacing match).
    Output width covers the largest achievable ground range across pings;
    pixels beyond a ping's own swath are *fill* (NaN by default so later
    stages can mask them).
    """
    if pa.meta.get("ground_range"):
        raise ValueError("input is already ground-range (mosaic); skip correction")

    altitude = (bt.altitude_m if bt is not None else pa.nav["altitude"]).astype(np.float64)
    if not np.isfinite(altitude).all() or (altitude <= 0).any():
        raise ValueError("slant correction needs finite positive altitude for every ping")

    slant_res = {s: pa.slant_resolution(s) for s in ("port", "starboard")}
    usable = [r for r in slant_res.values() if r > 0]
    if not usable:
        raise ValueError("no non-empty side to correct")
    if ground_res is None:
        ground_res = min(usable)

    out: dict[str, np.ndarray] = {}
    for side in ("port", "starboard"):
        res = slant_res[side]
        n_samples = pa.n_samples(side)
        if n_samples == 0 or res <= 0:
            out[side] = np.zeros((pa.n_pings, 0), dtype=np.float32)
            continue
        max_slant = n_samples * res
        # Widest swath any ping achieves (smallest altitude wins).
        max_ground = float(np.sqrt(max(max_slant**2 - float(altitude.min()) ** 2, 0.0)))
        n_cols = int(np.floor(max_ground / ground_res))
        img = np.full((pa.n_pings, n_cols), fill, dtype=np.float32)

        ground_centres = (np.arange(n_cols) + 0.5) * ground_res
        src = pa.side(side)
        sample_axis = np.arange(n_samples, dtype=np.float64)
        first_return = bt.first_return[side] if bt is not None else None
        for i in range(pa.n_pings):
            a = altitude[i]
            r = np.hypot(ground_centres, a)
            s = r / res - 0.5
            # Columns whose source sample precedes the first bottom return
            # would interpolate against water-column fill — a silent blend of
            # artificial and real data. Mask them to the fill value instead.
            cut = float(first_return[i]) if first_return is not None else np.round(a / res)
            inside = (s <= n_samples - 1) & (s >= cut)
            if not inside.any():
                continue
            img[i, inside] = np.interp(s[inside], sample_axis, src[i])
        out[side] = img

    return GroundImage(
        port=out["port"],
        starboard=out["starboard"],
        ground_res=float(ground_res),
        altitude_m=altitude.astype(np.float32),
        slant_res=slant_res,
        nav=pa.nav.copy(),
        meta={**pa.meta, "ground_range": True, "fill": "nan" if np.isnan(fill) else fill},
    )
