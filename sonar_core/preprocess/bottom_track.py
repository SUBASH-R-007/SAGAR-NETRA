"""Bottom tracking: locate the first bottom return per ping, refine the
altitude series, and blank the water column.

The first return is the earliest sustained jump above the water-column noise
level. Header altitude (``nav["altitude"]``) is used only as a sanity prior;
the tracked value is authoritative for geometry because real loggers drop or
mis-record altimeter data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from sonar_core.parsers.base import PingArray


@dataclass
class BottomTrack:
    """Per-ping altitude solution and per-side first-return indices."""

    altitude_m: np.ndarray  # (n_pings,) fused + smoothed altitude, metres
    first_return: dict[str, np.ndarray]  # side -> (n_pings,) int32 sample index
    header_altitude_m: np.ndarray  # (n_pings,) altitude as recorded in nav
    valid: np.ndarray  # (n_pings,) bool, False where tracking fell back to header

    def altitude_samples(self, pa: PingArray, side: str) -> np.ndarray:
        """Altitude expressed in slant samples for *side*."""
        res = pa.slant_resolution(side)
        if res <= 0:
            return np.zeros(len(self.altitude_m), dtype=np.float32)
        return self.altitude_m / res


def _first_returns_one_side(
    intensity: np.ndarray,
    noise_factor: float,
    min_run: int,
) -> np.ndarray:
    """First sample per ping whose smoothed level exceeds the noise threshold
    for at least *min_run* consecutive samples. Returns -1 where not found."""
    n_pings, n_samples = intensity.shape
    smoothed = uniform_filter1d(intensity.astype(np.float32), size=3, axis=1)

    # Water-column noise level per ping: median of the first few samples
    # (guaranteed pre-bottom for any plausible altitude > 0).
    head = max(4, n_samples // 64)
    noise = np.median(smoothed[:, :head], axis=1)
    # Guard against zero noise floors in synthetic/clean data.
    floor = max(float(np.median(noise)) * 0.1, 1e-6)
    threshold = np.maximum(noise, floor) * noise_factor

    above = smoothed > threshold[:, None]
    # Sustained-run detection: sample s starts a run of `min_run` Trues.
    run = above.copy()
    for k in range(1, min_run):
        run[:, :-k] &= above[:, k:]
    first = np.argmax(run, axis=1).astype(np.int32)
    found = run[np.arange(n_pings), first]
    first[~found] = -1
    return first


def track_bottom(
    pa: PingArray,
    noise_factor: float = 4.0,
    min_run: int = 3,
    smooth_pings: int = 15,
    max_dev_m: float = 1.5,
) -> BottomTrack:
    """Track the first bottom return on both sides and fuse into one altitude.

    Outliers (deviation > *max_dev_m* from the median-smoothed track) and
    failed pings are replaced by interpolation; if a ping has no valid return
    on either side, the header altitude is used and ``valid`` is False there.
    """
    header_alt = pa.nav["altitude"].astype(np.float32).copy()
    n_pings = pa.n_pings

    estimates: list[np.ndarray] = []
    first_raw: dict[str, np.ndarray] = {}
    for side in ("port", "starboard"):
        if pa.n_samples(side) == 0:
            first_raw[side] = np.full(n_pings, -1, dtype=np.int32)
            continue
        res = pa.slant_resolution(side)
        first = _first_returns_one_side(pa.side(side), noise_factor, min_run)
        first_raw[side] = first
        alt = np.where(first >= 0, (first.astype(np.float32) + 0.5) * res, np.nan)
        estimates.append(alt)

    if estimates:
        stacked = np.vstack(estimates)
        all_nan = np.isnan(stacked).all(axis=0)
        fused = np.full(n_pings, np.nan, dtype=np.float64)
        if not all_nan.all():
            fused[~all_nan] = np.nanmedian(stacked[:, ~all_nan], axis=0)
    else:
        fused = np.full(n_pings, np.nan, dtype=np.float32)

    # Outlier rejection against a median-smoothed reference track.
    finite = np.isfinite(fused)
    valid = finite.copy()
    if finite.any():
        reference = fused.copy()
        reference[~finite] = np.interp(
            np.flatnonzero(~finite), np.flatnonzero(finite), fused[finite]
        )
        reference = median_filter(reference, size=max(3, smooth_pings))
        outlier = np.abs(fused - reference) > max_dev_m
        valid &= ~outlier
        if valid.any():
            fused[~valid] = np.interp(
                np.flatnonzero(~valid), np.flatnonzero(valid), fused[valid]
            )
            # Light final smoothing; median first killed spikes.
            fused = uniform_filter1d(fused, size=max(3, smooth_pings // 2))
        else:
            fused = header_alt.copy()
            valid = np.zeros(n_pings, dtype=bool)
    else:
        fused = header_alt.copy()
        valid = np.zeros(n_pings, dtype=bool)

    # Consistent per-side first-return indices from the fused track.
    first_return: dict[str, np.ndarray] = {}
    for side in ("port", "starboard"):
        res = pa.slant_resolution(side)
        if res <= 0:
            first_return[side] = np.zeros(n_pings, dtype=np.int32)
        else:
            first_return[side] = np.clip(
                np.round(fused / res).astype(np.int32), 0, pa.n_samples(side) - 1
            )

    return BottomTrack(
        altitude_m=fused.astype(np.float32),
        first_return=first_return,
        header_altitude_m=header_alt,
        valid=valid,
    )


def blank_water_column(pa: PingArray, bt: BottomTrack, fill: float = 0.0) -> PingArray:
    """Copy of *pa* with all samples before the first bottom return set to
    *fill*, and nav altitude replaced by the tracked altitude."""
    nav = pa.nav.copy()
    nav["altitude"] = bt.altitude_m
    out = PingArray(
        port=pa.port.copy(),
        starboard=pa.starboard.copy(),
        nav=nav,
        source=pa.source,
        meta={**pa.meta, "water_column_blanked": True},
    )
    for side in ("port", "starboard"):
        img = out.side(side)
        if img.shape[1] == 0:
            continue
        cols = np.arange(img.shape[1])[None, :]
        img[cols < bt.first_return[side][:, None]] = fill
    return out
