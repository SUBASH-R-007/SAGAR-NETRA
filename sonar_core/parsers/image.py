"""Plain-image adapter: PNG/JPG waterfall crops with declared survey geometry.

Useful for public benchmark datasets (SCTD, KLSG, ...) that ship as images
rather than logs, and for demonstrating the pipeline on a single waterfall
capture.

An image carries no navigation, so the geometry that a real log records per
ping has to be *declared* by the operator instead — the altitude and range
their sonar was set to, and where the line ran. Supplying them unlocks the
full chain: slant-range correction, height from shadow (``H = L*A/R``) and
WGS-84 geotagging. Omit them and the adapter still loads the image, but nav
stays NaN and downstream stages degrade to pixel space.

The synthesised track is an honest fiction and is labelled as one in
``meta["nav_source"]``: a straight line at constant speed and heading from
the declared start. It positions detections relative to that line correctly,
which is what a benchmark image or a demo needs; it is not recorded
navigation and must never be presented as such.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sonar_core.parsers.base import NAV_DTYPE, PingArray, SonarParser, register_parser

#: Metres per degree of latitude — the along-track step is small enough that a
#: local flat-earth advance is exact to well under the position accuracy the
#: reports quote.
EARTH_M_PER_DEG_LAT = 111_320.0

#: Defaults for a declared line when the operator gives a position but no
#: vessel motion: a slow survey speed and a typical side-scan ping interval.
DEFAULT_SPEED_MPS = 2.0
DEFAULT_PING_INTERVAL_S = 0.1


@register_parser
class ImageParser(SonarParser):
    suffixes = (".png", ".jpg", ".jpeg", ".bmp", ".tif")

    def parse(
        self,
        path: Path,
        *,
        combined: bool = True,
        slant_range_m: float = float("nan"),
        altitude_m: float = float("nan"),
        lat: float | None = None,
        lon: float | None = None,
        heading_deg: float = 90.0,
        speed_mps: float = DEFAULT_SPEED_MPS,
        ping_interval_s: float = DEFAULT_PING_INTERVAL_S,
        start_time: float = float("nan"),
        sensor_depth_m: float = float("nan"),
        gain_normalized: bool = True,
        **kwargs: Any,
    ) -> PingArray:
        """Load an image as ping data under declared survey geometry.

        Parameters
        ----------
        combined:
            ``True`` treats the image as a full waterfall (port mirrored on
            the left half, starboard on the right); ``False`` treats the whole
            image as a single starboard channel.
        slant_range_m, altitude_m:
            The sonar's range setting and towfish height above the seabed.
            Both are needed for slant-range correction and for height from
            shadow; without them those stages cannot run.
        lat, lon, heading_deg, speed_mps, ping_interval_s:
            Declared start of the survey line and the vessel motion along it.
            Given a position, a straight constant-heading track is synthesised
            so detections geotag correctly relative to that line.
        start_time:
            UTC epoch seconds of the first ping, for contact timestamps.
        sensor_depth_m:
            Towfish depth below the surface, used with altitude to report
            water depth at each contact.
        gain_normalized:
            Whether the image has already been gain-corrected and contrast
            stretched — true for any waterfall written for display, which is
            what a PNG/JPG almost always is. The pipeline then skips its own
            gain normalization rather than applying it twice. Set False for an
            export of raw uncorrected amplitudes.
        """
        img = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        n_pings, width = img.shape

        if combined and width >= 2:
            half = width // 2
            port = img[:, :half][:, ::-1]  # un-mirror back to nadir-first order
            starboard = img[:, half:]
        else:
            port = np.zeros((n_pings, 0), dtype=np.float32)
            starboard = img

        nav = np.zeros(n_pings, dtype=NAV_DTYPE)
        nav["altitude"] = altitude_m
        nav["slant_range"] = slant_range_m
        nav["sound_velocity"] = 1500.0
        nav["sensor_depth"] = sensor_depth_m
        nav["heading"] = heading_deg
        nav["speed"] = speed_mps
        nav["layback"] = 0.0

        if np.isfinite(start_time):
            nav["time"] = start_time + np.arange(n_pings) * ping_interval_s
        else:
            nav["time"] = np.nan

        has_track = lat is not None and lon is not None
        if has_track:
            # Straight line at constant heading: advance each ping by the
            # along-track step, resolved into lat/lon at the start latitude.
            step_m = float(speed_mps) * float(ping_interval_s)
            along = np.arange(n_pings, dtype=np.float64) * step_m
            bearing = np.deg2rad(float(heading_deg))
            dlat = along * np.cos(bearing) / EARTH_M_PER_DEG_LAT
            lat0 = float(lat)
            nav["lat"] = lat0 + dlat
            nav["lon"] = float(lon) + along * np.sin(bearing) / (
                EARTH_M_PER_DEG_LAT * np.cos(np.deg2rad(lat0))
            )
        else:
            nav["lat"] = np.nan
            nav["lon"] = np.nan

        return PingArray(
            port=port,
            starboard=starboard,
            nav=nav,
            source=str(path),
            meta={
                "format": "image",
                "sonar_name": "unknown",
                "has_nav": bool(has_track),
                "gain_normalized": bool(gain_normalized),
                # Never let a synthesised line be mistaken for a recorded one.
                "nav_source": "declared-line" if has_track else "none",
                "declared_geometry": {
                    "altitude_m": altitude_m,
                    "slant_range_m": slant_range_m,
                    "heading_deg": heading_deg,
                    "speed_mps": speed_mps,
                    "ping_interval_s": ping_interval_s,
                },
            },
        )
