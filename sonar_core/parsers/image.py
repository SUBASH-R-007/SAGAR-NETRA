"""Plain-image adapter: PNG/JPG waterfall crops without navigation.

Useful for public benchmark datasets (SCTD, KLSG, ...) that ship as images.
Navigation fields are NaN; downstream stages that need geo/physics degrade
gracefully (detections are reported in pixel space only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sonar_core.parsers.base import NAV_DTYPE, PingArray, SonarParser, register_parser


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
        **kwargs: Any,
    ) -> PingArray:
        """Load an image as ping data.

        ``combined=True`` treats the image as a full waterfall (port mirrored
        on the left half, starboard on the right); ``False`` treats the whole
        image as a single starboard channel.
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
        for name in ("time", "lat", "lon", "heading"):
            nav[name] = np.nan
        nav["altitude"] = altitude_m
        nav["slant_range"] = slant_range_m
        nav["sound_velocity"] = 1500.0

        return PingArray(
            port=port,
            starboard=starboard,
            nav=nav,
            source=str(path),
            meta={"format": "image", "sonar_name": "unknown", "has_nav": False},
        )
