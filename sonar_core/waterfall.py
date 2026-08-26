"""Waterfall rendering: :class:`PingArray` -> displayable image, with exact
pixel <-> (side, ping, sample) bookkeeping.

Layout convention (matches common survey software): port is mirrored so its
far range sits at the image's left edge, both nadirs meet at the centreline,
and starboard's far range is at the right edge. Rows are ping indices
(row 0 = first ping recorded).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from sonar_core.parsers.base import PingArray


@dataclass(frozen=True)
class WaterfallLayout:
    """Column bookkeeping for a combined port|starboard waterfall image."""

    n_port: int
    n_starboard: int

    @property
    def width(self) -> int:
        return self.n_port + self.n_starboard

    def col_to_sample(self, col: int) -> tuple[str, int]:
        """Image column -> (side, sample index), sample 0 at nadir."""
        if not 0 <= col < self.width:
            raise IndexError(f"column {col} outside waterfall width {self.width}")
        if col < self.n_port:
            return "port", self.n_port - 1 - col
        return "starboard", col - self.n_port

    def sample_to_col(self, side: str, sample_idx: int) -> int:
        """(side, sample index) -> image column. Inverse of :meth:`col_to_sample`."""
        if side == "port":
            if not 0 <= sample_idx < self.n_port:
                raise IndexError(f"port sample {sample_idx} outside 0..{self.n_port - 1}")
            return self.n_port - 1 - sample_idx
        if side == "starboard":
            if not 0 <= sample_idx < self.n_starboard:
                raise IndexError(
                    f"starboard sample {sample_idx} outside 0..{self.n_starboard - 1}"
                )
            return self.n_port + sample_idx
        raise KeyError(f"side must be 'port' or 'starboard', got {side!r}")


def layout_for(pa: PingArray) -> WaterfallLayout:
    return WaterfallLayout(n_port=pa.n_samples("port"), n_starboard=pa.n_samples("starboard"))


def combine(pa: PingArray) -> np.ndarray:
    """Stack both sides into one ``(n_pings, n_port + n_stbd)`` float32 image."""
    return np.hstack([pa.port[:, ::-1], pa.starboard]).astype(np.float32, copy=False)


def normalize_u8(
    img: np.ndarray, p_low: float = 1.0, p_high: float = 99.5
) -> np.ndarray:
    """Percentile-stretch a float image to uint8 for display.

    Percentiles are computed over finite, positive-signal pixels so the black
    water-column gap does not compress the seabed's dynamic range.
    """
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(finite, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    out = np.nan_to_num(out, nan=0.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def save_waterfall_png(
    pa: PingArray,
    path: str | Path,
    p_low: float = 1.0,
    p_high: float = 99.5,
    max_height: int | None = None,
) -> Path:
    """Render *pa* to an 8-bit grayscale PNG at *path*.

    ``max_height`` optionally decimates pings (never samples) for quick-look
    exports of very long surveys; detection always runs on full data.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = combine(pa)
    if max_height is not None and img.shape[0] > max_height:
        step = int(np.ceil(img.shape[0] / max_height))
        img = img[::step]
    Image.fromarray(normalize_u8(img, p_low, p_high), mode="L").save(path)
    return path
