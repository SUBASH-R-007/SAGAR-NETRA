"""SAHI-style overlapping tiling of ground-range imagery.

Detectors ingest fixed-size chips, but a side-scan survey strip is arbitrarily
long along-track (one row per ping) and its swath width is set by the range
scale, so each side of a :class:`~sonar_core.preprocess.slant_range.GroundImage`
must be cut into tiles. Debris targets are small — a drum can be under ten
pixels across at typical range scales — and must not be split by every tile
boundary, so tiles overlap: any object smaller than the overlap margin appears
whole in at least one tile. The last tile of each axis is *shifted back* to end
exactly at the image edge rather than padded: padding would inject synthetic
texture that skews per-tile intensity statistics (which detectors normalize
against) and creates phantom edges to fire on, while shifting keeps every pixel
real and loses none.

Coordinate bookkeeping is exact. Each tile records the global (ping index,
ground-range column) of its top-left pixel, so any detection in a tile maps
back to a ping and ground-range column, and from there via
:meth:`GroundImage.to_slant_sample` to the fractional raw slant sample — and
thus to the NAV record needed to georeference the find.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sonar_core.preprocess.slant_range import GroundImage


@dataclass(frozen=True)
class Tile:
    """One detector-ready chip plus its exact placement in the ground image.

    ``image`` is a C-contiguous float32 *copy* of the source slice, so a
    detector may normalize or augment it in place without corrupting the
    mosaic. NaN pixels still mark out-of-swath fill (ground range beyond what
    that ping's slant range and altitude can reach).
    """

    side: str  # "port" | "starboard"
    row0: int  # global ping index of tile row 0
    col0: int  # global ground-range column of tile column 0
    image: np.ndarray  # (h, w) float32, C-contiguous copy
    index: int  # sequential tile id, unique within one tiling call

    def to_global(
        self, r: np.ndarray | int, c: np.ndarray | int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tile-local pixel -> global (ping index, ground column). Vectorized."""
        return np.asarray(r) + self.row0, np.asarray(c) + self.col0

    def from_global(
        self, ping: np.ndarray | int, col: np.ndarray | int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Global (ping, ground column) -> tile-local pixel; exact inverse of
        :meth:`to_global`. Raises ``ValueError`` if any coordinate falls
        outside this tile's footprint."""
        r = np.asarray(ping) - self.row0
        c = np.asarray(col) - self.col0
        h, w = self.image.shape
        if np.any(r < 0) or np.any(r >= h) or np.any(c < 0) or np.any(c >= w):
            raise ValueError(
                f"(ping, col) outside tile {self.index}: rows "
                f"[{self.row0}, {self.row0 + h}), cols [{self.col0}, {self.col0 + w})"
            )
        return r, c

    def contains(self, ping: np.ndarray | int, col: np.ndarray | int) -> bool:
        """True when every given global (ping, col) lies inside this tile."""
        r = np.asarray(ping) - self.row0
        c = np.asarray(col) - self.col0
        h, w = self.image.shape
        return bool(np.all((r >= 0) & (r < h) & (c >= 0) & (c < w)))


def _axis_origins(extent: int, tile_size: int, stride: int) -> tuple[np.ndarray, int]:
    """SAHI origins along one axis, plus the tile extent in that axis.

    Origins step by *stride*; if that leaves a remainder, one final origin is
    added at ``extent - tile_size`` so the last tile ends exactly at the edge.
    An axis shorter than *tile_size* yields a single full-extent tile.
    """
    if extent <= tile_size:
        return np.array([0], dtype=np.int64), extent
    origins = list(range(0, extent - tile_size + 1, stride))
    if origins[-1] + tile_size < extent:
        origins.append(extent - tile_size)
    return np.asarray(origins, dtype=np.int64), tile_size


def tile_image(
    img: np.ndarray,
    side: str,
    tile_size: int = 1024,
    overlap: float = 0.25,
    min_content: float = 0.0,
    start_index: int = 0,
) -> list[Tile]:
    """Cut one side of a ground image into overlapping SAHI-style tiles.

    Tile origins step by ``stride = round(tile_size * (1 - overlap))``; the
    last tile of each axis is shifted back to end exactly at the image edge
    (never padded, no pixels lost). An image smaller than *tile_size* in an
    axis yields one tile spanning that full axis, so tiles may be non-square
    at edges and for short test surveys.

    Drop policy — lossless by default: a tile with *zero* finite pixels is
    always dropped, because it is pure out-of-swath fill with no acoustic
    content. Tiles whose finite-pixel fraction is below *min_content* are
    also dropped; the default of 0.0 disables that extra filter so no finite
    pixel can ever lose tile coverage. Raise it only when sparse swath-edge
    slivers are an acceptable loss for the detector.

    *start_index* offsets the sequential tile ids so successive calls (e.g.
    port then starboard) share one id space.
    """
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"img must be 2-D (n_pings, n_cols), got {img.ndim}-D")
    if side not in ("port", "starboard"):
        raise KeyError(f"side must be 'port' or 'starboard', got {side!r}")
    if tile_size < 1:
        raise ValueError(f"tile_size must be >= 1, got {tile_size}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    n_rows, n_cols = img.shape
    if n_rows == 0 or n_cols == 0:
        return []
    stride = max(int(round(tile_size * (1.0 - overlap))), 1)
    row_origins, tile_h = _axis_origins(n_rows, tile_size, stride)
    col_origins, tile_w = _axis_origins(n_cols, tile_size, stride)

    tiles: list[Tile] = []
    index = start_index
    for r0 in row_origins:
        for c0 in col_origins:
            view = img[r0 : r0 + tile_h, c0 : c0 + tile_w]
            n_finite = int(np.count_nonzero(np.isfinite(view)))
            if n_finite == 0 or n_finite < min_content * view.size:
                continue
            tiles.append(
                Tile(
                    side=side,
                    row0=int(r0),
                    col0=int(c0),
                    image=np.array(view, dtype=np.float32, order="C", copy=True),
                    index=index,
                )
            )
            index += 1
    return tiles


def tiles_for_ground_image(
    gi: GroundImage,
    tile_size: int = 1024,
    overlap: float = 0.25,
    min_content: float = 0.0,
) -> list[Tile]:
    """Tile both sides of *gi* with ids unique across sides (port first).

    Each returned tile's ``(row0, col0)`` are directly usable with the ground
    image's inverse mapping: ``tile.to_global(r, c)`` gives the (ping, column)
    that :meth:`GroundImage.to_slant_sample` turns into a raw slant sample.
    """
    tiles = tile_image(
        gi.port, "port", tile_size=tile_size, overlap=overlap, min_content=min_content
    )
    tiles += tile_image(
        gi.starboard,
        "starboard",
        tile_size=tile_size,
        overlap=overlap,
        min_content=min_content,
        start_index=len(tiles),
    )
    return tiles
