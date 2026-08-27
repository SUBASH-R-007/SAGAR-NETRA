"""Geometrically consistent acoustic-shadow rendering for ground-range tiles.

A proud object of height ``H`` insonified by a towfish at altitude ``A``
blocks the ray that grazes its top: everything on the seabed behind the
object receives no energy until that ray touches down again. By similar
triangles (towfish above nadir, object top, seabed), a shadow cast from the
object's far (down-range) edge at absolute ground range ``x_far`` ends at::

    x_end = x_far * A / (A - H)

This is exactly the relation the PhysiCheck module inverts to estimate target
height from shadow length, and the same relation :mod:`sonar_core.synth.scene`
uses when it ray-traces seeded targets — so a shadow rendered here behind a
*pasted* object is indistinguishable, geometrically, from a natively rendered
one, and copy-paste training chips keep the highlight+shadow pair detectors
key on.

Ground-range convention (see ``GroundImage``/``Tile``): rows are pings,
columns are uniform ground range with column 0 nearest nadir on BOTH sides,
so shadows always extend toward INCREASING column and this module only ever
darkens down-range of the footprint.
"""

from __future__ import annotations

import numpy as np

#: Guard against division by zero when ``feather_px`` is 0: a vanishing
#: feather width degenerates the edge ramps into hard steps, which is the
#: intended behaviour for a zero feather.
_MIN_FEATHER_PX: float = 1e-6


def render_shadow(
    tile_img: np.ndarray,
    mask: np.ndarray,
    ground_res: float,
    altitude_m: float | np.ndarray,
    nadir_col0: int,
    height_m: float,
    shadow_level: float = 0.12,
    feather_px: float = 1.0,
    height_cap_frac: float = 0.95,
) -> np.ndarray:
    """Darken the acoustic shadow a pasted footprint would cast; returns a copy.

    For every row (ping) containing footprint pixels, the shadow spans from
    the far edge of the footprint at ``x_far`` to ``x_end = x_far * A /
    (A - min(height_m, height_cap_frac * A))`` and existing pixels there are
    multiplied down to ``shadow_level`` (residual reverberation keeps real
    shadows from being perfectly black). Edges are feathered over
    ``feather_px`` columns because a real shadow boundary is smeared by the
    beam footprint and by the object's rounded silhouette. Rows without
    footprint pixels are returned bit-identical, NaN (out-of-swath) pixels
    stay NaN, and the input array is never modified.

    Parameters
    ----------
    tile_img:
        ``(h, w)`` float ground-range tile (rows = pings, cols = ground range).
    mask:
        ``(h, w)`` bool object footprint within the tile.
    ground_res:
        Metres per ground-range column.
    altitude_m:
        Towfish altitude in metres: scalar, or ``(h,)`` per-row array so a
        heaving towfish casts row-varying shadow lengths.
    nadir_col0:
        Ground-range column offset of tile column 0, so the absolute ground
        range of tile column ``c`` is ``(nadir_col0 + c + 0.5) * ground_res``.
        Tiles cut by the SAHI tiler carry this as ``Tile.col0``.
    height_m:
        Proud height of the pasted object above the seabed, metres.
    shadow_level:
        Multiplicative floor inside the shadow (fraction of the original
        intensity that survives, from volume reverberation and sidelobes).
    feather_px:
        Half-transition width of the shadow edges, in columns. ``0`` gives
        hard edges.
    height_cap_frac:
        An object taller than the towfish altitude would shadow to infinite
        range; heights are capped at this fraction of the altitude, matching
        the scene renderer.

    Notes
    -----
    Each row casts a single shadow from the far-most footprint pixel in that
    row. To composite several separate objects into one tile, call this once
    per object so each casts its own shadow.
    """
    img = np.asarray(tile_img, dtype=np.float32)
    fp = np.asarray(mask, dtype=bool)
    if img.ndim != 2:
        raise ValueError(f"tile_img must be 2-D (h, w), got {img.ndim}-D")
    if fp.shape != img.shape:
        raise ValueError(f"mask shape {fp.shape} != tile shape {img.shape}")
    if ground_res <= 0:
        raise ValueError(f"ground_res must be positive, got {ground_res}")
    if not 0.0 <= shadow_level <= 1.0:
        raise ValueError(f"shadow_level must be in [0, 1], got {shadow_level}")
    if height_m < 0:
        raise ValueError(f"height_m must be non-negative, got {height_m}")
    if feather_px < 0:
        raise ValueError(f"feather_px must be non-negative, got {feather_px}")

    h, w = img.shape
    alt = np.asarray(altitude_m, dtype=np.float64)
    if alt.ndim == 0:
        alt = np.full(h, float(alt))
    elif alt.shape != (h,):
        raise ValueError(f"altitude_m must be scalar or ({h},), got shape {alt.shape}")
    if (alt <= 0).any():
        raise ValueError("altitude_m must be positive everywhere")

    has_obj = fp.any(axis=1)
    if not has_obj.any():
        return img.copy()

    # Far (down-range) footprint edge per row: outer boundary of the last
    # True column. argmax on the column-reversed mask finds it vectorized;
    # rows without footprint yield garbage that is masked out below.
    c_far = (w - 1) - np.argmax(fp[:, ::-1], axis=1)
    x_far = (nadir_col0 + c_far + 1.0) * ground_res
    h_eff = np.minimum(height_m, height_cap_frac * alt)
    x_end = x_far * alt / (alt - h_eff)

    # Absolute ground range at each column centre, broadcast over rows.
    g = (nadir_col0 + np.arange(w, dtype=np.float64) + 0.5) * ground_res
    feather_m = max(feather_px, _MIN_FEATHER_PX) * ground_res
    ramp_in = np.clip((g[None, :] - x_far[:, None]) / feather_m, 0.0, 1.0)
    ramp_out = np.clip((x_end[:, None] - g[None, :]) / feather_m, 0.0, 1.0)
    weight = ramp_in * ramp_out
    weight[~has_obj] = 0.0
    weight[fp] = 0.0  # the footprint itself is highlight, never its own shadow

    shade = 1.0 - (1.0 - shadow_level) * weight
    return (img * shade).astype(np.float32)
