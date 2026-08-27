"""M2 preprocessing orchestrator: raw slant-range pings to detector-ready tiles.

The stage order is physics-driven, not arbitrary:

1. **Bottom tracking** runs first because every later stage needs to know
   where the water column ends: EGN must exclude water-column samples from
   its statistics and the slant correction needs a per-ping altitude.
2. **EGN in the slant domain.** The residual time-varied-gain error is a
   function of two-way travel time — of the *slant* sample index — and is
   shared by every ping. After ground projection each ping is re-mapped by
   its own altitude, so range bins would no longer line up across pings and
   the per-range median would smear the very gain curve it estimates.
3. **Water-column blanking** before ground projection, so near-nadir ambient
   noise is never interpolated onto real seabed columns.
4. **Slant-to-ground** resamples onto a uniform ground grid. A snapshot is
   kept as ``ground_raw`` because PhysiCheck's shadow physics (target height
   from shadow length) needs true edge positions and unsmoothed radiometry.
5. **Despeckle before CLAHE**: CLAHE raises local contrast indiscriminately,
   so speckle variance must be reduced first or it would be amplified into
   salt-and-pepper noise.
6. **Tiling** last, on the enhanced image the detector actually ingests.

All tunables live in :data:`DEFAULTS`; ``configs/preprocess.yaml`` mirrors it
line for line and user config is deep-merged on top, so YAML and code can
never disagree about a default. Mosaic input (``pa.meta["ground_range"]``
truthy) is already in the ground domain: the slant-domain stages are skipped
and the enhancement/tiling tail still runs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from sonar_core.parsers.base import PingArray
from sonar_core.preprocess.bottom_track import BottomTrack, blank_water_column, track_bottom
from sonar_core.preprocess.clahe import clahe
from sonar_core.preprocess.despeckle import despeckle
from sonar_core.preprocess.egn import empirical_gain_normalize
from sonar_core.preprocess.slant_range import GroundImage, slant_to_ground
from sonar_core.preprocess.tiler import Tile, tiles_for_ground_image

SIDES: tuple[str, str] = ("port", "starboard")

#: Ground resolution assumed for a mosaic input that carries no resolution
#: metadata and no usable per-ping range records — 1 m/pixel is the
#: conventional quick-look scale; real mosaics should set ``meta["ground_res"]``.
FALLBACK_GROUND_RES: float = 1.0

#: Single source of truth for every tunable of every stage.
#: ``configs/preprocess.yaml`` mirrors this structure (with units in its
#: comments); user config dicts are deep-merged over it by :func:`preprocess`.
DEFAULTS: dict[str, Any] = {
    "bottom_track": {
        "noise_factor": 4.0,  # first-return threshold, x water-column noise level
        "min_run": 3,  # consecutive above-threshold samples to accept a return
        "smooth_pings": 15,  # median window (pings) for the outlier reference track
        "max_dev_m": 1.5,  # outlier rejection band around the reference, metres
    },
    "egn": {
        "enabled": True,
        "n_bins": 256,  # range bins pooled for the gain estimate
        "eps_frac": 0.01,  # dead-bin gain clamp, fraction of the median gain
        "nadir_guard": 8,  # samples after first return excluded from gain stats
    },
    "water_column": {
        "fill": 0.0,  # intensity written over pre-bottom (water-column) samples
    },
    "slant_range": {
        "ground_res": None,  # m/column; None -> finest side's slant resolution
    },
    "despeckle": {
        "enabled": True,
        "method": "lee",  # "lee" (MMSE speckle filter) | "median" (impulse-only)
        "size": 5,  # filter window side, pixels
    },
    "clahe": {
        "enabled": True,
        "clip_limit": 2.5,  # per-tile histogram slope cap (x uniform histogram)
        "tile_grid": (8, 8),  # (tiles across range, tiles along track)
        "p_low": 1.0,  # percentile mapped to black before CLAHE
        "p_high": 99.7,  # percentile mapped to white before CLAHE
    },
    "tiler": {
        "tile_size": 512,  # detector chip side, pixels
        "overlap": 0.25,  # overlap fraction so small debris is never always split
        "min_content": 0.0,  # min finite-pixel fraction to keep a tile (0 = lossless)
    },
}


@dataclass
class PreprocessResult:
    """Everything the M2 pipeline produces for one survey.

    ``ground`` is the enhanced (despeckled + CLAHE) image, finite values in
    ``[0, 1]`` with NaN beyond the swath — the detector's input. ``ground_raw``
    is the slant-corrected, EGN-normalized image *before* any enhancement:
    shadow-length height estimation must measure edges the smoothing filters
    have not moved and amplitudes CLAHE has not remapped. The two never share
    array memory.
    """

    bottom: BottomTrack
    ground: GroundImage  # despeckled + CLAHE-enhanced, values [0,1], NaN swath
    ground_raw: GroundImage  # slant-corrected but unenhanced (for physics/shadow work)
    tiles: list[Tile]
    egn_gain: dict[str, np.ndarray]  # side -> per-sample gain curve
    timings: dict[str, float]  # seconds per stage


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively overlay *override* onto *base*; neither input is mutated."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _safe_progress(
    progress: Callable[[str, float], None] | None, stage: str, fraction: float
) -> None:
    """Invoke the progress callback; a broken observer must never abort a run."""
    if progress is None:
        return
    try:
        progress(stage, fraction)
    except Exception:  # noqa: BLE001 - observer failures are deliberately swallowed
        pass


def _copy_ground(gi: GroundImage) -> GroundImage:
    """Independent deep copy so enhancement can never alias raw physics data."""
    return GroundImage(
        port=gi.port.copy(),
        starboard=gi.starboard.copy(),
        ground_res=gi.ground_res,
        altitude_m=gi.altitude_m.copy(),
        slant_res=dict(gi.slant_res),
        nav=gi.nav.copy(),
        meta=dict(gi.meta),
    )


def _identity_gains(pa: PingArray) -> dict[str, np.ndarray]:
    """Unit gain per side: the curve that was effectively applied when EGN is off."""
    return {side: np.ones(pa.n_samples(side), dtype=np.float64) for side in SIDES}


def _mosaic_ground_res(pa: PingArray, cfg: dict[str, Any]) -> float:
    """Best available metres-per-column for an already-ground-range input.

    Preference order: explicit config, the mosaic's own ``meta["ground_res"]``,
    the GeoTIFF adapter's ``meta["pixel_size"]``, the finest per-side sample
    spacing derivable from the nav records, and finally
    :data:`FALLBACK_GROUND_RES` so a metadata-poor mosaic still flows through
    enhancement and tiling instead of failing.
    """
    for candidate in (
        cfg["slant_range"]["ground_res"],
        pa.meta.get("ground_res"),
        pa.meta.get("pixel_size"),
    ):
        if candidate:
            return float(candidate)
    usable = [pa.slant_resolution(side) for side in SIDES if pa.slant_resolution(side) > 0]
    return float(min(usable)) if usable else FALLBACK_GROUND_RES


def _mosaic_passthrough(pa: PingArray, cfg: dict[str, Any]) -> tuple[BottomTrack, GroundImage]:
    """Wrap ground-range input as a :class:`GroundImage` with a header-only
    bottom track (no slant geometry exists to re-track)."""
    alt = pa.nav["altitude"].astype(np.float32)
    bt = BottomTrack(
        altitude_m=alt.copy(),
        first_return={side: np.zeros(pa.n_pings, dtype=np.int32) for side in SIDES},
        header_altitude_m=alt.copy(),
        valid=np.zeros(pa.n_pings, dtype=bool),
    )
    gi = GroundImage(
        port=pa.port.copy(),
        starboard=pa.starboard.copy(),
        ground_res=_mosaic_ground_res(pa, cfg),
        altitude_m=alt.copy(),
        slant_res={side: pa.slant_resolution(side) for side in SIDES},
        nav=pa.nav.copy(),
        meta={**pa.meta, "ground_range": True},
    )
    return bt, gi


def preprocess(
    pa: PingArray,
    config: dict[str, Any] | None = None,
    progress: Callable[[str, float], None] | None = None,
) -> PreprocessResult:
    """Run the full M2 chain: track -> EGN -> blank -> ground -> enhance -> tile.

    Parameters
    ----------
    pa:
        Raw slant-range survey, or a mosaic already in ground range
        (``pa.meta["ground_range"]`` truthy), in which case the slant-domain
        stages (bottom tracking, EGN, blanking, slant correction) are skipped
        and the input is wrapped as the ground image directly. The input is
        never mutated.
    config:
        Partial override of :data:`DEFAULTS`, deep-merged, mirroring
        ``configs/preprocess.yaml``. Per-stage ``enabled`` flags switch EGN,
        despeckle and CLAHE off individually.
    progress:
        Optional observer called at each stage boundary with
        ``(stage_name, fraction)`` where the fraction is the share of enabled
        stages already completed — monotonically nondecreasing in ``[0, 1]``
        — and finally with ``("done", 1.0)``. Exceptions it raises are
        swallowed: a UI glitch must never kill a survey run.

    Returns
    -------
    PreprocessResult
        See the dataclass docstring; ``timings`` holds wall-clock seconds
        keyed by stage name for every stage that actually ran.
    """
    cfg = _deep_merge(DEFAULTS, config or {})
    is_mosaic = bool(pa.meta.get("ground_range"))

    stage_names: list[str] = []
    if not is_mosaic:
        stage_names.append("track_bottom")
        if cfg["egn"]["enabled"]:
            stage_names.append("egn")
        stage_names += ["blank_water_column", "slant_to_ground"]
    if cfg["despeckle"]["enabled"]:
        stage_names.append("despeckle")
    if cfg["clahe"]["enabled"]:
        stage_names.append("clahe")
    stage_names.append("tile")
    n_stages = len(stage_names)

    timings: dict[str, float] = {}
    completed = 0

    def begin(name: str) -> float:
        _safe_progress(progress, name, completed / n_stages)
        return time.perf_counter()

    def finish(name: str, t0: float) -> None:
        nonlocal completed
        timings[name] = time.perf_counter() - t0
        completed += 1

    if is_mosaic:
        bt, gi = _mosaic_passthrough(pa, cfg)
        egn_gain = _identity_gains(pa)
    else:
        t0 = begin("track_bottom")
        bt = track_bottom(pa, **cfg["bottom_track"])
        finish("track_bottom", t0)

        work = pa
        if cfg["egn"]["enabled"]:
            t0 = begin("egn")
            egn_kwargs = {k: v for k, v in cfg["egn"].items() if k != "enabled"}
            normalized: dict[str, np.ndarray] = {}
            egn_gain = {}
            for side in SIDES:
                normalized[side], egn_gain[side] = empirical_gain_normalize(
                    work.side(side), first_return=bt.first_return[side], **egn_kwargs
                )
            work = PingArray(
                port=normalized["port"],
                starboard=normalized["starboard"],
                nav=work.nav.copy(),
                source=work.source,
                meta={**work.meta, "egn_normalized": True},
            )
            finish("egn", t0)
        else:
            egn_gain = _identity_gains(pa)

        t0 = begin("blank_water_column")
        work = blank_water_column(work, bt, fill=cfg["water_column"]["fill"])
        finish("blank_water_column", t0)

        t0 = begin("slant_to_ground")
        gi = slant_to_ground(work, bt, ground_res=cfg["slant_range"]["ground_res"])
        finish("slant_to_ground", t0)

    # Snapshot BEFORE enhancement: shadow physics needs unsmoothed radiometry
    # and edge positions no filter has moved. Deep copy so the two ground
    # images can never share mutated arrays.
    ground_raw = _copy_ground(gi)

    port, starboard = gi.port, gi.starboard
    if cfg["despeckle"]["enabled"]:
        t0 = begin("despeckle")
        ds_kwargs = {k: v for k, v in cfg["despeckle"].items() if k != "enabled"}
        port = despeckle(port, **ds_kwargs)
        starboard = despeckle(starboard, **ds_kwargs)
        finish("despeckle", t0)
    if cfg["clahe"]["enabled"]:
        t0 = begin("clahe")
        cl_kwargs = {k: v for k, v in cfg["clahe"].items() if k != "enabled"}
        cl_kwargs["tile_grid"] = tuple(int(v) for v in cl_kwargs["tile_grid"])
        port = clahe(port, **cl_kwargs)
        starboard = clahe(starboard, **cl_kwargs)
        finish("clahe", t0)
    if port is gi.port:  # no enhancement ran: still hand out independent arrays
        port, starboard = port.copy(), starboard.copy()

    ground = GroundImage(
        port=port,
        starboard=starboard,
        ground_res=gi.ground_res,
        altitude_m=gi.altitude_m.copy(),
        slant_res=dict(gi.slant_res),
        nav=gi.nav.copy(),
        meta={
            **gi.meta,
            "despeckled": bool(cfg["despeckle"]["enabled"]),
            "clahe": bool(cfg["clahe"]["enabled"]),
        },
    )

    t0 = begin("tile")
    tiles = tiles_for_ground_image(ground, **cfg["tiler"])
    finish("tile", t0)

    _safe_progress(progress, "done", 1.0)
    return PreprocessResult(
        bottom=bt,
        ground=ground,
        ground_raw=ground_raw,
        tiles=tiles,
        egn_gain=egn_gain,
        timings=timings,
    )
