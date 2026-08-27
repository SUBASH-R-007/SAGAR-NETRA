"""PS-named acquisition-artifact augmentations for ground-range training chips.

The problem statement calls out towfish motion (heave, pitch, roll), ping
dropout and resolution variation as the artifacts a fielded detector must
tolerate, so simulating them at training time is the cheapest robustness win
available (strategy 2.2 Step E). Each transform below models one artifact at
chip level in pure NumPy, driven by an explicit
:class:`numpy.random.Generator` so every draw is reproducible.

Chip geometry contract (identical to the rest of the synth stack): rows are
along-track pings, columns are ground range with column 0 at nadir on BOTH
sides, and acoustic shadows always extend toward increasing column. Every
transform preserves that invariant — none mirrors, reverses or reorders the
column axis:

* :func:`heave_banding` — vertical towfish motion modulates ensonification
  and residual TVG ping by ping, printing horizontal intensity bands.
  Modelled as a smooth ROW-wise multiplicative gain profile; pixels never
  move.
* :func:`pitch_stretch` — pitch (and tow-speed) variation changes along-track
  sampling density. Modelled as a linear resample of the ROW axis only,
  cropped or reflect-padded back to shape; columns are untouched.
* :func:`roll_shear` — roll tilts the transducer so the across-track mapping
  drifts slowly along the survey. Modelled as a per-row column shift linear
  in row index, bounded small (see the function docstring) so highlight-to-
  shadow adjacency survives.
* :func:`ping_dropout` — lost pings / telemetry gaps. Full or partial row
  stripes forced to 0, the blanked-water level a real recorder writes.
* :func:`resolution_jitter` — range-setting / transducer resolution
  differences. Down- then up-sample each axis; the round trip is a
  coordinate identity (endpoints map to endpoints), so only frequency
  content changes and no pixel is displaced.

Label safety: :func:`pitch_stretch` and :func:`roll_shear` MOVE pixels, so a
box drawn on the input no longer wraps the target on the output. Rather than
carry a box transform through every consumer, the label-bearing dataset path
(:func:`tridentnet.data.build_synthetic_dataset`) restricts itself to
:data:`LABEL_SAFE_ARTIFACTS` — the three transforms that displace no pixel —
while the image-only training pipeline
(:func:`sonar_core.synth.augment.train_augment`) composes the full set.

All transforms take and return 2-D ``(rows, cols)`` arrays, preserve shape
and dtype (integer dtypes are computed in float32 and rounded/clipped back),
and never modify their input in place.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def _to_float(img: np.ndarray) -> np.ndarray:
    """Working copy in float32 — always a copy, so inputs are never mutated."""
    return np.asarray(img).astype(np.float32, copy=True)


def _restore_dtype(out: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast back to the caller's dtype; integer dtypes are rounded and clipped
    to the dtype range (a >1.0 heave gain must saturate, not wrap)."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(out), info.min, info.max).astype(dtype)
    return out.astype(dtype, copy=False)


def _resample_axis(arr: np.ndarray, new_len: int, axis: int) -> np.ndarray:
    """Linear resample along *axis* to *new_len* samples.

    Endpoints map to endpoints (``linspace(0, n-1, m)``), so the round trip
    ``n -> m -> n`` is a coordinate IDENTITY: position ``j`` samples the
    m-grid at ``j * (m-1)/(n-1)``, which sits at original coordinate ``j``
    again. Only interpolation loss (frequency content) remains — the property
    that makes :func:`resolution_jitter` label-safe.
    """
    old_len = arr.shape[axis]
    if new_len == old_len:
        return arr.copy()
    moved = np.moveaxis(arr, axis, 0)
    if old_len == 1:
        return np.moveaxis(np.repeat(moved, new_len, axis=0), 0, axis)
    pos = np.linspace(0.0, old_len - 1.0, new_len)
    i0 = np.floor(pos).astype(np.intp)
    i1 = np.minimum(i0 + 1, old_len - 1)
    w = (pos - i0).astype(np.float32).reshape(-1, *([1] * (moved.ndim - 1)))
    res = moved[i0] * (1.0 - w) + moved[i1] * w
    return np.moveaxis(res, 0, axis)


def heave_banding(
    img: np.ndarray,
    rng: np.random.Generator,
    *,
    max_gain: float = 0.25,
    band_period_rows: tuple[float, float] = (8.0, 40.0),
) -> np.ndarray:
    """Row-wise multiplicative gain banding from vertical towfish motion.

    Heave changes altitude ping to ping, so ensonification level and the TVG
    residue oscillate along-track: the waterfall shows horizontal bright/dark
    bands. Modelled as a smooth per-row gain — a sinusoid at a random period
    drawn from *band_period_rows* mixed with box-smoothed noise (real heave is
    quasi-periodic swell plus turbulence, not a clean tone), normalized so the
    gain stays within ``1 +/- max_gain``. Purely photometric per row: no pixel
    moves and each row is scaled by one positive constant, so within-row
    column ordering (highlight before shadow) is preserved exactly.
    """
    out = _to_float(img)
    n_rows = out.shape[0]
    period = float(rng.uniform(*band_period_rows))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    rows = np.arange(n_rows, dtype=np.float32)
    sine = np.sin(2.0 * np.pi * rows / period + phase)

    noise = rng.standard_normal(n_rows).astype(np.float32)
    win = max(3, int(round(period / 2.0)))
    smooth = np.convolve(noise, np.ones(win, dtype=np.float32) / win, mode="same")
    peak = float(np.max(np.abs(smooth)))
    if peak > 0.0:
        smooth /= peak

    mix = float(rng.uniform(0.3, 0.7))
    profile = mix * sine + (1.0 - mix) * smooth
    profile /= max(float(np.max(np.abs(profile))), 1e-6)
    amp = float(rng.uniform(0.25, 1.0)) * max_gain
    gain = 1.0 + amp * profile
    shaped = gain.reshape(-1, *([1] * (out.ndim - 1)))
    return _restore_dtype(out * shaped, img.dtype)


def pitch_stretch(
    img: np.ndarray, rng: np.random.Generator, *, max_frac: float = 0.15
) -> np.ndarray:
    """Along-track stretch/compress from pitch and tow-speed variation.

    Pitch (and speed-over-ground changes) alter the along-track distance
    covered per ping, so the same seabed renders taller or shorter in rows.
    The ROW axis is linearly resampled by a factor in ``1 +/- max_frac`` and
    restored to shape: a stretch is centre-cropped, a compression is
    reflect-padded (a reflected ping block is itself a valid sonar image — it
    equals a locally reversed ping order — unlike constant black bars, which
    would add phantom edges). Columns are never touched, so across-track
    shadow geometry is exactly preserved. NOT label-safe: row positions move.
    """
    out = _to_float(img)
    n_rows = out.shape[0]
    factor = float(rng.uniform(1.0 - max_frac, 1.0 + max_frac))
    m = max(2, int(round(n_rows * factor)))
    if m == n_rows:
        return _restore_dtype(out, img.dtype)
    resampled = _resample_axis(out, m, axis=0)
    if m > n_rows:
        off = (m - n_rows) // 2
        resampled = resampled[off : off + n_rows]
    else:
        pad_lo = (n_rows - m) // 2
        pad_hi = n_rows - m - pad_lo
        width = [(pad_lo, pad_hi)] + [(0, 0)] * (out.ndim - 1)
        resampled = np.pad(resampled, width, mode="reflect")
    return _restore_dtype(resampled, img.dtype)


def roll_shear(
    img: np.ndarray, rng: np.random.Generator, *, max_px: float = 6.0
) -> np.ndarray:
    """Small horizontal shear from slow roll of the towfish.

    Roll tilts the transducer, drifting the slant-to-ground mapping slowly
    along the survey: each ping's columns shift by a small, slowly varying
    amount. Modelled as a per-row column shift linear in row index, ramping
    from ``-s/2`` on the first row to ``+s/2`` on the last with
    ``|s| <= max_px``.

    The bound matters: no single row moves more than ``max_px / 2`` columns,
    and — because a target and its shadow share rows — the DIFFERENTIAL shift
    across a target's along-track extent of ``k`` rows is only
    ``k * |s| / n_rows`` (< 1 px for a 40-row target on a 256-row chip at the
    default bound), so the highlight-then-shadow adjacency that PhysiCheck
    and the detector key on survives intact. Each row shifts by one constant
    (linear interpolation, edge columns clamped, never wrapped — wrapping
    would teleport nadir pixels down-range), so within-row column ordering is
    preserved. NOT label-safe: column positions move.
    """
    out = _to_float(img)
    n_rows, n_cols = out.shape[0], out.shape[1]
    s = float(rng.uniform(-max_px, max_px))
    shifts = s * (np.arange(n_rows, dtype=np.float32) / max(n_rows - 1, 1) - 0.5)
    src = np.arange(n_cols, dtype=np.float32)[np.newaxis, :] - shifts[:, np.newaxis]
    src = np.clip(src, 0.0, n_cols - 1.0)
    i0 = np.floor(src).astype(np.intp)
    i1 = np.minimum(i0 + 1, n_cols - 1)
    w = (src - i0).astype(np.float32)
    rows_idx = np.arange(n_rows, dtype=np.intp)[:, np.newaxis]
    sheared = out[rows_idx, i0] * (1.0 - w) + out[rows_idx, i1] * w
    return _restore_dtype(sheared, img.dtype)


def ping_dropout(
    img: np.ndarray,
    rng: np.random.Generator,
    *,
    max_rows_frac: float = 0.06,
    partial_p: float = 0.5,
) -> np.ndarray:
    """Blank random ping stripes to 0, the blanked-water level.

    Trigger glitches and telemetry gaps lose whole pings (the recorder writes
    zeros) or truncate them mid-return. At most ``max_rows_frac`` of the rows
    are blanked, in stripes of 1-3 consecutive rows; each stripe is, with
    probability *partial_p*, PARTIAL — only a contiguous column interval is
    zeroed (a mid-ping dropout) — and otherwise a full row. Stripes may
    overlap, so the blanked fraction is an upper bound. Zeroing removes
    signal but moves nothing, so labels stay valid (a small occluded band is
    exactly the robustness being trained).
    """
    out = _to_float(img)
    n_rows, n_cols = out.shape[0], out.shape[1]
    budget = max(1, int(round(max_rows_frac * n_rows)))
    n_drop = int(rng.integers(1, budget + 1))
    dropped = 0
    while dropped < n_drop:
        h = int(min(rng.integers(1, 4), n_drop - dropped))
        r0 = int(rng.integers(0, n_rows - h + 1))
        if rng.random() < partial_p:
            width = int(rng.integers(max(1, n_cols // 5), n_cols + 1))
            c0 = int(rng.integers(0, n_cols - width + 1))
            out[r0 : r0 + h, c0 : c0 + width] = 0.0
        else:
            out[r0 : r0 + h, :] = 0.0
        dropped += h
    return _restore_dtype(out, img.dtype)


def resolution_jitter(
    img: np.ndarray,
    rng: np.random.Generator,
    *,
    scale: tuple[float, float] = (0.6, 1.4),
) -> np.ndarray:
    """Vary effective pixel resolution by a down-then-up resample round trip.

    Different range settings, frequencies and towfish models sample the same
    seabed at different resolutions per axis. Each axis is independently
    resampled to a factor drawn from *scale* and straight back to the
    original length. Factors < 1 discard high frequencies (a coarser sonar);
    factors > 1 are near-identity. Because :func:`_resample_axis` maps
    endpoints to endpoints, the round trip is a coordinate identity — no
    pixel is displaced, only sharpness changes — which makes this transform
    label-safe even though it resamples both axes.
    """
    out = _to_float(img)
    n_rows, n_cols = out.shape[0], out.shape[1]
    s_r = float(rng.uniform(*scale))
    s_c = float(rng.uniform(*scale))
    m_r = max(2, int(round(n_rows * s_r))) if n_rows > 1 else 1
    m_c = max(2, int(round(n_cols * s_c))) if n_cols > 1 else 1
    tmp = _resample_axis(out, m_r, axis=0)
    tmp = _resample_axis(tmp, m_c, axis=1)
    tmp = _resample_axis(tmp, n_rows, axis=0)
    tmp = _resample_axis(tmp, n_cols, axis=1)
    return _restore_dtype(tmp, img.dtype)


#: Canonical application order: platform motion (heave, pitch, roll) acts on
#: the acoustics before recording, dropout is a recording fault, resolution is
#: a sensor property applied last. The fixed order also pins the rng stream,
#: so one seed always yields one output.
_ARTIFACTS: tuple[tuple[str, Callable[..., np.ndarray]], ...] = (
    ("heave_banding", heave_banding),
    ("pitch_stretch", pitch_stretch),
    ("roll_shear", roll_shear),
    ("ping_dropout", ping_dropout),
    ("resolution_jitter", resolution_jitter),
)

#: The subset that displaces no pixel — safe to apply after YOLO boxes are
#: computed (see module docstring). ``pitch_stretch`` and ``roll_shear`` move
#: pixels and are deliberately excluded.
LABEL_SAFE_ARTIFACTS: tuple[str, ...] = ("heave_banding", "ping_dropout", "resolution_jitter")


def apply_artifacts(
    img: np.ndarray,
    rng: np.random.Generator,
    p_each: float = 0.35,
    *,
    names: Sequence[str] | None = None,
) -> np.ndarray:
    """Compose a random subset of the PS artifacts on one chip.

    Each artifact in the canonical order (:data:`_ARTIFACTS`) is applied
    independently with probability *p_each*; *names* restricts the candidate
    set (e.g. :data:`LABEL_SAFE_ARTIFACTS` on the label-bearing dataset
    path) and raises ``ValueError`` on an unknown name. Selection draws are
    only made for candidate artifacts, so a given ``(rng seed, names)`` pair
    always produces the same output. Returns a new array (a copy even when
    nothing fires); shape and dtype are preserved throughout.
    """
    if names is not None:
        unknown = set(names) - {n for n, _ in _ARTIFACTS}
        if unknown:
            raise ValueError(f"unknown artifact names: {sorted(unknown)}")
    selected = {n for n, _ in _ARTIFACTS} if names is None else set(names)
    out = np.asarray(img).copy()
    for name, fn in _ARTIFACTS:
        if name not in selected:
            continue
        if rng.random() < p_each:
            out = fn(out, rng)
    return out
