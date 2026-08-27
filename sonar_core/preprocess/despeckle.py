"""Light, edge-preserving speckle reduction for side-scan imagery.

Side-scan speckle is *multiplicative*: each resolution cell sums echoes from
many unresolved seabed scatterers with random phase, so the recorded amplitude
is (mean backscatter) x (a Rayleigh-distributed unit-mean random factor).
Averaging can therefore trade resolution for radiometric stability — but
debris *detection* in this pipeline leans on acoustic shadows, and PhysiCheck
derives target height from shadow length, so the boundary between insonified
seabed and shadow must stay sharp. Both filters here are deliberately light:

* :func:`lee_filter` — the classic minimum-mean-square-error filter for
  multiplicative noise. On homogeneous seabed the local signal variance is
  ~zero, the gain ``k`` drops to 0 and the output is the local mean (maximum
  smoothing); across a shadow edge the local variance is dominated by real
  scene contrast, ``k`` rises to 1 and the pixel passes through untouched.
* :func:`adaptive_median` — replaces only isolated impulse pixels (crosstalk
  spikes, dropped samples, bit errors) with the local median, leaving every
  other pixel bit-identical, so texture and shadow geometry are untouched.

Both are NaN-aware because ground-range imagery (:mod:`.slant_range`) marks
beyond-swath pixels as NaN: NaN in stays NaN out, and NaNs never poison the
statistics of their finite neighbours (normalized-convolution masking).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.ndimage import median_filter, uniform_filter

#: Coefficient of variation (std/mean) of fully developed Rayleigh amplitude
#: speckle, ``sqrt(4/pi - 1)`` ~ 0.5227. The theoretical noise level for raw
#: single-look side-scan amplitude; :func:`lee_filter` estimates ``cu`` from
#: the data by default because gain stages and prior averaging change it.
RAYLEIGH_CU: float = float(np.sqrt(4.0 / np.pi - 1.0))

#: Consistency constant scaling a median absolute deviation to the standard
#: deviation of an equivalent Gaussian, ``1 / Phi^-1(3/4)``. Lets
#: :func:`adaptive_median` express its threshold in robust-sigma units.
MAD_TO_SIGMA: float = 1.4826


def _local_stats(img: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NaN-aware local mean and variance over a ``size`` x ``size`` box.

    Uses normalized convolution: the data (with NaNs zeroed) and a validity
    mask are box-filtered separately and the ratio recovers the mean over
    *valid* pixels only, so a NaN shrinks its neighbours' effective window
    instead of poisoning it. Returns ``(mean, variance, finite_mask)``;
    mean/variance are NaN only where the whole window is invalid.
    """
    finite = np.isfinite(img)
    data = np.where(finite, img, 0.0).astype(np.float64)
    weight = finite.astype(np.float64)
    wsum = uniform_filter(weight, size=size)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        mu = uniform_filter(data, size=size) / wsum
        m2 = uniform_filter(data * data, size=size) / wsum
        var = np.maximum(m2 - mu * mu, 0.0)
    return mu, var, finite


def lee_filter(img: np.ndarray, size: int = 5, cu: float | None = None) -> np.ndarray:
    """Classic Lee filter for multiplicative (speckle) noise.

    Models each pixel as ``x = s * n`` with unit-mean noise of coefficient of
    variation *cu* and returns the linear MMSE estimate

        ``out = mu + k * (img - mu)``,  ``k = var_s / (var_s + var_n)``

    where ``mu``/``var`` are local box statistics, the underlying signal
    variance is ``var_s = (var - (cu * mu)^2) / (1 + cu^2)`` and the noise
    variance at the estimated signal level is ``var_n = cu^2 * (mu^2 +
    var_s)``. Homogeneous seabed gives ``k ~ 0`` (full smoothing); a shadow
    or highlight edge inflates ``var`` far above the speckle prediction, so
    ``k -> 1`` and the edge passes through unsmeared — which is why Lee is
    safe ahead of shadow-based height estimation.

    Parameters
    ----------
    img:
        2-D intensity image (slant- or ground-range). NaNs are preserved and
        excluded from neighbourhood statistics.
    size:
        Box window side in pixels. Keep small (5-7): larger windows smooth
        more but bias ``k`` near thin shadows.
    cu:
        Noise coefficient of variation. ``None`` (default) estimates it as
        the median of local std/mean over the image — a robust proxy because
        most windows of a survey are homogeneous seabed where the local
        variation *is* the speckle. Pass :data:`RAYLEIGH_CU` to assert raw
        single-look statistics.
    """
    img = np.asarray(img)
    out_dtype = img.dtype if np.issubdtype(img.dtype, np.floating) else np.float32
    mu, var, finite = _local_stats(img, size)

    if cu is None:
        with np.errstate(invalid="ignore", divide="ignore"):
            local_cv = np.sqrt(var) / mu
        valid_cv = local_cv[finite & np.isfinite(local_cv)]
        cu = float(np.median(valid_cv)) if valid_cv.size else RAYLEIGH_CU

    cu2 = cu * cu
    var_signal = np.maximum(var - cu2 * mu * mu, 0.0) / (1.0 + cu2)
    var_noise = cu2 * (mu * mu + var_signal)
    denom = var_signal + var_noise
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(denom > 0, var_signal / denom, 0.0)

    out = mu + k * (img.astype(np.float64) - mu)
    out[~finite] = np.nan
    return out.astype(out_dtype, copy=False)


def adaptive_median(
    img: np.ndarray,
    size: int = 3,
    threshold: float = 3.0,
    mad_size: int | None = None,
) -> np.ndarray:
    """Replace only impulse pixels with the local median; pass the rest through.

    An impulse (electrical crosstalk, a dropped sample, a surface-return
    spike) is a single-pixel outlier, statistically incompatible with the
    heavy-but-continuous Rayleigh speckle tail. A pixel is flagged when it
    deviates from the local median by more than ``threshold`` robust sigmas,
    where the robust sigma is the local median absolute deviation scaled by
    :data:`MAD_TO_SIGMA`. Median/MAD are used (not mean/std) so the impulses
    being hunted cannot inflate their own detection threshold. Non-flagged
    pixels are returned bit-identical — speckle texture and shadow boundaries
    are never touched, making this safe at any point in the pipeline.

    Parameters
    ----------
    img:
        2-D intensity image. NaNs are preserved (never counted as impulses);
        for the neighbourhood medians they are stand-in filled with the
        global median so they cannot drag local statistics to NaN.
    size:
        Median window side in pixels; 3 targets single-pixel impulses.
    threshold:
        Detection level in robust-sigma units. For raw single-look Rayleigh
        speckle the upper tail is heavy, so 3.5-4.0 is appropriate there;
        after EGN/mosaicking the default 3.0 is a sound 3-sigma rule.
    mad_size:
        Window side for the MAD estimate. Defaults to ``2 * size + 1``: a
        window as small as the median's own gives a badly downward-biased
        spread estimate (too few samples), which over-triggers replacement.
    """
    img = np.asarray(img)
    finite = np.isfinite(img)
    if not finite.any():
        return img.copy()
    if mad_size is None:
        mad_size = 2 * size + 1
    filled = img if finite.all() else np.where(finite, img, np.nanmedian(img))
    med = median_filter(filled, size=size)
    mad = median_filter(np.abs(filled - med), size=mad_size)
    impulse = finite & (np.abs(img - med) > threshold * MAD_TO_SIGMA * mad)
    return np.where(impulse, med, img)


_METHODS: dict[str, Callable[..., np.ndarray]] = {
    "lee": lee_filter,
    "median": adaptive_median,
    "adaptive_median": adaptive_median,
}


def despeckle(img: np.ndarray, method: str = "lee", **kwargs: object) -> np.ndarray:
    """Dispatch to a named despeckling filter.

    ``method`` is one of ``"lee"``, ``"median"``/``"adaptive_median"``;
    remaining keyword arguments are forwarded to the filter. Raises
    ``ValueError`` for an unknown method so config typos fail loudly.
    """
    try:
        fn = _METHODS[method]
    except KeyError:
        raise ValueError(
            f"unknown despeckle method {method!r}; known: {sorted(set(_METHODS))}"
        ) from None
    return fn(img, **kwargs)
