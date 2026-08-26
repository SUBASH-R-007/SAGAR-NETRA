"""Empirical Gain Normalization: remove residual time-varied-gain banding.

Side-scan receivers apply a time-varied gain (nominally ``A*log(r) + 2*a*r``)
to undo spherical spreading and absorption, but the hardware curve never
matches the true transmission loss, and the vertical beam pattern adds its
own range-dependent lobe structure. The mismatch is a smooth gain residual
that depends on *range only* — it is a function of two-way travel time, so
every ping shares the same profile. Over many pings the seabed reflectivity
at a fixed range is a stationary random variable, which means the per-range
MEDIAN of seabed pixels estimates the residual directly; the median is robust
to the minority outliers every survey contains (bright target highlights,
dark acoustic shadows) that would bias a mean. Dividing the residual out
flattens range banding without touching along-track reflectivity structure.

Water-column samples (before the first bottom return) contain only ambient
noise and the near-field reverberation tail; they carry no seabed gain
information, so they are excluded from the statistics and passed through the
normalization unmodified.
"""

from __future__ import annotations

import numpy as np


def empirical_gain_normalize(
    intensity: np.ndarray,
    first_return: np.ndarray | None = None,
    n_bins: int = 256,
    target: float | None = None,
    eps_frac: float = 0.01,
    nadir_guard: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten range-dependent gain banding in one side's slant-domain image.

    Sample indices are grouped into ``n_bins`` contiguous range bins (capped
    at the sample count) and the median seabed intensity of each bin, taken
    across all pings, is divided by *target* (default: the global median of
    all seabed pixels, so the overall brightness is preserved) to form a
    per-bin gain. Binning trades range resolution of the gain estimate for
    statistical support — the TVG residual is smooth in range, so a few
    hundred bins resolve it fully while each bin still pools enough pixels
    for a stable median. The per-bin gains are linearly interpolated to a
    per-sample curve; empty bins (all water column or NaN) inherit values
    interpolated from their nearest estimated neighbours, and gains below
    ``eps_frac * median(gain)`` are clamped to that floor so dead bins never
    amplify noise into artificial brightness.

    Parameters
    ----------
    intensity:
        ``(n_pings, n_samples)`` float32 slant-domain image for one side,
        sample 0 at nadir. NaN pixels are excluded from statistics and stay
        NaN in the output.
    first_return:
        Optional ``(n_pings,)`` integer first-bottom-return sample indices
        (see :func:`sonar_core.preprocess.bottom_track.track_bottom`).
        Samples before it are water column: excluded from the gain statistics
        and returned bit-identical to the input.
    n_bins:
        Number of range bins for the gain estimate.
    target:
        Intensity every bin median is normalized towards. Defaults to the
        global median of all seabed pixels.
    eps_frac:
        Fraction of the median gain used as the clamp floor for dead bins.
    nadir_guard:
        Samples excluded from the gain *statistics* immediately after the
        first return. The bottom-return transient (a bright pulse-length
        peak that sweeps across samples as altitude wobbles) is not seabed
        reflectivity; without the guard it inflates near-nadir gain by up
        to ~2x. Normalization is still *applied* from ``first_return``
        onward — only the statistics skip the guard band.

    Returns
    -------
    tuple
        ``(normalized, gain)``: the normalized ``(n_pings, n_samples)``
        float32 image (``intensity / gain`` on the seabed, water column
        untouched) and the strictly positive ``(n_samples,)`` float64
        per-sample gain curve.
    """
    img = np.asarray(intensity, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"intensity must be 2-D (n_pings, n_samples), got shape {img.shape}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    n_pings, n_samples = img.shape
    identity = np.ones(n_samples, dtype=np.float64)
    if n_pings == 0 or n_samples == 0:
        return img.copy(), identity
    n_bins = min(n_bins, n_samples)

    cols = np.arange(n_samples)
    seabed = np.isfinite(img)
    fr: np.ndarray | None = None
    if first_return is not None:
        fr = np.asarray(first_return)
        if fr.shape != (n_pings,):
            raise ValueError(
                f"first_return must have shape ({n_pings},), got {fr.shape}"
            )
        fr = np.clip(fr.astype(np.int64), 0, None)
        seabed &= cols[None, :] >= (fr + max(int(nadir_guard), 0))[:, None]

    # Seabed-only working copy: excluded pixels become NaN so nan-statistics
    # see exactly the pixels that carry gain information.
    work = np.where(seabed, img.astype(np.float64), np.nan)

    # Contiguous range bins over the sample axis; n_bins <= n_samples
    # guarantees every bin spans at least one sample column.
    bin_of = (cols * n_bins) // n_samples
    edges = np.searchsorted(bin_of, np.arange(n_bins + 1))
    centres = (edges[:-1] + edges[1:] - 1) / 2.0

    bin_median = np.full(n_bins, np.nan)
    for b in range(n_bins):
        block = work[:, edges[b] : edges[b + 1]]
        if np.isfinite(block).any():
            bin_median[b] = np.nanmedian(block)

    estimated = np.isfinite(bin_median)
    if not estimated.any():
        # No seabed pixel anywhere: nothing to normalize against.
        return img.copy(), identity
    if not estimated.all():
        # Empty bins inherit interpolated values (flat extension at the ends).
        bin_median = np.interp(centres, centres[estimated], bin_median[estimated])

    if target is None:
        target_val = float(np.nanmedian(work))
        if not np.isfinite(target_val) or target_val <= 0:
            # Degenerate seabed (e.g. all-zero image): identity gain is the
            # only normalization that cannot invent structure.
            return img.copy(), identity
    else:
        target_val = float(target)
        if not np.isfinite(target_val) or target_val <= 0:
            raise ValueError(f"target must be finite and positive, got {target}")

    gain = np.interp(cols.astype(np.float64), centres, bin_median / target_val)

    med_gain = float(np.median(gain))
    if not np.isfinite(med_gain) or med_gain <= 0:
        return img.copy(), identity
    # Clamp dead-bin gains: dividing by a near-zero gain would blow ambient
    # noise up into bright bands, so the floor keeps them dark instead.
    gain = np.maximum(gain, eps_frac * med_gain)

    normalized = (img.astype(np.float64) / gain[None, :]).astype(np.float32)
    if fr is not None:
        # Water column passes through bit-identical (np.where copies the
        # original float32 values, no arithmetic touches them).
        normalized = np.where(cols[None, :] < fr[:, None], img, normalized)
    return normalized, gain
