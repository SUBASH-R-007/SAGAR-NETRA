"""Classical CAD baseline — the pre-deep-learning detector, for comparison only.

This is **not part of the deployed stack**. It exists so the ablation ladder
has a floor that is not SAGAR-NETRA: a faithful reimplementation of the
threshold-and-blob computer-aided-detection scheme that side-scan survey
software used before learned detectors, and that a reviewer will reasonably
ask us to beat.

The classical recipe, as it appears across the sonar ATR literature and in
legacy survey packages:

1. **Range-normalize.** Background brightness falls off across the swath, so
   the threshold cannot be global. A per-range-column median and MAD-scaled
   sigma give a robust background model that is immune to the very targets
   being searched for (a handful of bright pixels cannot move a median).
2. **Threshold** at ``median + k * sigma`` per column.
3. **Morphological opening** to erase isolated speckle survivors.
4. **Connected components**, each candidate's box taken from its extent.
5. **Geometry filters** — minimum and maximum area, maximum aspect ratio —
   to discard both single-pixel noise and swath-spanning banding artefacts.
6. Optionally, **shadow gating**: require a dark region immediately down-range
   of the highlight, which is the one physical cue the classical method can
   cheaply exploit.

Two variants are exposed because the difference between them is itself
evidence: ``require_shadow=False`` is the plain blob detector, and
``require_shadow=True`` adds the shadow test. Both are honest classical
methods; neither learns anything.

**The baseline runs on gain-corrected but *not* contrast-equalized imagery**
(``PreprocessResult.ground_raw``), which is both historically correct — classical
CAD predates CLAHE in this pipeline and operated on TVG/EGN-corrected data — and
enormously more favourable to it. Measured on the held-out scenes, true targets
peak at **8-30 sigma** above the robust per-column background on ``ground_raw``
but only **1.7-3.2 sigma** after CLAHE, because local contrast equalization
deliberately compresses exactly the global target-to-background separation a
fixed threshold depends on. Running the baseline on the CLAHE'd image would have
made it look far worse for a reason that has nothing to do with the classical
method's merits, so it would have been a strawman.

Boxes follow the same convention as the training labels and the truth boxes
(:func:`tridentnet.data._target_bbox`): the highlight extent padded by
``shadow_pad_cols`` down-range. The full shadow is deliberately *not* included
— truth boxes do not include it either, so a detector that swallowed the whole
shadow would score worse on IoU despite being no less correct.

Candidates carry ``cls="unclassified"``: a threshold-and-blob scheme has no
notion of what it found. That is a real limitation of the baseline, not an
artefact of this implementation, and any comparison against it must therefore
be scored on localization alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from tridentnet.detector import Detection

#: Class label for every classical candidate. A blob detector localizes; it
#: does not classify. Comparisons must be run with class matching relaxed.
UNCLASSIFIED = "unclassified"

#: Contrast at which a candidate's score reaches 0.5 under ``snr / (snr + K)``.
#: Only the *ranking* matters for PR-AUC, so this constant sets the shape of
#: the score curve, never which candidates are emitted.
SCORE_HALF_SNR = 6.0


@dataclass(frozen=True)
class ClassicalConfig:
    """Tunables for the classical detector.

    Defaults are a reasonable mid-swath operating point; ``k_sigma`` is the
    knob a survey operator actually turns, and the comparison harness sweeps
    it rather than assuming this value, so the baseline is judged at its own
    best threshold instead of an arbitrary one.
    """

    k_sigma: float = 3.0
    min_area_px: int = 12
    #: Upper area bound exists to reject swath-spanning banding artefacts, not
    #: to bound target size. Set well clear of the largest real target: the
    #: biggest truth box in the held-out set is an aircraft at ~19.5k px, so a
    #: 20k cap would have sat one bad seed away from silently dropping it and
    #: charging the miss to the baseline.
    max_area_px: int = 100_000
    open_radius: int = 1
    max_aspect: float = 12.0
    require_shadow: bool = False
    shadow_search_cols: int = 48
    #: A shadow pixel is one dimmer than this fraction of local background.
    shadow_darkness: float = 0.70
    #: Fraction of the down-range window that must be dark to call it a shadow.
    shadow_min_frac: float = 0.35
    #: Down-range padding, matching the training-label box convention.
    shadow_pad_cols: int = 3
    #: Detect on gain-corrected pre-CLAHE imagery. See the module docstring:
    #: contrast equalization costs the baseline roughly an order of magnitude
    #: of target-to-background separation, so this default is the fair one.
    use_raw_imagery: bool = True


def _background(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-range-column robust background: median and MAD-scaled sigma.

    Taken down the ping axis, so each across-track column gets its own
    statistics — this is the classical answer to grazing-angle falloff, and it
    is robust by construction: targets occupy a small minority of pings in any
    one column, so they cannot drag the median or the MAD.
    """
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(img, axis=0)
        mad = np.nanmedian(np.abs(img - med[None, :]), axis=0)
    sigma = 1.4826 * mad  # MAD -> sigma for a normal distribution

    # Dead or constant columns (blanked water column, swath edge) yield zero
    # spread; borrow the typical column's sigma so they cannot threshold at
    # exactly the median and light up entirely.
    positive = sigma[np.isfinite(sigma) & (sigma > 0)]
    fallback = float(np.median(positive)) if positive.size else 1e-3
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, fallback)
    med = np.where(np.isfinite(med), med, 0.0)
    return med, sigma


def _shadow_fraction(
    img: np.ndarray, med: np.ndarray, r0: int, r1: int, c1: int, cfg: ClassicalConfig
) -> float:
    """Fraction of the down-range window behind a candidate that reads as shadow.

    Looks at the rows the candidate spans, in the columns immediately beyond
    it — where an acoustic shadow must fall if the object has any height.
    Returns 0.0 when the window runs off the swath.
    """
    n_cols = img.shape[1]
    start = min(c1 + 1, n_cols)
    stop = min(start + cfg.shadow_search_cols, n_cols)
    if stop <= start:
        return 0.0
    window = img[r0 : r1 + 1, start:stop]
    finite = np.isfinite(window)
    if not finite.any():
        return 0.0
    floor = cfg.shadow_darkness * med[start:stop][None, :]
    dark = finite & (np.nan_to_num(window, nan=np.inf) < floor)
    return float(dark.sum() / finite.sum())


def detect_side(img: np.ndarray, side: str, cfg: ClassicalConfig) -> list[Detection]:
    """Classical candidates on one side's ground-range image."""
    if img.size == 0:
        return []
    finite = np.isfinite(img)
    med, sigma = _background(img)

    thresh = med + cfg.k_sigma * sigma
    mask = finite & (np.nan_to_num(img, nan=-np.inf) > thresh[None, :])
    if cfg.open_radius > 0:
        size = 2 * cfg.open_radius + 1
        mask = ndimage.binary_opening(mask, structure=np.ones((size, size), dtype=bool))

    labels, n_found = ndimage.label(mask)
    if n_found == 0:
        return []

    n_cols = img.shape[1]
    detections: list[Detection] = []
    for index, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        rows, cols = sl
        r0, r1 = int(rows.start), int(rows.stop) - 1
        c0, c1 = int(cols.start), int(cols.stop) - 1

        component = labels[sl] == index
        area = int(component.sum())
        if area < cfg.min_area_px or area > cfg.max_area_px:
            continue

        height, width = r1 - r0 + 1, c1 - c0 + 1
        aspect = height / width
        if aspect > cfg.max_aspect or aspect < 1.0 / cfg.max_aspect:
            continue

        if cfg.require_shadow:
            if _shadow_fraction(img, med, r0, r1, c1, cfg) < cfg.shadow_min_frac:
                continue

        patch = img[sl]
        values = patch[component & np.isfinite(patch)]
        if values.size == 0:
            continue
        bg = float(np.mean(med[c0 : c1 + 1]))
        spread = float(np.mean(sigma[c0 : c1 + 1])) or 1e-6
        snr = max((float(values.mean()) - bg) / spread, 0.0)

        detections.append(
            Detection(
                side=side,
                ping0=r0,
                ping1=r1,
                col0=c0,
                col1=min(c1 + cfg.shadow_pad_cols, n_cols - 1),
                cls=UNCLASSIFIED,
                score=snr / (snr + SCORE_HALF_SNR),
                brain="classical",
            )
        )
    return detections


class ClassicalCAD:
    """Threshold-and-blob detector over a whole preprocessed survey.

    Deliberately mirrors the ``detect_tiles`` shape of the learned brains so
    the comparison harness can score both through identical code, but it works
    on the full ground image rather than tiles: classical CAD has no tiling
    step, and forcing one on it would invent a failure mode it does not have.
    """

    def __init__(self, config: ClassicalConfig | None = None) -> None:
        self.config = config or ClassicalConfig()

    def detect(self, pre) -> list[Detection]:
        """All candidates across both sides of a :class:`PreprocessResult`.

        ``ground_raw`` and ``ground`` share slant-corrected geometry exactly —
        same shape, same ``ground_res`` — so candidate boxes found on either
        are directly comparable with truth boxes and with the learned brains'
        output.
        """
        image = pre.ground_raw if self.config.use_raw_imagery else pre.ground
        found: list[Detection] = []
        for side in ("port", "starboard"):
            found.extend(detect_side(image.side(side), side, self.config))
        return found
