"""Physics-safe albumentations pipeline for ground-range training chips.

Side-scan geometry admits only a restricted augmentation group, because a
ground-range chip is not a photograph -- its axes carry meaning:

* **Vertical flip (rows) is valid.** Rows are along-track; surveying the same
  seabed on the reciprocal heading simply reverses ping order and leaves the
  across-track highlight/shadow geometry untouched.
* **Horizontal flip (columns) is FORBIDDEN.** Columns are ground range with
  column 0 at nadir on both sides (port is stored nadir-first too), and
  acoustic shadows always extend toward increasing column. Mirroring columns
  would put every shadow on the *nadir* side of its highlight -- a physically
  impossible image no sonar can produce, and training on it would teach the
  detector to accept (and the shadow-physics validator to reject) nonsense.
* **Rotation is forbidden** for the same reason: it tilts the
  highlight-to-shadow axis away from the across-track direction.
* **Translate/scale are along-track only.** Across-track rescaling would
  silently change the shadow-length-to-height relation PhysiCheck relies on;
  along-track scaling merely mimics tow-speed / ping-rate variation, which is
  real.

The photometric transforms mimic acquisition variability: multiplicative
noise approximates Rayleigh speckle (which is multiplicative, not additive),
small gaussian blur stands in for defocus and motion smear, and
brightness/contrast/gamma cover residual TVG and display-mapping differences
between sonar models.

A final ``Lambda`` stage composes the PS-named acquisition artifacts — heave
banding, pitch stretch, roll shear, ping dropout, resolution jitter (see
:mod:`sonar_core.synth.artifacts` for the physics). All five respect the same
column-axis rule: none mirrors or reorders ground range. This image-only
pipeline may use the full set, including the two transforms that move pixels;
the label-bearing dataset path in :mod:`tridentnet.data` is restricted to the
label-safe subset instead.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import albumentations as alb
import cv2
import numpy as np

from sonar_core.synth.artifacts import apply_artifacts


def _artifact_fn(p_each: float) -> Callable[..., np.ndarray]:
    """Adapt :func:`apply_artifacts` to ``alb.Lambda``'s image callback.

    albumentations 1.4.15 passes no RNG to Lambda callbacks, so a fresh
    :class:`numpy.random.Generator` is derived per call from the NumPy legacy
    global state — the same state ``train_augment(seed=...)`` seeds — keeping
    artifact draws reproducible alongside the rest of the pipeline.
    """

    def _apply(image: np.ndarray, **_: object) -> np.ndarray:
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        return apply_artifacts(image, rng, p_each=p_each)

    return _apply


def train_augment(
    seed: int | None = None,
    *,
    noise_multiplier: tuple[float, float] = (0.85, 1.15),
    p_noise: float = 0.7,
    blur_limit: tuple[int, int] = (3, 5),
    p_blur: float = 0.3,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    p_brightness_contrast: float = 0.6,
    gamma_limit: tuple[float, float] = (80.0, 120.0),
    p_gamma: float = 0.4,
    p_vflip: float = 0.5,
    translate_frac: float = 0.06,
    scale_range: tuple[float, float] = (0.95, 1.05),
    p_affine: float = 0.5,
    artifact_p_each: float = 0.35,
) -> alb.Compose:
    """Build the training-chip augmentation pipeline (see module docstring).

    albumentations 1.4.15 draws its randomness from the global ``random`` and
    NumPy legacy RNG states, so passing *seed* seeds those globals -- call
    once per worker/epoch for reproducible batches.

    All magnitudes are keyword-tunable; the defaults are mild on purpose:
    augmentation must vary radiometry and along-track sampling, never
    manufacture geometry a sonar could not record.

    *artifact_p_each* is the per-artifact probability of the PS acquisition
    artifacts composed as the final stage (see module docstring); 0 disables
    the stage entirely.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    transforms: list[alb.BasicTransform] = [
        # Speckle is multiplicative in amplitude, so multiplicative
        # elementwise noise is the physically right family (additive
        # gaussian would wash out shadows, which are near-zero signal).
        alb.MultiplicativeNoise(multiplier=noise_multiplier, elementwise=True, p=p_noise),
        alb.GaussianBlur(blur_limit=blur_limit, p=p_blur),
        alb.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=p_brightness_contrast,
        ),
        alb.RandomGamma(gamma_limit=gamma_limit, p=p_gamma),
        # Along-track (row) flip only -- NEVER HorizontalFlip (see module
        # docstring: it would mirror shadows to the nadir side).
        alb.VerticalFlip(p=p_vflip),
        # Along-track-only jitter: y translate/scale, x frozen, no
        # rotation/shear (both would break shadow direction). Reflection
        # padding along-track is itself a valid sonar image (it equals a
        # locally reversed ping order), unlike constant black bars.
        alb.Affine(
            scale={"x": 1.0, "y": scale_range},
            translate_percent={"x": 0.0, "y": (-translate_frac, translate_frac)},
            rotate=0.0,
            shear=0.0,
            mode=cv2.BORDER_REFLECT_101,
            p=p_affine,
        ),
    ]
    if artifact_p_each > 0.0:
        # PS acquisition artifacts run last: they model recording-time faults
        # (banding, dropout, resampling) that act on the already-formed image.
        transforms.append(
            alb.Lambda(image=_artifact_fn(artifact_p_each), name="ps_artifacts", p=1.0)
        )
    return alb.Compose(transforms)


def apply_chip(aug: alb.Compose, img_u8: np.ndarray) -> np.ndarray:
    """Run one chip through *aug*; uint8 in, uint8 out, shape preserved.

    Detector training chips are uint8 (the pipeline's enhanced [0, 1] tiles
    quantized to 255 levels); enforcing the dtype here catches accidental
    float chips, for which albumentations would silently assume a [0, 1]
    range and clip real intensities.
    """
    img = np.asarray(img_u8)
    if img.dtype != np.uint8:
        raise TypeError(f"apply_chip expects uint8 input, got {img.dtype}")
    out = aug(image=img)["image"]
    if out.shape != img.shape:
        raise RuntimeError(f"augmentation changed chip shape: {img.shape} -> {out.shape}")
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out
