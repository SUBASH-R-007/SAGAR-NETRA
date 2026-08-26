"""EGN: banding removal, water-column safety, NaN robustness, scene flattening."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d

from sonar_core.preprocess.bottom_track import track_bottom
from sonar_core.preprocess.egn import empirical_gain_normalize
from sonar_core.synth.scene import SceneConfig, make_scene


def _speckled_image(
    rng: np.random.Generator,
    n_pings: int,
    n_samples: int,
    gain_curve: np.ndarray,
    level: float = 1000.0,
) -> np.ndarray:
    """Flat reflectivity * known gain * unit-mean Rayleigh speckle."""
    speckle = rng.rayleigh(scale=np.sqrt(2 / np.pi), size=(n_pings, n_samples))
    return (level * gain_curve[None, :] * speckle).astype(np.float32)


def _banding_amplitude(med: np.ndarray, slant: np.ndarray, period_m: float) -> float:
    """Amplitude of the sinusoid at 1/period_m in the relative column medians,
    estimated jointly with a quadratic trend so grazing-angle falloff does not
    leak into the sinusoid coefficients."""
    rel = med / np.mean(med)
    x = np.linspace(-1.0, 1.0, len(rel))
    w = 2 * np.pi * slant / period_m
    design = np.column_stack([np.ones_like(x), x, x**2, np.sin(w), np.cos(w)])
    coef, *_ = np.linalg.lstsq(design, rel, rcond=None)
    return float(np.hypot(coef[3], coef[4]))


def test_flattens_known_gain_curve() -> None:
    """Flat reflectivity x smooth g(s) x speckle -> flat column medians."""
    rng = np.random.default_rng(42)
    n_pings, n_samples = 500, 320
    s = np.arange(n_samples)
    g = 1.0 + 0.6 * np.sin(2 * np.pi * s / 90.0) + 0.4 * s / n_samples  # in [0.4, 2.0]
    img = _speckled_image(rng, n_pings, n_samples, g)

    norm, gain = empirical_gain_normalize(img)

    def smoothed_ratio(a: np.ndarray) -> float:
        med = uniform_filter1d(np.median(a, axis=0).astype(np.float64), size=9)
        return float(med.max() / med.min())

    before = smoothed_ratio(img)
    after = smoothed_ratio(norm)
    assert before > 3.0, "synthetic banding should be strong before EGN"
    assert after < 1.12, "column medians must be flat within a few percent after EGN"
    assert before / after > 3.0, "EGN must improve flatness dramatically"
    # The recovered gain curve follows the injected one.
    assert np.corrcoef(gain, g)[0, 1] > 0.99


def test_water_column_excluded_and_untouched() -> None:
    """Samples before first_return stay bit-identical and never skew the gain."""
    rng = np.random.default_rng(7)
    n_pings, n_samples = 300, 200
    s = np.arange(n_samples)
    g = 1.0 + 0.5 * np.sin(2 * np.pi * s / 70.0)
    base = _speckled_image(rng, n_pings, n_samples, g, level=800.0)
    fr = (55 + np.arange(n_pings) % 11).astype(np.int32)
    water = np.arange(n_samples)[None, :] < fr[:, None]

    img_hot = base.copy()
    img_hot[water] = 1e6  # would wreck the gain estimate if included
    img_cold = base.copy()
    img_cold[water] = 0.0

    norm_hot, gain_hot = empirical_gain_normalize(img_hot, first_return=fr)
    _, gain_cold = empirical_gain_normalize(img_cold, first_return=fr)

    # Identical seabed pixels -> identical gain, whatever the water column holds.
    np.testing.assert_array_equal(gain_hot, gain_cold)
    # Water column is returned bit-identical to the input.
    np.testing.assert_array_equal(norm_hot[water], img_hot[water])
    # Seabed pixels were actually normalized (gain deviates from 1 somewhere).
    seabed_cols = s >= int(fr.max())
    assert not np.array_equal(norm_hot[:, seabed_cols], img_hot[:, seabed_cols])
    # Where every ping is seabed, gain/g must be constant: the estimate tracks
    # the injected curve without any water-column bias.
    ratio = gain_hot[seabed_cols] / g[seabed_cols]
    assert np.std(ratio) / np.mean(ratio) < 0.06


def test_nan_pixels_excluded_and_preserved() -> None:
    """NaNs never enter the statistics; empty bins interpolate from neighbours."""
    rng = np.random.default_rng(3)
    n_pings, n_samples = 200, 128
    s = np.arange(n_samples)
    g = 1.2 + 0.4 * np.cos(2 * np.pi * s / 50.0)
    base = _speckled_image(rng, n_pings, n_samples, g)
    img = base.copy()
    img[10:50, 30] = np.nan  # partial column
    img[:, 64] = np.nan  # full column -> empty bin (n_bins == n_samples)

    norm, gain = empirical_gain_normalize(img, n_bins=n_samples)

    assert np.isnan(norm[10:50, 30]).all()
    assert np.isnan(norm[:, 64]).all()
    assert np.isfinite(norm[np.isfinite(img)]).all()
    assert np.isfinite(gain).all()
    assert (gain > 0).all()
    # Empty bin 64 sits exactly on the interpolant between its neighbours.
    assert gain[64] == pytest.approx((gain[63] + gain[65]) / 2, rel=1e-9)
    # Gain statistics barely move versus the NaN-free image.
    _, gain_clean = empirical_gain_normalize(base, n_bins=n_samples)
    np.testing.assert_allclose(gain, gain_clean, rtol=0.2)


def test_gain_curve_shape_guards_and_target() -> None:
    """Length, dtype, positivity, n_bins cap, dead-band clamp, target scaling."""
    rng = np.random.default_rng(11)
    n_pings, n_samples = 64, 100
    g = np.full(n_samples, 1.0)
    img = _speckled_image(rng, n_pings, n_samples, g)
    img[:, 40:50] = 0.0  # dead band: gain would be 0 without the clamp

    norm, gain = empirical_gain_normalize(img, n_bins=10_000)  # capped at n_samples

    assert gain.shape == (n_samples,)
    assert gain.dtype == np.float64
    assert (gain > 0).all()
    assert norm.shape == img.shape
    assert norm.dtype == np.float32
    # Dead-band gains are clamped near the eps floor, far below the median.
    assert (gain[41:49] < 0.05 * np.median(gain)).all()
    assert gain[41:49].min() >= 0.01 * np.median(gain) * 0.9
    # Explicit target scales the gain curve inversely (target is the level the
    # bin medians are divided by), clamp floor included.
    _, gain_t1 = empirical_gain_normalize(img, target=500.0)
    _, gain_t2 = empirical_gain_normalize(img, target=1000.0)
    np.testing.assert_allclose(gain_t2, gain_t1 / 2.0, rtol=1e-12)


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="2-D"):
        empirical_gain_normalize(np.zeros(10, dtype=np.float32))
    with pytest.raises(ValueError, match="first_return"):
        empirical_gain_normalize(
            np.ones((4, 8), dtype=np.float32), first_return=np.zeros(3, dtype=np.int32)
        )
    with pytest.raises(ValueError, match="target"):
        empirical_gain_normalize(np.ones((4, 8), dtype=np.float32), target=-1.0)
    with pytest.raises(ValueError, match="n_bins"):
        empirical_gain_normalize(np.ones((4, 8), dtype=np.float32), n_bins=0)


def test_flattens_rendered_scene_banding() -> None:
    """The scene's imperfect-TVG sinusoid (period 17 m in slant range) must
    shrink by at least 3x in the seabed column medians."""
    cfg = SceneConfig(n_pings=240, n_samples=384, slant_range=48.0, seed=11)
    pa, _ = make_scene(cfg, targets=[])
    bt = track_bottom(pa)
    img = pa.starboard
    fr = bt.first_return["starboard"]

    norm, gain = empirical_gain_normalize(img, first_return=fr)
    assert gain.shape == (cfg.n_samples,)
    assert (gain > 0).all()

    seabed = np.arange(cfg.n_samples)[None, :] >= fr[:, None]
    s0 = int(fr.max()) + 8  # skip the smeared first-return peak

    def column_medians(a: np.ndarray) -> np.ndarray:
        w = np.where(seabed, a.astype(np.float64), np.nan)[:, s0:]
        return np.nanmedian(w, axis=0)  # every column past s0 has seabed pixels

    slant_res = cfg.slant_range / cfg.n_samples
    slant = (np.arange(cfg.n_samples)[s0:] + 0.5) * slant_res
    amp_before = _banding_amplitude(column_medians(img), slant, period_m=17.0)
    amp_after = _banding_amplitude(column_medians(norm), slant, period_m=17.0)
    assert amp_before > 0.05, "scene should render visible TVG banding"
    assert amp_before / amp_after > 3.0, "EGN must suppress the banding at least 3x"
