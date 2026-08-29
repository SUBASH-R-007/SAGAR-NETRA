"""Brain C: the autoencoder must flag what it has never seen — and only that.
Uses a tiny CPU training run (module-scoped) to keep the suite honest but fast."""

from __future__ import annotations

import numpy as np
import pytest

from sonar_core.preprocess.pipeline import preprocess
from sonar_core.synth.scene import SceneConfig, SynthTarget, make_scene
from tridentnet.anomaly import AnomalyDetector, train_anomaly
from tridentnet.classes import ANOMALY_CLASS
from tridentnet.detector import Detection
from tridentnet.ensemble import merge_brains


@pytest.fixture(scope="module")
def trained_weights(tmp_path_factory):
    """~1 minute CPU training on two small clean scenes; returns the checkpoint.

    Exposed as a path, not just a detector, so tests that need to build several
    detectors with different configs off the same weights can do so without
    retraining.
    """
    tiles = []
    for seed in (50, 51):
        cfg = SceneConfig(n_pings=300, n_samples=512, slant_range=40.0, seed=seed)
        pa, _ = make_scene(cfg, targets=[])
        tiles.extend(t.image for t in preprocess(pa).tiles)
    out = tmp_path_factory.mktemp("weights") / "anomaly.pt"
    train_anomaly(
        tiles,
        out_path=out,
        config={"epochs": 4, "patch": 96, "batch": 8},
        device="cpu",
        seed=0,
    )
    return out


@pytest.fixture(scope="module")
def trained(trained_weights):
    return AnomalyDetector(weights=trained_weights, device="cpu")


@pytest.fixture(scope="module")
def target_scene():
    cfg = SceneConfig(n_pings=300, n_samples=512, slant_range=40.0, seed=77)
    target = SynthTarget(
        "container", "starboard", 150, 20.0, 5.0, 2.5, height=2.2, reflectivity=7.0
    )
    pa, _ = make_scene(cfg, [target])
    return preprocess(pa), target


def test_seeded_target_is_flagged(trained, target_scene) -> None:
    pre, target = target_scene
    stbd_tiles = [t for t in pre.tiles if t.side == "starboard"]
    found = trained.detect_tiles(stbd_tiles)
    assert found, "anomaly brain found nothing at all"
    assert all(d.cls == ANOMALY_CLASS and d.brain == "C" for d in found)

    gi = pre.ground
    t_col = gi.col_of_ground_range(target.ground_range)
    hit = any(
        d.ping0 - 25 <= target.ping <= d.ping1 + 25
        and d.col0 - 25 <= t_col <= d.col1 + 25
        for d in found
    )
    assert hit, f"no anomaly near seeded target (found {found[:5]})"


def test_clean_background_mostly_quiet(trained) -> None:
    cfg = SceneConfig(n_pings=300, n_samples=512, slant_range=40.0, seed=52)
    pa, _ = make_scene(cfg, targets=[])
    pre = preprocess(pa)
    found = trained.detect_tiles(pre.tiles)
    # The threshold is the 99.5th percentile of clean error, so a few small
    # flukes are expected; a flood means the brain is broken.
    assert len(found) <= 6, f"too many false anomalies on clean seabed: {len(found)}"


def test_error_map_shape_and_swath_masking(trained) -> None:
    img = np.random.default_rng(0).random((100, 130)).astype(np.float32)
    img[:, -10:] = np.nan
    err = trained.error_map(img)
    assert err.shape == img.shape
    assert np.all(err[:, -10:] == 0.0), "out-of-swath must not be anomalous"


def test_ensemble_corroboration_and_standalone() -> None:
    a = Detection("starboard", 10, 20, 30, 50, "container", 0.80, brain="A")
    c_overlap = Detection("starboard", 12, 22, 32, 52, ANOMALY_CLASS, 0.6, brain="C")
    c_alone = Detection("port", 100, 110, 200, 220, ANOMALY_CLASS, 0.7, brain="C")
    merged = merge_brains([a], [c_overlap, c_alone])
    assert len(merged) == 2
    corroborated = next(d for d in merged if d.cls == "container")
    assert corroborated.brain == "AC" and corroborated.score > 0.80
    standalone = next(d for d in merged if d.cls == ANOMALY_CLASS)
    assert standalone.side == "port" and standalone.brain == "C"


# ---------------------------------------------------------- candidate budget ----
# Brain C answers "what here is unlike plain seabed". On a real seabed that is
# often "a great deal" -- rock fields and ripples are genuinely unlike flat
# sediment, and unlike synthetic speckle they are spatially coherent, so they
# survive min_blob_px as blobs instead of being discarded as stray pixels.
# Measured on real imagery: a median of 12 blobs per tile against a synthetic
# maximum of 12. The budget bounds what a busy seabed can cost downstream; it
# must not change what a quiet one produces.


def _key(dets):
    return sorted(
        (d.side, d.ping0, d.ping1, d.col0, d.col1, round(d.score, 6)) for d in dets
    )


def test_candidate_budget_ships_disabled() -> None:
    """The budget must stay off by default, in code and in config.

    Measured over 70 real images, a budget of 16 removed every detection that
    would have survived the confidence floor (6 -> 0): ranking is by raw
    reconstruction peak while survival is decided by highlight/shadow physics,
    so the two are close to anti-correlated. Enabling it silently would look
    like a tidy reduction in candidates and would quietly cost recall, so the
    default is pinned here as well as documented.
    """
    from pathlib import Path

    import yaml

    from tridentnet.anomaly import DEFAULTS

    assert DEFAULTS["max_blobs_per_tile"] == 0
    shipped = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "anomaly.yaml")
        .read_text(encoding="utf-8")
    )
    assert shipped["max_blobs_per_tile"] == 0


def test_generous_budget_is_a_no_op(trained_weights, target_scene) -> None:
    """A budget above a scene's blob count must change nothing."""
    pre, _ = target_scene
    off = AnomalyDetector(weights=trained_weights, device="cpu",
                          config={"max_blobs_per_tile": 0})
    generous = AnomalyDetector(weights=trained_weights, device="cpu",
                               config={"max_blobs_per_tile": 10_000})
    assert _key(off.detect_tiles(pre.tiles)) == _key(generous.detect_tiles(pre.tiles))


def test_budget_truncates_and_keeps_the_strongest(trained_weights, target_scene) -> None:
    """When the budget bites, the survivors are the highest-scoring blobs.

    The threshold is lowered so the scene reliably produces far more blobs than
    the budget allows -- the truncation path must be exercised, not skipped on a
    quiet scene. A real target peaks well above the threshold while texture
    barely crosses it, so ranking by score drops the least target-like
    candidates first; dropping an arbitrary subset would make the budget a
    silent quality change rather than a cost bound.
    """
    pre, _ = target_scene
    budget = 2

    uncapped = AnomalyDetector(weights=trained_weights, device="cpu",
                               config={"max_blobs_per_tile": 0})
    uncapped.threshold *= 0.5  # force a blob-rich error map
    every = uncapped.detect_tiles(pre.tiles)

    capped = AnomalyDetector(weights=trained_weights, device="cpu",
                             config={"max_blobs_per_tile": budget})
    capped.threshold = uncapped.threshold
    kept = capped.detect_tiles(pre.tiles)

    per_tile: dict[int, int] = {}
    for d in every:
        per_tile[d.tile_index] = per_tile.get(d.tile_index, 0) + 1
    assert max(per_tile.values()) > budget, "scene must exceed the budget to test it"
    assert len(kept) < len(every)

    kept_per_tile: dict[int, list[float]] = {}
    for d in kept:
        kept_per_tile.setdefault(d.tile_index, []).append(d.score)
    assert all(len(v) <= budget for v in kept_per_tile.values())

    # Survivors must be that tile's top scores, not an arbitrary subset.
    all_per_tile: dict[int, list[float]] = {}
    for d in every:
        all_per_tile.setdefault(d.tile_index, []).append(d.score)
    for idx, scores in kept_per_tile.items():
        best = sorted(all_per_tile[idx], reverse=True)[: len(scores)]
        assert sorted(scores, reverse=True) == pytest.approx(best)
