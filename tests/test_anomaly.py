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
def trained(tmp_path_factory):
    """~1 minute CPU training on two small clean scenes."""
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
    return AnomalyDetector(weights=out, device="cpu")


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
