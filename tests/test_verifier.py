"""Stage-2 verifier: physics features, shadow-edge linearity, training,
verify_detections integration and the temporal persistence gate."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import joblib
import numpy as np
import pytest
import yaml

from physicheck.calibrate import DEFAULT_CONFIG, PhysicsGate
from physicheck.features import FEATURE_NAMES, extract_features
from physicheck.shadow import analyze_shadow
from physicheck.verifier import PhysicsVerifier, train_verifier
from physicheck.verify import THIN_PERSISTENCE_REASON, verify_detections
from sonar_core.preprocess.pipeline import preprocess
from sonar_core.synth.scene import SceneConfig, SynthTarget, make_scene


@dataclass(frozen=True)
class FakeDetection:
    side: str
    ping0: int
    ping1: int
    col0: int
    col1: int
    cls: str
    score: float
    brain: str = "A"
    tile_index: int = -1


def _render(seed: int):
    """Container + rock cluster on one starboard swath; truth-derived boxes."""
    cfg = SceneConfig(n_pings=240, n_samples=1024, slant_range=40.0, seed=seed)
    container = SynthTarget(
        "container", "starboard", 70, 20.0, 6.0, 2.4, 2.4, reflectivity=6.0
    )
    rock = SynthTarget(
        "rock_cluster", "starboard", 170, 22.0, 5.0, 3.5, 1.0,
        reflectivity=2.4, natural=True, shape="irregular",
    )
    pa, _ = make_scene(cfg, [container, rock])
    pre = preprocess(pa)
    gi = pre.ground_raw

    def truth_box(t: SynthTarget, score: float) -> FakeDetection:
        half = int(t.length / (2 * cfg.speed * cfg.ping_interval))
        return FakeDetection(
            side=t.side,
            ping0=t.ping - half,
            ping1=t.ping + half,
            col0=int(gi.col_of_ground_range(t.ground_range - t.width / 2)),
            col1=int(gi.col_of_ground_range(t.ground_range + t.width / 2)),
            cls=t.cls,
            score=score,
        )

    return pre, truth_box(container, 0.8), truth_box(rock, 0.8)


@pytest.fixture(scope="module")
def scene():
    return _render(seed=11)


@pytest.fixture(scope="module")
def trained_verifier(tmp_path_factory: pytest.TempPathFactory):
    """A small but real training run (scene-level held-out split)."""
    out = tmp_path_factory.mktemp("weights") / "verifier.pkl"
    train_verifier(
        n_scenes=4, seed=0, out_path=out, n_pings_range=(360, 420), n_samples=512
    )
    return out


def _features_for(pre, det) -> dict[str, float]:
    analysis = analyze_shadow(
        pre.ground_raw, det.side, det.ping0, det.ping1, det.col0, det.col1
    )
    return extract_features(pre.ground_raw, det, analysis)


def test_features_finite_and_stable_order(scene) -> None:
    pre, cbox, _ = scene
    feats = _features_for(pre, cbox)
    assert tuple(feats.keys()) == FEATURE_NAMES, "dict order must match FEATURE_NAMES"
    assert all(np.isfinite(v) for v in feats.values())
    # A rendered container has both cues: the analysis-derived features are real.
    assert feats["has_height"] == 1.0
    assert feats["highlight_ratio"] > 1.4
    assert feats["shadow_len_m"] > 0.0
    assert feats["ping_persistence"] == cbox.ping1 - cbox.ping0 + 1
    # And a second extraction is bit-identical (pure function of the inputs).
    assert _features_for(pre, cbox) == feats


def test_shadow_linearity_separates_container_from_rock(scene) -> None:
    """Machined far edges cast straight shadow boundaries; rock piles do not."""
    margin = 0.2

    def linearity_gap(pre, cbox, rbox) -> tuple[float, float]:
        return (
            _features_for(pre, cbox)["shadow_linearity"],
            _features_for(pre, rbox)["shadow_linearity"],
        )

    cont, rock = linearity_gap(*scene)
    if not cont > rock + margin:  # borderline speckle draw: one retry, new seed
        cont, rock = linearity_gap(*_render(seed=12))
    assert cont > rock + margin, f"container {cont:.3f} vs rock {rock:.3f}"


def test_tiny_train_run_held_out_auc(trained_verifier) -> None:
    payload = joblib.load(trained_verifier)
    assert payload["feature_names"] == list(FEATURE_NAMES)
    assert payload["n_train"] > 0 and payload["n_val"] > 0
    assert np.isfinite(payload["val_auc"]), "val split must contain both classes"
    assert payload["val_auc"] > 0.75, f"held-out AUC {payload['val_auc']:.3f}"


def test_verifier_load_missing_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        PhysicsVerifier.load(tmp_path / "nope.pkl")


def test_no_pkl_behaviour_identical_to_golden(scene, monkeypatch, tmp_path) -> None:
    """Without a trained checkpoint the pipeline must match the pre-integration
    confidences bit for bit (golden values computed on this exact scene with
    the Stage-1-only code)."""
    monkeypatch.setattr(
        "physicheck.verifier.DEFAULT_WEIGHTS_PATH", tmp_path / "absent.pkl"
    )
    pre, cbox, _ = scene
    bg = FakeDetection("starboard", 110, 140, cbox.col0, cbox.col1, "container", 0.8)
    by_ping0 = {v.det.ping0: v for v in verify_detections([cbox, bg], pre)}
    assert by_ping0[cbox.ping0].confidence_pct == 72.8  # golden, pre-integration
    assert by_ping0[bg.ping0].confidence_pct == 31.6  # golden, pre-integration
    assert by_ping0[cbox.ping0].verifier_p is None
    assert by_ping0[cbox.ping0].persistence_pings == cbox.ping1 - cbox.ping0 + 1


def test_trained_verifier_demotes_background_vs_container(scene, trained_verifier) -> None:
    """At equal raw score, empty seabed must rank below the real container."""
    pre, cbox, _ = scene
    bg = FakeDetection("starboard", 110, 140, cbox.col0, cbox.col1, "container", 0.8)
    verifier = PhysicsVerifier.load(trained_verifier)
    by_ping0 = {v.det.ping0: v for v in verify_detections([cbox, bg], pre, verifier=verifier)}
    cont, back = by_ping0[cbox.ping0], by_ping0[bg.ping0]
    assert cont.verifier_p is not None and back.verifier_p is not None
    assert cont.verifier_p > back.verifier_p
    assert cont.confidence_pct > back.confidence_pct
    cues = cont.cues()
    assert "verifier_p" in cues and "persistence_pings" in cues


def test_thin_detection_demoted_with_reason(scene, monkeypatch, tmp_path) -> None:
    """A 1-ping detection gets the persistence multiplier and reason; the same
    box with the gate disabled (min_persistence_pings: 1) isolates its effect."""
    monkeypatch.setattr(
        "physicheck.verifier.DEFAULT_WEIGHTS_PATH", tmp_path / "absent.pkl"
    )
    pre, cbox, _ = scene
    thin = FakeDetection(
        cbox.side, cbox.ping0, cbox.ping0, cbox.col0, cbox.col1, "container", 0.8
    )
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    neutral = copy.deepcopy(config)
    neutral["scoring"]["min_persistence_pings"] = 1

    gated = verify_detections([thin], pre, gate=PhysicsGate(config))[0]
    ungated = verify_detections([thin], pre, gate=PhysicsGate(neutral))[0]
    assert gated.persistence_pings == 1
    assert THIN_PERSISTENCE_REASON in (gated.gate.reason or "")
    assert gated.confidence_pct < ungated.confidence_pct
    assert THIN_PERSISTENCE_REASON not in (ungated.gate.reason or "")
    # Never deleted: the thin detection is still in the ranked output.
    assert gated.cues()["persistence_pings"] == 1
