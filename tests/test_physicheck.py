"""PhysiCheck: shadow physics on rendered scenes, plausibility gating,
temperature calibration, and evidence cards."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from physicheck.calibrate import (
    PhysicsGate,
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    reliability_diagram,
)
from physicheck.evidence import render_evidence_card
from physicheck.shadow import analyze_shadow
from physicheck.verify import verify_detections
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


@pytest.fixture(scope="module")
def container_scene():
    """One container (H = 2.0 m) rendered and preprocessed; box coords derived
    from the seeded geometry, not from a detector."""
    cfg = SceneConfig(n_pings=80, n_samples=1024, slant_range=40.0, seed=3)
    target = SynthTarget(
        "container", "starboard", 40, 20.0, 6.0, 2.0, height=2.0, reflectivity=6.0
    )
    pa, _ = make_scene(cfg, [target])
    pre = preprocess(pa)

    half_pings = int(target.length / (2 * cfg.speed * cfg.ping_interval))
    gi = pre.ground_raw
    box = FakeDetection(
        side="starboard",
        ping0=target.ping - half_pings,
        ping1=target.ping + half_pings,
        col0=int(gi.col_of_ground_range(target.ground_range - target.width / 2)),
        col1=int(gi.col_of_ground_range(target.ground_range + target.width / 2)),
        cls="container",
        score=0.8,
    )
    return pre, target, box


def test_shadow_analysis_recovers_height(container_scene) -> None:
    pre, target, box = container_scene
    analysis = analyze_shadow(
        pre.ground_raw, box.side, box.ping0, box.ping1, box.col0, box.col1
    )
    assert analysis.has_highlight, f"highlight ratio {analysis.highlight_ratio}"
    assert analysis.has_shadow, f"shadow len {analysis.shadow_len_m}"
    assert analysis.shadow_ratio < 0.5
    assert analysis.height_m == pytest.approx(target.height, rel=0.25)


def test_gate_bonus_violation_and_missing_cues(container_scene) -> None:
    pre, _, box = container_scene
    gate = PhysicsGate()
    analysis = analyze_shadow(
        pre.ground_raw, box.side, box.ping0, box.ping1, box.col0, box.col1
    )
    # Plausible container: bonus.
    ok = gate.evaluate("container", analysis)
    assert not ok.violation and ok.multiplier > 1.0

    # Same physical evidence claimed as a tire (max 1.2 m): violation, demoted.
    bad = gate.evaluate("tire", analysis)
    assert bad.violation and bad.multiplier < 0.5
    assert "outside" in (bad.reason or "")
    assert gate.confidence_pct(0.8, bad) < gate.confidence_pct(0.8, ok)

    # Empty seabed box: no highlight -> weak-evidence multiplier, no violation.
    empty = analyze_shadow(pre.ground_raw, box.side, 5, 15, box.col0, box.col1)
    weak = gate.evaluate("container", empty)
    assert not weak.violation and weak.multiplier < 1.0


def test_temperature_fitting_improves_calibration() -> None:
    rng = np.random.default_rng(0)
    # Overconfident scores: true P(correct) is much lower than the raw score.
    raw = rng.uniform(0.55, 0.99, size=4000)
    true_p = 0.5 + (raw - 0.5) * 0.45
    correct = rng.uniform(size=raw.size) < true_p
    t = fit_temperature(raw, correct)
    assert t > 1.0, f"overconfident scores need T > 1, got {t}"
    ece_before = expected_calibration_error(raw, correct)
    ece_after = expected_calibration_error(np.asarray(apply_temperature(raw, t)), correct)
    assert ece_after < ece_before * 0.5


def test_reliability_diagram_written(tmp_path) -> None:
    rng = np.random.default_rng(1)
    scores = rng.uniform(size=500)
    correct = rng.uniform(size=500) < scores
    out = reliability_diagram(scores, correct, tmp_path / "reliability.png")
    assert out.exists() and out.stat().st_size > 0


def test_verify_detections_sorted_and_flagged(container_scene) -> None:
    pre, _, box = container_scene
    dets = [
        box,  # good container
        FakeDetection(  # same box claimed as tire -> physics violation
            box.side, box.ping0, box.ping1, box.col0, box.col1, "tire", 0.8
        ),
        FakeDetection(box.side, 5, 15, box.col0, box.col1, "container", 0.8),  # empty
    ]
    verified = verify_detections(dets, pre)
    assert [v.confidence_pct for v in verified] == sorted(
        (v.confidence_pct for v in verified), reverse=True
    )
    assert verified[0].cls == "container" and not verified[0].physics_violation
    flags = {v.cls: v.physics_violation for v in verified if v.det.ping0 == box.ping0}
    assert flags["tire"] is True
    cues = verified[0].cues()
    assert {"highlight", "shadow", "height_m", "brains", "confidence_pct"} <= cues.keys()


def test_evidence_card_rendered(container_scene, tmp_path) -> None:
    pre, _, box = container_scene
    verified = verify_detections([box], pre)[0]
    png = render_evidence_card(pre, verified, tmp_path / "cards" / "SN-0001.png")
    assert png.exists() and png.stat().st_size > 0
    assert png.with_suffix(".json").exists()
    import json

    cues = json.loads(png.with_suffix(".json").read_text())
    assert cues["highlight"] is True and cues["shadow"] is True
