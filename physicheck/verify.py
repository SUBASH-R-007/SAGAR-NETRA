"""PhysiCheck orchestrator: raw detections in, physics-verified detections out.

For every candidate box this runs the highlight–shadow analysis on the
*unenhanced* ground image (``PreprocessResult.ground_raw``), applies the
class-conditional plausibility gate, and produces the final calibrated
0–100% confidence. Detections are never deleted here — implausible ones are
demoted and flagged so the operator sees an honest ranking, not a filtered
world view.

Two optional Stage-2 refinements layer on top of the Stage-1 gate, both
additive (with no trained verifier and no thin detections, the confidence is
bit-identical to the Stage-1-only pipeline):

* **ML verifier** — when ``weights/verifier.pkl`` exists (or a
  :class:`~physicheck.verifier.PhysicsVerifier` is passed), the full physics
  feature vector (:mod:`physicheck.features`) is scored and the multiplier
  gains ``clip(verifier_floor + verifier_gain * p, lo, hi)``: an implausible
  cue *combination* demotes further, a strongly debris-like one restores or
  boosts.
* **Temporal persistence gate** — a real seabed object is ensonified by many
  consecutive pings; a detection spanning fewer than
  ``min_persistence_pings`` scan lines is characteristic of impulsive noise
  and gets ``thin_persistence_multiplier`` plus an explicit reason string.

Detections are duck-typed: anything with ``side, ping0, ping1, col0, col1,
cls, score, brain`` attributes works (``tridentnet.detector.Detection`` does).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from physicheck.calibrate import GateResult, PhysicsGate
from physicheck.features import extract_features
from physicheck.shadow import ShadowAnalysis, analyze_shadow
from sonar_core.preprocess.pipeline import PreprocessResult

if TYPE_CHECKING:  # pragma: no cover - typing only; sklearn import stays lazy
    from physicheck.verifier import PhysicsVerifier

#: Stable prefix of the reason attached to detections thinner than
#: ``min_persistence_pings``; the full string states the measured extent vs
#: the threshold so the evidence card never misstates the measurement.
THIN_PERSISTENCE_REASON = "thin return"


@runtime_checkable
class DetectionLike(Protocol):
    side: str
    ping0: int
    ping1: int
    col0: int
    col1: int
    cls: str
    score: float
    brain: str


@dataclass(frozen=True)
class VerifiedDetection:
    """A detection plus its acoustic evidence and final confidence."""

    det: Any  # the original DetectionLike candidate
    analysis: ShadowAnalysis
    gate: GateResult
    confidence_pct: float  # 0-100, temperature-scaled x physics multiplier
    verifier_p: float | None = None  # Stage-2 P(debris); None when no verifier ran
    persistence_pings: int | None = None  # along-track extent in scan lines

    @property
    def cls(self) -> str:
        return self.det.cls

    @property
    def physics_violation(self) -> bool:
        return self.gate.violation

    def cues(self) -> dict[str, Any]:
        """Evidence-card cue list: acoustic cues + provenance + verdict."""
        cues: dict[str, Any] = {
            **self.analysis.cues(),
            "brains": sorted(set(getattr(self.det, "brain", "A"))),
            "physics_violation": self.gate.violation,
            "violation_reason": self.gate.reason,
            "raw_score": round(float(self.det.score), 3),
            "confidence_pct": self.confidence_pct,
        }
        if self.verifier_p is not None:
            cues["verifier_p"] = round(float(self.verifier_p), 3)
        if self.persistence_pings is not None:
            cues["persistence_pings"] = int(self.persistence_pings)
        return cues


def _default_verifier() -> PhysicsVerifier | None:
    """The trained Stage-2 verifier when its default checkpoint exists.

    Returns None (Stage-1-only behaviour, unchanged) when no checkpoint has
    been trained. Imports stay inside the function so consumers that never
    verify with ML never pay the sklearn import.
    """
    from physicheck.verifier import DEFAULT_WEIGHTS_PATH, PhysicsVerifier

    if not DEFAULT_WEIGHTS_PATH.exists():
        return None
    return PhysicsVerifier.load()


def verify_detections(
    detections: list[DetectionLike],
    pre: PreprocessResult,
    gate: PhysicsGate | None = None,
    progress: Callable[[str, float], None] | None = None,
    verifier: PhysicsVerifier | None = None,
    *,
    use_verifier: bool = True,
) -> list[VerifiedDetection]:
    """Run shadow physics + calibration over every candidate.

    When *verifier* is None the default checkpoint (``weights/verifier.pkl``)
    is used if it exists; otherwise the Stage-1 gate runs alone and — for
    detections at least ``min_persistence_pings`` thick — the confidences are
    identical to the pre-verifier pipeline. ``use_verifier=False`` forces
    Stage-1-only scoring even when a checkpoint (or explicit *verifier*)
    exists — ablation studies need the gate-only confidence for the *same*
    detections a full pipeline scores, which no checkpoint shuffling can
    provide race-free. Detections are never deleted: every demotion is a
    multiplier plus a human-readable reason.

    Returns verified detections sorted by descending final confidence.
    """
    gate = gate or PhysicsGate()
    shadow_kwargs = gate.shadow_kwargs()
    if not use_verifier:
        verifier = None
    elif verifier is None:
        verifier = _default_verifier()
    scoring = gate.scoring
    min_persist = int(scoring.get("min_persistence_pings", 3))
    thin_mult = float(scoring.get("thin_persistence_multiplier", 0.6))
    v_floor = float(scoring.get("verifier_floor", 0.4))
    v_gain = float(scoring.get("verifier_gain", 0.8))
    v_lo = float(scoring.get("verifier_clip_lo", 0.4))
    v_hi = float(scoring.get("verifier_clip_hi", 1.2))

    verified: list[VerifiedDetection] = []
    n = max(len(detections), 1)
    for i, det in enumerate(detections):
        analysis = analyze_shadow(
            pre.ground_raw,
            det.side,
            det.ping0,
            det.ping1,
            det.col0,
            det.col1,
            **shadow_kwargs,
        )
        result = gate.evaluate(det.cls, analysis)

        # Temporal persistence gate: real objects span many scan lines.
        persistence = int(det.ping1) - int(det.ping0) + 1
        if persistence < min_persist:
            thin_reason = (
                f"{THIN_PERSISTENCE_REASON} ({persistence} < {min_persist} pings): "
                "likely impulsive noise"
            )
            reason = f"{result.reason}; {thin_reason}" if result.reason else thin_reason
            result = GateResult(result.multiplier * thin_mult, result.violation, reason)

        # Stage-2 ML verifier: learned multiplier from the joint cue vector.
        verifier_p: float | None = None
        if verifier is not None:
            features = extract_features(pre.ground_raw, det, analysis)
            verifier_p = verifier.probability(features)
            factor = max(v_lo, min(v_hi, v_floor + v_gain * verifier_p))
            result = GateResult(result.multiplier * factor, result.violation, result.reason)

        confidence = gate.confidence_pct(float(det.score), result)
        verified.append(
            VerifiedDetection(
                det=det,
                analysis=analysis,
                gate=result,
                confidence_pct=confidence,
                verifier_p=verifier_p,
                persistence_pings=persistence,
            )
        )
        if progress is not None:
            try:
                progress("physics", (i + 1) / n)
            except Exception:  # noqa: BLE001 - observer failures never abort verification
                pass
    verified.sort(key=lambda v: v.confidence_pct, reverse=True)
    return verified
