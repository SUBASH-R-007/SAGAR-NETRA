"""PhysiCheck orchestrator: raw detections in, physics-verified detections out.

For every candidate box this runs the highlight–shadow analysis on the
*unenhanced* ground image (``PreprocessResult.ground_raw``), applies the
class-conditional plausibility gate, and produces the final calibrated
0–100% confidence. Detections are never deleted here — implausible ones are
demoted and flagged so the operator sees an honest ranking, not a filtered
world view.

Detections are duck-typed: anything with ``side, ping0, ping1, col0, col1,
cls, score, brain`` attributes works (``tridentnet.detector.Detection`` does).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from physicheck.calibrate import GateResult, PhysicsGate
from physicheck.shadow import ShadowAnalysis, analyze_shadow
from sonar_core.preprocess.pipeline import PreprocessResult


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

    @property
    def cls(self) -> str:
        return self.det.cls

    @property
    def physics_violation(self) -> bool:
        return self.gate.violation

    def cues(self) -> dict[str, Any]:
        """Evidence-card cue list: acoustic cues + provenance + verdict."""
        return {
            **self.analysis.cues(),
            "brains": sorted(set(getattr(self.det, "brain", "A"))),
            "physics_violation": self.gate.violation,
            "violation_reason": self.gate.reason,
            "raw_score": round(float(self.det.score), 3),
            "confidence_pct": self.confidence_pct,
        }


def verify_detections(
    detections: list[DetectionLike],
    pre: PreprocessResult,
    gate: PhysicsGate | None = None,
    progress: Callable[[str, float], None] | None = None,
) -> list[VerifiedDetection]:
    """Run shadow physics + calibration over every candidate.

    Returns verified detections sorted by descending final confidence.
    """
    gate = gate or PhysicsGate()
    shadow_kwargs = gate.shadow_kwargs()
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
        confidence = gate.confidence_pct(float(det.score), result)
        verified.append(
            VerifiedDetection(det=det, analysis=analysis, gate=result, confidence_pct=confidence)
        )
        if progress is not None:
            try:
                progress("physics", (i + 1) / n)
            except Exception:  # noqa: BLE001 - observer failures never abort verification
                pass
    verified.sort(key=lambda v: v.confidence_pct, reverse=True)
    return verified
