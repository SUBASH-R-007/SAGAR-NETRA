"""Confidence calibration: temperature scaling plus physics-derived multipliers.

Neural detectors are systematically overconfident; temperature scaling
(Guo et al., 2017) is the one-parameter fix — divide the logit by a
temperature ``T`` fitted on validation data by minimizing negative
log-likelihood. The final SAGAR-NETRA confidence is::

    confidence% = 100 * clip(sigmoid(logit(raw) / T) * physics_multiplier, 0, 1)

where the multiplier comes from the highlight/shadow plausibility gate. All
knobs live in ``configs/physics.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import minimize_scalar

from physicheck.shadow import ShadowAnalysis

_EPS = 1e-6
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "physics.yaml"


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(scores: np.ndarray, correct: np.ndarray) -> float:
    """Fit T by NLL minimization on validation (score, was-it-correct) pairs."""
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if scores.shape != correct.shape or scores.size == 0:
        raise ValueError("scores and correct must be equal-length, non-empty arrays")
    z = _logit(scores)

    def nll(t: float) -> float:
        p = np.clip(_sigmoid(z / t), _EPS, 1.0 - _EPS)
        return float(-np.mean(correct * np.log(p) + (1.0 - correct) * np.log(1.0 - p)))

    result = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(result.x)


def apply_temperature(scores: np.ndarray | float, temperature: float) -> np.ndarray | float:
    """Rescale raw confidences by the fitted temperature."""
    single = np.isscalar(scores)
    out = _sigmoid(_logit(np.atleast_1d(np.asarray(scores, dtype=np.float64))) / temperature)
    return float(out[0]) if single else out


def expected_calibration_error(
    scores: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> float:
    """Standard ECE: confidence-weighted mean |accuracy - confidence| per bin."""
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (scores > lo) & (scores <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - scores[mask].mean())
    return float(ece)


def reliability_diagram(
    scores: np.ndarray,
    correct: np.ndarray,
    out_path: str | Path,
    n_bins: int = 10,
    title: str = "Reliability diagram",
) -> Path:
    """Save the standard reliability plot (accuracy vs confidence per bin)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres, accuracy = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (scores > lo) & (scores <= hi)
        if mask.any():
            centres.append((lo + hi) / 2)
            accuracy.append(correct[mask].mean())

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.bar(centres, accuracy, width=1.0 / n_bins * 0.9, alpha=0.7, label="observed")
    ece = expected_calibration_error(scores, correct, n_bins)
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title(f"{title} (ECE = {ece:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


@dataclass(frozen=True)
class GateResult:
    multiplier: float
    violation: bool
    reason: str | None


class PhysicsGate:
    """Class-conditional plausibility gate driven by ``configs/physics.yaml``."""

    def __init__(self, config: dict[str, Any] | str | Path | None = None) -> None:
        if config is None or isinstance(config, (str, Path)):
            path = Path(config) if config is not None else DEFAULT_CONFIG
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.plausibility: dict[str, dict[str, float]] = config["plausibility"]
        self.scoring: dict[str, Any] = config["scoring"]

    def shadow_kwargs(self) -> dict[str, Any]:
        return dict(self.scoring.get("shadow_analysis", {}))

    @property
    def temperature(self) -> float:
        return float(self.scoring.get("temperature", 1.0))

    def evaluate(self, cls: str, analysis: ShadowAnalysis) -> GateResult:
        """Physics multiplier for one detection's cue set."""
        if not analysis.has_highlight:
            return GateResult(
                float(self.scoring["no_highlight_multiplier"]),
                False,
                "no acoustic highlight above local seabed",
            )
        if not analysis.has_shadow or not np.isfinite(analysis.height_m):
            return GateResult(
                float(self.scoring["no_shadow_multiplier"]),
                False,
                "no measurable shadow: height unverified",
            )
        band = self.plausibility.get(cls)
        if band is not None:
            lo, hi = float(band["min_height_m"]), float(band["max_height_m"])
            if not lo <= analysis.height_m <= hi:
                return GateResult(
                    float(self.scoring["violation_multiplier"]),
                    True,
                    f"height {analysis.height_m:.2f} m outside {cls} band [{lo}, {hi}] m",
                )
        return GateResult(float(self.scoring["both_cues_bonus"]), False, None)

    def confidence_pct(self, raw_score: float, gate: GateResult) -> float:
        """Final 0-100% confidence: temperature-scaled, physics-multiplied."""
        calibrated = float(apply_temperature(raw_score, self.temperature))
        return round(100.0 * float(np.clip(calibrated * gate.multiplier, 0.0, 1.0)), 1)
