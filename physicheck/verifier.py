"""Stage-2 ML verifier: a gradient-boosted classifier over physics features.

Stage 1 (:class:`physicheck.calibrate.PhysicsGate`) applies hand-set
multipliers to three binary cues; this module learns the *joint* distribution
of the full continuous cue vector (:data:`physicheck.features.FEATURE_NAMES`)
so correlated evidence — a weak highlight but a razor-straight shadow edge
and a broken texture field — is weighed the way an experienced sonar operator
weighs it. The output probability nudges the final confidence multiplier
(see ``physicheck.verify``); it never deletes a detection.

Training needs **no detector and no annotation**: the scene simulator
(:mod:`sonar_core.synth.scene`) knows exactly where every target sits, so
labelled boxes come free —

* positives: truth boxes of man-made targets;
* negatives: truth boxes of *natural* targets (rock_cluster — the hard
  negative that fools intensity-only screening), random background boxes
  (overlap-checked against every target footprint), and boxes inside the
  renderer's sand-ripple band (periodic texture that mimics man-made
  regularity in single features but not in the joint vector).

The train/val split is **by scene, not by box** — boxes from one scene share
a speckle realization and seabed patch field, and splitting them across sets
would leak texture (same doctrine as ``tridentnet.data``). Held-out
accuracy/AUC are stored in the checkpoint so any consumer can audit the model
instead of trusting it.

Like Brain B/C, :meth:`PhysicsVerifier.load` raises ``FileNotFoundError``
when no checkpoint exists — a verifier that silently answers 0.5 would
masquerade as evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from physicheck.features import FEATURE_NAMES, extract_features

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_PATH = _REPO_ROOT / "weights" / "verifier.pkl"

#: Along-track centre / half-width (metres) of the scene renderer's sand
#: ripple band (see ``sonar_core.synth.scene.make_scene``): ripple-band
#: negative boxes are cut from pings whose along-track distance lies inside.
_RIPPLE_CENTRE_M = 100.0
_RIPPLE_HALF_M = 45.0


@dataclass(frozen=True)
class _SampleBox:
    """Duck-typed DetectionLike for truth/negative boxes (no detector ran)."""

    side: str
    ping0: int
    ping1: int
    col0: int
    col1: int
    cls: str
    score: float
    brain: str = "T"  # truth-derived, not a real brain


def _safe_progress(
    progress: Callable[[str, float], None] | None, stage: str, fraction: float
) -> None:
    """Invoke the progress observer; a broken observer must never abort a run."""
    if progress is None:
        return
    try:
        progress(stage, fraction)
    except Exception:  # noqa: BLE001 - observer failures are deliberately swallowed
        pass


def _truth_box(gi, t, cfg, shadow_pad_cols: int, score: float) -> _SampleBox | None:
    """Inclusive integer box for one seeded target on its side's ground image.

    Same geometry as the detector dataset's label boxes
    (``tridentnet.data._target_bbox``): the along-track extent the renderer
    illuminates, the across-track footprint plus *shadow_pad_cols* columns
    down-range so the near shadow edge sits inside the box — matching what a
    trained detector actually outputs. None when the box falls off the swath.
    """
    n_rows, n_cols = gi.side(t.side).shape
    half_len = max(t.length / (2.0 * cfg.speed * cfg.ping_interval), 1.0)
    p0 = int(np.clip(np.floor(t.ping + 0.5 - half_len), 0, n_rows - 1))
    p1 = int(np.clip(np.ceil(t.ping + 0.5 + half_len) - 1, p0, n_rows - 1))
    c0 = int(np.clip(np.floor(gi.col_of_ground_range(t.ground_range - t.width / 2.0) + 0.5),
                     0, n_cols - 1))
    c1 = int(np.clip(np.ceil(gi.col_of_ground_range(t.ground_range + t.width / 2.0) + 0.5)
                     + shadow_pad_cols - 1, c0, n_cols - 1))
    if p1 <= p0 or c1 <= c0:
        return None
    return _SampleBox(t.side, p0, p1, c0, c1, t.cls, score)


def _boxes_overlap(a: _SampleBox, b: _SampleBox) -> bool:
    """Positive-area intersection (IoU > 0) between two same-side boxes."""
    if a.side != b.side:
        return False
    return (
        a.ping0 <= b.ping1 and b.ping0 <= a.ping1
        and a.col0 <= b.col1 and b.col0 <= a.col1
    )


def _random_clear_box(
    gi,
    side: str,
    rng: np.random.Generator,
    taken: list[_SampleBox],
    score: float,
    ping_range: tuple[int, int] | None = None,
    max_tries: int = 40,
    min_finite_frac: float = 0.7,
) -> _SampleBox | None:
    """A random box with zero IoU against every box in *taken*, mostly in-swath.

    *ping_range* confines the draw (used for ripple-band boxes); sizes span
    typical debris box scales so the negative population matches the positive
    one in geometry, differing only in acoustic content.
    """
    img = gi.side(side)
    n_rows, n_cols = img.shape
    p_lo, p_hi = ping_range if ping_range is not None else (0, n_rows - 1)
    p_lo, p_hi = max(p_lo, 0), min(p_hi, n_rows - 1)
    if p_hi - p_lo < 8 or n_cols < 32:
        return None
    for _ in range(max_tries):
        h = int(rng.integers(8, min(40, p_hi - p_lo) + 1))
        w = int(rng.integers(8, min(60, n_cols // 4) + 1))
        p0 = int(rng.integers(p_lo, p_hi - h + 1))
        c0 = int(rng.integers(n_cols // 8, n_cols - w))
        cand = _SampleBox(side, p0, p0 + h - 1, c0, c0 + w - 1, "background", score)
        if any(_boxes_overlap(cand, other) for other in taken):
            continue
        view = img[p0 : p0 + h, c0 : c0 + w]
        if float(np.isfinite(view).mean()) < min_finite_frac:
            continue
        return cand
    return None


def _ripple_ping_range(cfg) -> tuple[int, int] | None:
    """Ping index range covered by the renderer's ripple band, None if off-scene."""
    spacing = cfg.speed * cfg.ping_interval
    p_lo = int(np.ceil((_RIPPLE_CENTRE_M - _RIPPLE_HALF_M) / spacing))
    p_hi = int(np.floor((_RIPPLE_CENTRE_M + _RIPPLE_HALF_M) / spacing))
    p_lo, p_hi = max(p_lo, 0), min(p_hi, cfg.n_pings - 1)
    return (p_lo, p_hi) if p_hi - p_lo >= 8 else None


def train_verifier(
    n_scenes: int = 10,
    seed: int = 0,
    out_path: str | Path = DEFAULT_WEIGHTS_PATH,
    *,
    val_frac: float = 0.25,
    n_pings_range: tuple[int, int] = (700, 1100),
    n_samples: int = 1024,
    altitude_range: tuple[float, float] = (6.0, 12.0),
    slant_range_range: tuple[float, float] = (40.0, 60.0),
    n_bg_boxes: int = 3,
    n_ripple_boxes: int = 2,
    shadow_pad_cols: int = 3,
    threshold: float = 0.5,
    progress: Callable[[str, float], None] | None = None,
) -> Path:
    """Train the Stage-2 verifier on simulator truth and save it with metrics.

    Per scene: draw a random :class:`~sonar_core.synth.scene.SceneConfig`
    (same ranges as the detector dataset builder, so the verifier sees the
    same spread of ground resolutions and shadow geometries), seed a debris
    field with :func:`tridentnet.data.random_targets`, render, preprocess,
    and cut labelled boxes (see module docstring). Each box gets a random raw
    score in [0.3, 0.95] uncorrelated with its label, so ``score_raw`` cannot
    leak the answer — the model must argue from physics.

    Every box runs the *same* :func:`~physicheck.shadow.analyze_shadow` +
    :func:`~physicheck.features.extract_features` path inference uses (shadow
    kwargs from ``configs/physics.yaml``), keeping train and serve identical.

    The last ``round(val_frac * n_scenes)`` scenes (at least 1, never all)
    are held out whole; accuracy and ROC-AUC on them are stored in the
    checkpoint (``val_auc`` is NaN when the val split has a single class —
    reported honestly rather than faked). Deterministic for a given *seed*.

    Returns the checkpoint path (joblib payload: model, feature names,
    threshold, metrics, provenance).
    """
    # Heavy imports kept local: the verifier is loadable (PhysicsVerifier)
    # without pulling the renderer/preprocessing chain into every consumer.
    from physicheck.calibrate import PhysicsGate
    from physicheck.shadow import analyze_shadow
    from sonar_core.preprocess.pipeline import preprocess
    from sonar_core.synth.scene import SceneConfig, make_scene
    from tridentnet.data import random_targets

    if n_scenes < 2:
        raise ValueError("need at least 2 scenes for a scene-level held-out split")
    shadow_kwargs = PhysicsGate().shadow_kwargs()

    rows: list[list[float]] = []
    labels: list[int] = []
    scene_ids: list[int] = []
    for scene_idx in range(n_scenes):
        rng = np.random.default_rng([seed, scene_idx])
        cfg = SceneConfig(
            n_pings=int(rng.integers(n_pings_range[0], n_pings_range[1] + 1)),
            n_samples=int(n_samples),
            slant_range=float(rng.uniform(*slant_range_range)),
            altitude=float(rng.uniform(*altitude_range)),
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        targets = random_targets(cfg, rng)
        pa, targets = make_scene(cfg, targets)
        pre = preprocess(pa)
        gi = pre.ground_raw

        samples: list[tuple[_SampleBox, int]] = []
        truth_boxes: list[_SampleBox] = []
        for t in targets:
            box = _truth_box(gi, t, cfg, shadow_pad_cols, float(rng.uniform(0.3, 0.95)))
            if box is None:
                continue
            truth_boxes.append(box)
            samples.append((box, 0 if t.natural else 1))
        for _ in range(n_bg_boxes):
            side = ("port", "starboard")[int(rng.integers(2))]
            box = _random_clear_box(gi, side, rng, truth_boxes, float(rng.uniform(0.3, 0.95)))
            if box is not None:
                samples.append((box, 0))
        ripple_pings = _ripple_ping_range(cfg)
        if ripple_pings is not None:
            for _ in range(n_ripple_boxes):
                side = ("port", "starboard")[int(rng.integers(2))]
                box = _random_clear_box(
                    gi, side, rng, truth_boxes, float(rng.uniform(0.3, 0.95)),
                    ping_range=ripple_pings,
                )
                if box is not None:
                    samples.append((box, 0))

        for box, label in samples:
            analysis = analyze_shadow(
                gi, box.side, box.ping0, box.ping1, box.col0, box.col1, **shadow_kwargs
            )
            feats = extract_features(gi, box, analysis)
            rows.append([feats[name] for name in FEATURE_NAMES])
            labels.append(label)
            scene_ids.append(scene_idx)
        _safe_progress(progress, f"scene {scene_idx + 1}/{n_scenes}", (scene_idx + 1) / n_scenes)

    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    scenes = np.asarray(scene_ids)
    n_val_scenes = min(max(1, round(val_frac * n_scenes)), n_scenes - 1)
    val_mask = scenes >= n_scenes - n_val_scenes
    if not val_mask.any() or val_mask.all():
        raise ValueError("empty train or val split; increase n_scenes")

    model = GradientBoostingClassifier(random_state=seed)
    model.fit(x[~val_mask], y[~val_mask])

    from sklearn.metrics import accuracy_score, roc_auc_score

    val_y = y[val_mask]
    val_p = model.predict_proba(x[val_mask])[:, 1]
    val_accuracy = float(accuracy_score(val_y, val_p >= threshold))
    val_auc = (
        float(roc_auc_score(val_y, val_p)) if len(np.unique(val_y)) == 2 else float("nan")
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": list(FEATURE_NAMES),
            "threshold": float(threshold),
            "val_auc": val_auc,
            "val_accuracy": val_accuracy,
            "n_train": int((~val_mask).sum()),
            "n_val": int(val_mask.sum()),
            "n_scenes": int(n_scenes),
            "seed": int(seed),
        },
        out_path,
    )
    _safe_progress(progress, "done", 1.0)
    return out_path


class PhysicsVerifier:
    """Inference wrapper: physics feature dict in, P(man-made debris) out.

    Same weight-loading contract as Brains B/C: a missing checkpoint raises
    ``FileNotFoundError`` immediately instead of inventing probabilities.
    """

    def __init__(self, weights: str | Path | None = None) -> None:
        path = Path(weights) if weights is not None else DEFAULT_WEIGHTS_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"verifier weights not found at {path}; run scripts/train_verifier.py"
            )
        payload: dict[str, Any] = joblib.load(path)
        self.model: GradientBoostingClassifier = payload["model"]
        self.feature_names: list[str] = list(payload["feature_names"])
        self.threshold: float = float(payload.get("threshold", 0.5))
        self.val_auc: float = float(payload.get("val_auc", float("nan")))
        self.val_accuracy: float = float(payload.get("val_accuracy", float("nan")))
        if tuple(self.feature_names) != FEATURE_NAMES:
            raise ValueError(
                "verifier checkpoint feature set does not match physicheck.features."
                f"FEATURE_NAMES: {self.feature_names} vs {list(FEATURE_NAMES)}; retrain"
            )

    @classmethod
    def load(cls, weights: str | Path | None = None) -> PhysicsVerifier:
        """Load a trained verifier (default ``weights/verifier.pkl``)."""
        return cls(weights)

    def probability(self, features: dict[str, float]) -> float:
        """P(man-made debris) for one feature dict (ordering by stored names)."""
        row = np.asarray(
            [[float(features[name]) for name in self.feature_names]], dtype=np.float64
        )
        return float(self.model.predict_proba(row)[0, 1])
