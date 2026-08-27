"""Deep-ensemble fusion: consensus boosts, disagreement demotes."""

from __future__ import annotations

import pytest

from tridentnet.deep_ensemble import fuse_ensemble
from tridentnet.detector import Detection


def det(score: float, ping0: int = 10, col0: int = 20, cls: str = "container",
        side: str = "starboard") -> Detection:
    return Detection(side=side, ping0=ping0, ping1=ping0 + 10,
                     col0=col0, col1=col0 + 10, cls=cls, score=score)


def test_full_agreement_keeps_mean_score() -> None:
    per_model = [[det(0.9)], [det(0.8)], [det(0.7)]]
    fused = fuse_ensemble(per_model)
    assert len(fused) == 1
    assert fused[0].score == pytest.approx((0.9 + 0.8 + 0.7) / 3)
    assert fused[0].cls == "container"


def test_lone_find_is_demoted_by_missing_models() -> None:
    per_model = [[det(0.9)], [], []]
    fused = fuse_ensemble(per_model)
    assert len(fused) == 1
    assert fused[0].score == pytest.approx(0.9 / 3)


def test_different_classes_or_sides_never_fuse() -> None:
    per_model = [[det(0.9, cls="container")], [det(0.8, cls="tire")],
                 [det(0.7, side="port")]]
    fused = fuse_ensemble(per_model)
    assert len(fused) == 3


def test_min_models_filter() -> None:
    per_model = [[det(0.9)], [], []]
    assert fuse_ensemble(per_model, min_models=2) == []
    per_model = [[det(0.9)], [det(0.8)], []]
    assert len(fuse_ensemble(per_model, min_models=2)) == 1


def test_fused_box_is_weighted_mean() -> None:
    a = det(0.9, ping0=10, col0=20)
    b = Detection("starboard", 11, 21, 21, 31, "container", 0.1)  # IoU 0.70
    fused = fuse_ensemble([[a], [b]])
    assert len(fused) == 1
    # Heavily weighted toward the confident member's geometry.
    assert fused[0].ping0 == 10 and fused[0].col0 == 20


def test_single_model_passthrough() -> None:
    dets = [det(0.5), det(0.4, ping0=100)]
    assert fuse_ensemble([dets]) == dets
    assert fuse_ensemble([]) == []
