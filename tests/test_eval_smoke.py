"""Smoke test for the ablation evaluation (scripts/eval_detector.py).

A stub detector replaces the deployed stack (no model weights load) so the
test exercises the evaluation *plumbing*: scene rendering, truth-box
geometry, the four re-scoring configurations, greedy matching, metric
arithmetic and the markdown table. The stub emits, per side, one solid box
and one 2-ping-thin box — the thin one is exactly what the temporal
persistence gate exists to demote, so the full deployed config (d) must
never keep more false positives above the confidence floor than the raw
detector (a).

Needs the trained Stage-2 checkpoint (``weights/verifier.pkl``): config (c)
must score with the real verifier, and the eval refuses to silently skip it.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tridentnet.detector import Detection

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_WEIGHTS = REPO_ROOT / "weights" / "verifier.pkl"


def _load_script(name: str) -> ModuleType:
    """Load a CLI script by path (scripts/ is not a package; test_scripts pattern)."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module

pytestmark = pytest.mark.skipif(
    not VERIFIER_WEIGHTS.exists(),
    reason="weights/verifier.pkl not trained yet (run scripts/train_verifier.py)",
)


class StubDetector:
    """Two deterministic boxes per side: one solid, one thin (2 pings).

    The thin box carries a raw score whose calibrated confidence sits just
    above the 50% floor, so the raw config counts it as a false alarm while
    the physics/temporal configs demote it below the floor.
    """

    def detect_tiles(self, tiles: list, progress=None) -> list[Detection]:
        by_side: dict[str, object] = {}
        for tile in tiles:
            by_side.setdefault(tile.side, tile)
        detections: list[Detection] = []
        for side, tile in by_side.items():
            h, w = tile.image.shape
            r0, c0 = tile.row0 + h // 3, tile.col0 + w // 3
            detections.append(
                Detection(side=side, ping0=r0, ping1=r0 + 12, col0=c0, col1=c0 + 18,
                          cls="container", score=0.9, tile_index=tile.index)
            )
            r1, c1 = tile.row0 + h // 2, tile.col0 + w // 4
            detections.append(
                Detection(side=side, ping0=r1, ping1=r1 + 1, col0=c1, col1=c1 + 9,
                          cls="cylinder_drum", score=0.55, tile_index=tile.index)
            )
        return detections


@pytest.fixture(scope="module")
def eval_result(tmp_path_factory: pytest.TempPathFactory):
    run_eval = _load_script("eval_detector").run_eval

    out_path = tmp_path_factory.mktemp("metrics") / "ablation.md"
    metrics = run_eval(
        n_scenes=2,
        out_path=out_path,
        detector=StubDetector(),
        n_pings_range=(220, 280),
        n_samples=512,
    )
    return metrics, out_path


def test_table_written_with_honest_preamble(eval_result) -> None:
    _, out_path = eval_result
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "SYNTHETIC" in text  # the honesty caveat must travel with the table
    for label in ("raw detector", "physics gate", "ML verifier", "temporal"):
        assert label in text


def test_metrics_are_valid_fractions(eval_result) -> None:
    metrics, _ = eval_result
    assert set(metrics) == {"a_raw", "b_gate", "c_verifier", "d_full"}
    for m in metrics.values():
        for value in (m.precision, m.recall, m.f1, m.pr_auc):
            assert 0.0 <= value <= 1.0
        assert math.isfinite(m.fp_per_km2) and m.fp_per_km2 >= 0.0
        assert m.tp + m.fp <= m.n_detections


def test_same_raw_detections_in_every_config(eval_result) -> None:
    """Multipliers demote, never delete: the scored population is constant."""
    metrics, _ = eval_result
    counts = {m.n_detections for m in metrics.values()}
    assert len(counts) == 1 and counts.pop() > 0


def test_full_pipeline_never_noisier_than_raw(eval_result) -> None:
    metrics, _ = eval_result
    assert metrics["d_full"].fp <= metrics["a_raw"].fp
