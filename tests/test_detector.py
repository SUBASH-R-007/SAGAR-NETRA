"""TridentNet Brain A (Ultralytics detector wrapper) tests.

No training happens here. The only test touching real model weights is the
smoke test, which auto-downloads yolov8n.pt and skips if that fails; the
tile-to-global mapping arithmetic is verified deterministically against a
fabricated Ultralytics-like result instead of relying on model behaviour.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import tridentnet.detector as detector_mod
from sonar_core.preprocess.tiler import Tile
from tridentnet.detector import Detection, Detector, box_iou, merge_detections


def make_det(
    side: str = "port",
    ping0: int = 0,
    ping1: int = 9,
    col0: int = 0,
    col1: int = 9,
    cls: str = "tire",
    score: float = 0.9,
    tile_index: int = 0,
) -> Detection:
    return Detection(
        side=side,
        ping0=ping0,
        ping1=ping1,
        col0=col0,
        col1=col1,
        cls=cls,
        score=score,
        tile_index=tile_index,
    )


def make_tile(
    side: str = "port",
    row0: int = 0,
    col0: int = 0,
    shape: tuple[int, int] = (64, 80),
    fill: float = 0.5,
    index: int = 0,
) -> Tile:
    return Tile(
        side=side,
        row0=row0,
        col0=col0,
        image=np.full(shape, fill, dtype=np.float32),
        index=index,
    )


def fake_result(
    xyxy: list[list[float]], conf: list[float], cls_ids: list[float], names: dict[int, str]
) -> SimpleNamespace:
    """Minimal stand-in for an Ultralytics Results object."""
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.asarray(xyxy, dtype=np.float64).reshape(-1, 4),
            conf=np.asarray(conf, dtype=np.float64),
            cls=np.asarray(cls_ids, dtype=np.float64),
        ),
        names=names,
    )


# ---------------------------------------------------------------------------
# (1) box_iou on hand-computed inclusive-coordinate cases
# ---------------------------------------------------------------------------


class TestBoxIou:
    def test_identical_boxes(self) -> None:
        a = make_det(ping0=3, ping1=12, col0=5, col1=14)
        assert box_iou(a, a) == pytest.approx(1.0)

    def test_disjoint_boxes(self) -> None:
        a = make_det(ping0=0, ping1=9, col0=0, col1=9)
        b = make_det(ping0=0, ping1=9, col0=10, col1=19)  # touching edge, no shared pixel
        assert box_iou(a, b) == 0.0
        c = make_det(ping0=50, ping1=59, col0=50, col1=59)
        assert box_iou(a, c) == 0.0

    def test_half_overlap(self) -> None:
        # a: rows 0..9, cols 0..9 (100 px); b: rows 0..9, cols 5..14 (100 px).
        # Intersection cols 5..9 -> 5 * 10 = 50 px; union 150 px; IoU = 1/3.
        a = make_det(ping0=0, ping1=9, col0=0, col1=9)
        b = make_det(ping0=0, ping1=9, col0=5, col1=14)
        assert box_iou(a, b) == pytest.approx(1.0 / 3.0)

    def test_single_pixel_boxes(self) -> None:
        # Inclusive coords: a single pixel has area 1, not 0.
        a = make_det(ping0=5, ping1=5, col0=7, col1=7)
        assert box_iou(a, a) == pytest.approx(1.0)
        b = make_det(ping0=5, ping1=5, col0=8, col1=8)  # adjacent pixel
        assert box_iou(a, b) == 0.0
        # Single pixel inside a 2x2 box: intersection 1, union 4.
        c = make_det(ping0=5, ping1=6, col0=7, col1=8)
        assert box_iou(a, c) == pytest.approx(0.25)

    def test_symmetry(self) -> None:
        a = make_det(ping0=0, ping1=19, col0=0, col1=9)
        b = make_det(ping0=10, ping1=29, col0=4, col1=13)
        assert box_iou(a, b) == pytest.approx(box_iou(b, a))


# ---------------------------------------------------------------------------
# (2) merge_detections (cross-tile greedy NMS)
# ---------------------------------------------------------------------------


class TestMergeDetections:
    def test_overlapping_same_class_collapse_to_top_score(self) -> None:
        lo = make_det(score=0.6, tile_index=1)
        hi = make_det(ping0=1, ping1=10, col0=1, col1=10, score=0.9, tile_index=2)
        out = merge_detections([lo, hi], iou_thresh=0.45)
        assert out == [hi]

    def test_identical_boxes_from_two_tiles_dedup(self) -> None:
        a = make_det(score=0.8, tile_index=0)
        b = make_det(score=0.7, tile_index=3)
        out = merge_detections([b, a], iou_thresh=0.45)
        assert len(out) == 1
        assert out[0].score == pytest.approx(0.8)
        assert out[0].tile_index == 0

    def test_different_classes_never_merge(self) -> None:
        a = make_det(cls="tire", score=0.9)
        b = make_det(cls="mine_like", score=0.8)  # identical box, different class
        out = merge_detections([a, b], iou_thresh=0.45)
        assert len(out) == 2
        assert {d.cls for d in out} == {"tire", "mine_like"}

    def test_different_sides_never_merge(self) -> None:
        a = make_det(side="port", score=0.9)
        b = make_det(side="starboard", score=0.8)  # same columns, other side
        out = merge_detections([a, b], iou_thresh=0.45)
        assert len(out) == 2
        assert {d.side for d in out} == {"port", "starboard"}

    def test_chain_suppression_is_not_transitive(self) -> None:
        # A overlaps B, B overlaps C, A and C are disjoint. Greedy NMS keeps
        # A (top score), suppresses B against A, and must keep C: a box is
        # only ever compared against kept boxes, never against suppressed ones.
        a = make_det(col0=0, col1=9, score=0.9)  # rows 0..9
        b = make_det(col0=6, col1=15, score=0.8)  # IoU(A,B) = 40/160 = 0.25
        c = make_det(col0=12, col1=21, score=0.7)  # IoU(B,C) = 0.25, IoU(A,C) = 0
        out = merge_detections([a, b, c], iou_thresh=0.2)
        assert out == [a, c]

    def test_below_threshold_overlap_keeps_both(self) -> None:
        a = make_det(col0=0, col1=9, score=0.9)
        b = make_det(col0=6, col1=15, score=0.8)  # IoU 0.25 < 0.45
        out = merge_detections([a, b], iou_thresh=0.45)
        assert out == [a, b]

    def test_empty_and_singleton(self) -> None:
        assert merge_detections([], iou_thresh=0.45) == []
        only = make_det()
        assert merge_detections([only], iou_thresh=0.45) == [only]

    def test_output_sorted_by_score(self) -> None:
        dets = [
            make_det(col0=100, col1=109, score=0.3),
            make_det(col0=0, col1=9, score=0.9),
            make_det(col0=50, col1=59, score=0.6),
        ]
        out = merge_detections(dets, iou_thresh=0.45)
        assert [d.score for d in out] == sorted((d.score for d in dets), reverse=True)

    def test_bad_threshold_rejected(self) -> None:
        with pytest.raises(ValueError):
            merge_detections([make_det()], iou_thresh=0.0)
        with pytest.raises(ValueError):
            merge_detections([make_det()], iou_thresh=1.5)


# ---------------------------------------------------------------------------
# (3a) deterministic tile -> global mapping via a fabricated predict result
# ---------------------------------------------------------------------------


class TestMappingDeterministic:
    def test_xyxy_maps_to_global_inclusive_extents(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        det = Detector(weights="unused.pt")  # lazy model: never loaded here
        tile = make_tile(side="port", row0=100, col0=40, shape=(64, 80), index=3)
        names = {0: "cylinder_drum", 1: "tire"}
        seen: list[list[np.ndarray]] = []

        def fake_predict(images: list[np.ndarray]) -> list[Any]:
            seen.append(images)
            return [
                fake_result(
                    xyxy=[[10.0, 20.0, 30.0, 50.0], [0.4, 0.0, 79.6, 63.2]],
                    conf=[0.9, 0.5],
                    cls_ids=[0.0, 1.0],
                    names=names,
                )
            ]

        monkeypatch.setattr(det, "_predict", fake_predict)
        out = det.detect_tiles([tile])

        # Input conversion: float [0,1] -> uint8 x3 channels.
        assert len(seen) == 1 and len(seen[0]) == 1
        img = seen[0][0]
        assert img.dtype == np.uint8 and img.shape == (64, 80, 3)

        assert len(out) == 2
        by_cls = {d.cls: d for d in out}
        # Box [10, 20, 30, 50): cols floor(10)..ceil(30)-1 = 10..29 local,
        # rows 20..49 local; tile origin (row0=100, col0=40) shifts to global.
        d0 = by_cls["cylinder_drum"]
        assert (d0.ping0, d0.ping1, d0.col0, d0.col1) == (120, 149, 50, 69)
        assert d0.score == pytest.approx(0.9)
        assert d0.side == "port" and d0.brain == "A" and d0.tile_index == 3
        # Fractional box [0.4, 0.0, 79.6, 63.2): floor/ceil-1 then clamp keeps
        # the extent inside the tile: cols 0..79, rows 0..63 local.
        d1 = by_cls["tire"]
        assert (d1.ping0, d1.ping1, d1.col0, d1.col1) == (100, 163, 40, 119)

    def test_nan_swath_becomes_zero_uint8(self, monkeypatch: pytest.MonkeyPatch) -> None:
        det = Detector(weights="unused.pt")
        image = np.full((32, 32), 1.0, dtype=np.float32)
        image[:, 24:] = np.nan  # out-of-swath fill
        tile = Tile(side="starboard", row0=0, col0=0, image=image, index=0)
        captured: list[np.ndarray] = []

        def fake_predict(images: list[np.ndarray]) -> list[Any]:
            captured.extend(images)
            return [fake_result(xyxy=[], conf=[], cls_ids=[], names={})]

        monkeypatch.setattr(det, "_predict", fake_predict)
        assert det.detect_tiles([tile]) == []
        img = captured[0]
        assert img[:, :24].min() == 255  # finite 1.0 -> full scale
        assert img[:, 24:].max() == 0  # NaN swath -> water-column blank level

    def test_cross_tile_duplicates_dedup_in_global_coords(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        det = Detector(weights="unused.pt")
        # Two overlapping tiles of the same side; the same physical target is
        # seen by both at different local coordinates but one global footprint.
        tile_a = make_tile(side="port", row0=0, col0=0, shape=(64, 64), index=0)
        tile_b = make_tile(side="port", row0=0, col0=32, shape=(64, 64), index=1)
        names = {0: "container"}

        def fake_predict(images: list[np.ndarray]) -> list[Any]:
            assert len(images) == 2  # both tiles fit in one batch (batch=8)
            return [
                # local cols 40..59 -> global cols 40..59, score 0.8
                fake_result([[40.0, 10.0, 60.0, 30.0]], [0.8], [0.0], names),
                # local cols 8..27 -> global cols 40..59, score 0.6
                fake_result([[8.0, 10.0, 28.0, 30.0]], [0.6], [0.0], names),
            ]

        monkeypatch.setattr(det, "_predict", fake_predict)
        stages: list[tuple[str, float]] = []
        out = det.detect_tiles([tile_a, tile_b], progress=lambda s, f: stages.append((s, f)))

        assert len(out) == 1
        d = out[0]
        assert (d.ping0, d.ping1, d.col0, d.col1) == (10, 29, 40, 59)
        assert d.score == pytest.approx(0.8)
        assert d.tile_index == 0  # the higher-scoring copy wins

        # Progress observer contract: nondecreasing fractions, final done=1.0.
        fracs = [f for _, f in stages]
        assert all(0.0 <= f <= 1.0 for f in fracs)
        assert fracs == sorted(fracs)
        assert stages[-1] == ("done", 1.0)

    def test_empty_tile_list(self) -> None:
        det = Detector(weights="unused.pt")
        stages: list[tuple[str, float]] = []
        assert det.detect_tiles([], progress=lambda s, f: stages.append((s, f))) == []
        assert stages == [("done", 1.0)]


# ---------------------------------------------------------------------------
# (3b) end-to-end smoke with the real (auto-downloaded) pretrained model
# ---------------------------------------------------------------------------


def _real_detector() -> Detector:
    det = Detector(weights=None)
    try:
        det.class_names()  # forces the (possibly downloading) model load
    except Exception as exc:  # noqa: BLE001 - any load failure means skip, not fail
        pytest.skip(f"pretrained model unavailable: {exc}")
    return det


class TestRealModelSmoke:
    def test_detect_tiles_returns_valid_global_detections(self) -> None:
        det = _real_detector()
        rng = np.random.default_rng(26057)
        image = (0.15 + 0.05 * rng.random((640, 640))).astype(np.float32)
        # Bright blob with a dark "shadow" toward increasing column, the way a
        # proud sonar target presents; enough contrast that the COCO model may
        # or may not fire — the assertions hold either way.
        image[280:360, 200:280] = 0.95
        image[280:360, 280:420] = 0.03
        image[:, 600:] = np.nan  # swath edge
        tile = Tile(side="starboard", row0=1000, col0=300, image=image, index=5)

        out = det.detect_tiles([tile])
        assert isinstance(out, list)
        valid_names = set(det.class_names())
        for d in out:
            assert isinstance(d, Detection)
            assert d.side == "starboard" and d.brain == "A" and d.tile_index == 5
            assert 1000 <= d.ping0 <= d.ping1 <= 1000 + 639
            assert 300 <= d.col0 <= d.col1 <= 300 + 639
            assert 0.0 <= d.score <= 1.0
            assert d.cls in valid_names

    def test_class_names_are_ordered_strings(self) -> None:
        det = _real_detector()
        names = det.class_names()
        assert isinstance(names, list) and names
        assert all(isinstance(n, str) for n in names)


# ---------------------------------------------------------------------------
# (4) pretrained-fallback construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_fallback_warning_without_trained_weights(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Any,
    ) -> None:
        monkeypatch.setattr(detector_mod, "DEFAULT_WEIGHTS_PATH", tmp_path / "absent.pt")
        with caplog.at_level(logging.WARNING, logger="tridentnet.detector"):
            det = Detector(weights=None)
        assert det.using_pretrained_fallback
        assert det.weights == det.config["model"] == "yolov8n.pt"
        assert "sonar classes are unavailable" in caplog.text

    def test_trained_weights_preferred_when_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Any,
    ) -> None:
        trained = tmp_path / "detector.pt"
        trained.write_bytes(b"placeholder")  # existence is all __init__ checks
        monkeypatch.setattr(detector_mod, "DEFAULT_WEIGHTS_PATH", trained)
        with caplog.at_level(logging.WARNING, logger="tridentnet.detector"):
            det = Detector(weights=None)
        assert not det.using_pretrained_fallback
        assert det.weights == str(trained)
        assert caplog.text == ""

    def test_config_deep_merge(self) -> None:
        det = Detector(weights="unused.pt", config={"conf": 0.5, "batch": 2})
        assert det.config["conf"] == 0.5
        assert det.config["batch"] == 2
        assert det.config["imgsz"] == 640  # YAML/default survives partial override
        assert det.config["dedup_iou"] == 0.45
