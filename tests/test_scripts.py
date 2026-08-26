"""Unit tests for the CLI scripts' pure helpers (no training, no ultralytics).

The scripts are loaded by path (scripts/ is not a package); everything heavy
(torch, ultralytics) is imported lazily inside their run functions, so these
imports stay cheap and offline.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from sonar_core.parsers.base import NAV_DTYPE
from sonar_core.preprocess.slant_range import GroundImage
from tridentnet.detector import Detection

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


train_detector = _load_script("train_detector")
detect_demo = _load_script("detect_demo")


def _ground_image(n_pings: int = 8, n_port: int = 6, n_stbd: int = 4) -> GroundImage:
    rng = np.random.default_rng(3)
    return GroundImage(
        port=rng.random((n_pings, n_port)).astype(np.float32),
        starboard=rng.random((n_pings, n_stbd)).astype(np.float32),
        ground_res=0.5,
        altitude_m=np.full(n_pings, 8.0, dtype=np.float32),
        slant_res={"port": 0.5, "starboard": 0.5},
        nav=np.zeros(n_pings, dtype=NAV_DTYPE),
    )


def _det(side: str, col0: int, col1: int) -> Detection:
    return Detection(side=side, ping0=1, ping1=3, col0=col0, col1=col1, cls="tire", score=0.5)


class TestTrainResolve:
    def test_smoke_profile(self) -> None:
        args = train_detector.build_parser().parse_args(["--smoke"])
        params = train_detector.resolve(args)
        assert (params.scenes, params.epochs, params.imgsz, params.batch) == (6, 3, 320, 4)
        assert params.name == "detector_smoke"
        assert params.dest.name == "detector_smoke.pt"
        assert params.pings_range == train_detector.SMOKE_PINGS_RANGE

    def test_full_defaults(self) -> None:
        params = train_detector.resolve(train_detector.build_parser().parse_args([]))
        assert (params.scenes, params.epochs, params.imgsz, params.batch) == (24, 40, 640, 8)
        assert params.name == "detector"
        assert params.dest.name == "detector.pt"
        assert params.pings_range is None

    def test_explicit_overrides_beat_smoke(self) -> None:
        args = train_detector.build_parser().parse_args(["--smoke", "--epochs", "5"])
        assert train_detector.resolve(args).epochs == 5

    def test_map50_extraction(self) -> None:
        metrics = SimpleNamespace(box=SimpleNamespace(map50=0.42))
        assert train_detector.extract_map50(metrics) == 0.42
        fallback = SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.17})
        assert train_detector.extract_map50(fallback) == 0.17
        assert train_detector.extract_map50(object()) is None


class TestDemoGeometry:
    def test_port_x_span_is_mirrored(self) -> None:
        # Port col 0 is the nadir and must land just left of the centreline.
        x0, x1 = detect_demo.det_x_span(_det("port", 0, 4), n_port_cols=10)
        assert (x0, x1) == (5, 9)
        assert x0 <= x1

    def test_starboard_x_span_is_offset(self) -> None:
        assert detect_demo.det_x_span(_det("starboard", 0, 4), n_port_cols=10) == (10, 14)

    def test_combined_layout(self) -> None:
        gi = _ground_image()
        combo = detect_demo.combined_enhanced(gi)
        assert combo.shape == (gi.n_pings, gi.n_cols("port") + gi.n_cols("starboard"))
        # Leftmost column is port far range; centre-left is port nadir (col 0).
        np.testing.assert_array_equal(combo[:, 0], gi.port[:, -1])
        np.testing.assert_array_equal(combo[:, gi.n_cols("port") - 1], gi.port[:, 0])
        np.testing.assert_array_equal(combo[:, gi.n_cols("port")], gi.starboard[:, 0])

    def test_class_color_deterministic(self) -> None:
        assert detect_demo.class_color("tire") == detect_demo.class_color("tire")
        color = detect_demo.class_color("not_a_sonar_class")
        assert color in detect_demo.PALETTE

    def test_table_lists_every_detection(self) -> None:
        gi = _ground_image()
        dets = [_det("port", 1, 3), _det("starboard", 0, 2)]
        table = detect_demo.format_table(dets, gi)
        assert table.count("tire") == 2
        assert "port" in table and "starboard" in table
        # Ground range of col 1..3 at 0.5 m/col: centres 0.75..1.75 m.
        assert "0.8-1.8" in table

    def test_render_writes_png(self, tmp_path: Path) -> None:
        gi = _ground_image(n_pings=64, n_port=32, n_stbd=32)
        out = detect_demo.render_detections(gi, [_det("port", 2, 6)], tmp_path / "demo.png")
        assert out.is_file()
        from PIL import Image

        with Image.open(out) as img:
            assert img.size == (64, 64)
