"""Tests for the synthetic YOLO dataset builder (tridentnet.data).

All builds are tiny (2 scenes, ~300 pings, 512 slant samples, 256-px chips) so
the full scene-render + preprocess + chip pipeline runs in seconds while still
exercising real geometry."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image

from sonar_core.synth.scene import SceneConfig, SynthTarget
from tridentnet.classes import CLASS_NAMES, CLASS_TO_ID
from tridentnet.data import build_synthetic_dataset

TINY: dict[str, Any] = {
    "n_scenes": 2,
    "chip": 256,
    "seed": 11,
    "val_frac": 0.2,
    "backgrounds_per_scene": 1,
    "n_pings_range": (300, 300),
    "n_samples": 512,
    "altitude_range": (7.0, 9.0),
    "slant_range_range": (40.0, 40.0),
}

_NAME_RE = re.compile(r"^s(\d{3})_(port|starboard)_(t|bg)(\d{2})_r(\d+)_c(\d+)\.(png|txt)$")


def _images(root: Path) -> list[Path]:
    return sorted((root / "images").rglob("*.png"))


def _label_for(root: Path, img: Path) -> Path:
    return root / "labels" / img.parent.name / (img.stem + ".txt")


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("yolo")
    yaml_path = build_synthetic_dataset(root, **TINY)
    return root.resolve(), yaml_path


def test_data_yaml(tiny_dataset: tuple[Path, Path]) -> None:
    root, yaml_path = tiny_dataset
    assert yaml_path.is_file()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    base = Path(data["path"])
    assert base.is_dir()
    assert (base / data["train"]).is_dir()
    assert (base / data["val"]).is_dir()
    names = data["names"]
    assert [names[i] for i in range(len(CLASS_NAMES))] == list(CLASS_NAMES)


def test_labels_exist_and_are_valid(tiny_dataset: tuple[Path, Path]) -> None:
    root, _ = tiny_dataset
    images = _images(root)
    assert images, "build produced no chips"
    n_boxes = 0
    for img in images:
        assert _NAME_RE.match(img.name), f"unexpected chip name {img.name}"
        label = _label_for(root, img)
        assert label.is_file(), f"missing label for {img.name}"
        for line in label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            assert len(fields) == 5, f"malformed line in {label.name}: {line!r}"
            cls_id = int(fields[0])
            assert 0 <= cls_id < len(CLASS_NAMES)
            values = [float(v) for v in fields[1:]]
            assert all(0.0 <= v <= 1.0 for v in values), f"out-of-range values: {line!r}"
            n_boxes += 1
    assert n_boxes > 0, "no non-empty label file in the whole build"


def test_label_geometry_single_known_target(tmp_path: Path) -> None:
    """A label centre must map back (via the chip's global offset encoded in
    its filename) to the target's independently computed survey position."""
    slant_range = 40.0
    n_samples = 512
    target = SynthTarget(
        "cylinder_drum", "starboard", 150, 20.0, 1.6, 1.0, 0.9,
        reflectivity=6.0, shape="ellipse",
    )

    def one_target(cfg: SceneConfig, rng: np.random.Generator) -> list[SynthTarget]:
        return [target]

    yaml_path = build_synthetic_dataset(
        tmp_path,
        n_scenes=1,
        chip=256,
        seed=3,
        val_frac=0.0,
        backgrounds_per_scene=0,
        n_pings_range=(300, 300),
        n_samples=n_samples,
        altitude_range=(8.0, 8.0),
        slant_range_range=(slant_range, slant_range),
        targets_fn=one_target,
    )
    root = Path(yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["path"])
    img_path = next((root / "images" / "train").glob("*_t00_*.png"))
    m = re.search(r"_r(\d+)_c(\d+)\.png$", img_path.name)
    assert m is not None
    r_off, c_off = int(m.group(1)), int(m.group(2))
    with Image.open(img_path) as im:
        win_w, win_h = im.size

    line = _label_for(root, img_path).read_text(encoding="utf-8").splitlines()[0].split()
    cls_id, cx, cy = int(line[0]), float(line[1]), float(line[2])
    assert cls_id == CLASS_TO_ID["cylinder_drum"]

    # Independent geometry: ground_res is the finest side's slant resolution
    # (both sides identical here) = slant_range / n_samples; ping p is centred
    # at row p + 0.5 and ground range g at column g / ground_res.
    ground_res = slant_range / n_samples
    expected_row = target.ping + 0.5
    expected_col = target.ground_range / ground_res
    # Tight tolerances pin the coordinate chain: rows must be exact to the
    # sub-pixel; the column centre is deliberately offset DOWN-range by half
    # the shadow pad (the label includes the near shadow edge), so the offset
    # must be positive and bounded by the pad, never up-range.
    assert abs(r_off + cy * win_h - expected_row) <= 2.0
    col_offset = (c_off + cx * win_w) - expected_col
    assert 0.0 <= col_offset <= 4.0, f"label centre off across-track by {col_offset:.1f} px"


def test_split_is_by_scene(tiny_dataset: tuple[Path, Path]) -> None:
    root, _ = tiny_dataset
    scenes = {"train": set(), "val": set()}
    for img in _images(root):
        scenes[img.parent.name].add(img.name.split("_")[0])
    assert scenes["train"], "train split empty"
    assert scenes["val"], "val split empty"
    assert not scenes["train"] & scenes["val"], "a scene leaked across the split"


def test_deterministic_for_seed(tmp_path: Path) -> None:
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    build_synthetic_dataset(root_a, **TINY)
    build_synthetic_dataset(root_b, **TINY)

    rel_a = sorted(p.relative_to(root_a.resolve()) for p in root_a.resolve().rglob("*") if p.is_file())
    rel_b = sorted(p.relative_to(root_b.resolve()) for p in root_b.resolve().rglob("*") if p.is_file())
    assert rel_a == rel_b, "file lists differ between identical-seed builds"

    labels = [p for p in rel_a if p.suffix == ".txt" and (root_a.resolve() / p).stat().st_size > 0]
    assert labels, "no non-empty label to compare"
    sample = labels[0]
    assert (root_a.resolve() / sample).read_bytes() == (root_b.resolve() / sample).read_bytes()
