"""Tests for the UATD -> YOLO converter's pure logic.

The dataset itself is a 4.8 GB optional download, so these tests exercise
everything that does not need it: the class mapping's integrity against our
frozen taxonomy, VOC parsing, split discovery, and the hard failure on an
unmapped class. That last one is the important guarantee - a future UATD
revision adding a class must stop the build, not silently relabel data.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tridentnet.classes import CLASS_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


UB = _load("build_uatd_dataset")


def test_every_mapping_lands_in_the_frozen_taxonomy() -> None:
    """A typo in CLASS_MAP must fail here, not as a KeyError mid-conversion."""
    for source, target in UB.CLASS_MAP.items():
        assert target is None or target in CLASS_NAMES, (source, target)


def test_dropped_classes_are_explicit() -> None:
    """'rov' has no honest home in our taxonomy and must stay dropped."""
    assert UB.CLASS_MAP["rov"] is None


def test_test2_is_not_a_split() -> None:
    """Test_2 is the reserved untouched holdout; mapping it would spend it."""
    assert set(UB.SPLIT_MAP.values()) == {"train", "val"}
    assert "Test_2" not in UB.SPLIT_MAP


def test_voc_parser_reads_a_standard_annotation(tmp_path) -> None:
    xml = tmp_path / "a.xml"
    xml.write_text(
        "<annotation><object><name>Tyre</name><bndbox>"
        "<xmin>10</xmin><ymin>20</ymin><xmax>110</xmax><ymax>90</ymax>"
        "</bndbox></object>"
        "<object><name>ball</name><bndbox>"
        "<xmin>5</xmin><ymin>5</ymin><xmax>25</xmax><ymax>25</ymax>"
        "</bndbox></object></annotation>",
        encoding="utf-8",
    )
    boxes = UB._voc_boxes(xml)
    # Class names are lower-cased so 'Tyre' and 'tyre' are one class.
    assert boxes == [("tyre", 10.0, 20.0, 110.0, 90.0), ("ball", 5.0, 5.0, 25.0, 25.0)]


def test_voc_parser_survives_garbage(tmp_path) -> None:
    bad = tmp_path / "bad.xml"
    bad.write_text("<annotation><object><name>tyre</name>", encoding="utf-8")
    assert UB._voc_boxes(bad) is None

    empty = tmp_path / "empty.xml"
    empty.write_text("<annotation></annotation>", encoding="utf-8")
    assert UB._voc_boxes(empty) == []


def test_inventory_hard_fails_on_an_unmapped_class(tmp_path) -> None:
    img = tmp_path / "Training" / "images"
    ann = tmp_path / "Training" / "annotations"
    img.mkdir(parents=True)
    ann.mkdir(parents=True)
    from PIL import Image

    Image.new("L", (32, 32)).save(img / "f1.png")
    (ann / "f1.xml").write_text(
        "<annotation><object><name>submarine</name><bndbox>"
        "<xmin>1</xmin><ymin>1</ymin><xmax>9</xmax><ymax>9</ymax>"
        "</bndbox></object></annotation>",
        encoding="utf-8",
    )
    pairs = UB.find_pairs(tmp_path)
    assert pairs["train"], "the crafted pair must be discovered"
    with pytest.raises(SystemExit, match="unmapped"):
        UB.inventory(pairs)


def test_find_pairs_routes_official_splits_and_skips_test2(tmp_path) -> None:
    from PIL import Image

    for official in ("Training", "Test_1", "Test_2"):
        d = tmp_path / official
        d.mkdir()
        Image.new("L", (16, 16)).save(d / "x.bmp")
        (d / "x.xml").write_text("<annotation></annotation>", encoding="utf-8")
    pairs = UB.find_pairs(tmp_path)
    assert len(pairs.get("train", [])) == 1
    assert len(pairs.get("val", [])) == 1
    # Test_2 must not appear anywhere.
    all_paths = [str(i) for items in pairs.values() for i, _x in items]
    assert not any("Test_2" in p for p in all_paths)
