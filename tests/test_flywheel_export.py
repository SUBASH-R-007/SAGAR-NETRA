"""Active-learning flywheel export (blueprint N-08): a tiny synthetic survey
is processed with a stub detector, one contact confirmed and one rejected,
then scripts/export_review_labels.py must emit YOLO chips + labels — the
confirmed one with its frozen class id, the rejected one as an empty-label
hard negative — plus a data.yaml ready for train_detector.py --data."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from PIL import Image

from api.db import ContactRepo
from api.processing import process_survey
from geoscribe.contact import ReviewStatus
from tridentnet.classes import CLASS_TO_ID
from tridentnet.detector import Detection

REPO_ROOT = Path(__file__).resolve().parents[1]

CHIP = 96  # tiny survey, tiny chips: keep the CPU free for the trainings


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


export_mod = _load_script("export_review_labels")


class StubDetector:
    """One deterministic reportable box per side (test_api pattern)."""

    def detect_tiles(self, tiles, progress=None):
        by_side = {}
        for tile in tiles:
            by_side.setdefault(tile.side, tile)
        detections = []
        for side, tile in by_side.items():
            h, w = tile.image.shape
            r0, c0 = tile.row0 + h // 3, tile.col0 + w // 3
            detections.append(
                Detection(
                    side=side, ping0=r0, ping1=r0 + 12, col0=c0, col1=c0 + 18,
                    cls="container", score=0.82, tile_index=tile.index,
                )
            )
        if progress:
            progress("detect", 1.0)
        return detections


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> dict[str, Any]:
    from sonar_core.parsers.xtf_writer import write_xtf
    from sonar_core.synth.scene import SceneConfig, SynthTarget, make_scene

    root = tmp_path_factory.mktemp("flywheel")
    cfg = SceneConfig(n_pings=160, n_samples=256, slant_range=40.0, seed=23)
    targets = [
        SynthTarget("container", "starboard", 80, 20.0, 3.0, 2.0, 1.5, reflectivity=5.5)
    ]
    pa, _ = make_scene(cfg, targets)
    source = write_xtf(pa, root / "survey_fly.xtf")

    db_path = root / "contacts.db"
    repo = ContactRepo(db_path)
    process_survey(source, repo, detector_factory=StubDetector, output_root=root / "outputs")
    contacts = repo.query(limit=10)
    assert len(contacts) >= 2, "stub detector should yield one contact per side"
    confirmed, rejected = contacts[0], contacts[1]
    repo.set_review(confirmed.id, ReviewStatus.confirmed, notes="real container")
    repo.set_review(rejected.id, ReviewStatus.rejected, notes="rock, not debris")
    repo.close()

    out = root / "export"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            sys, "argv",
            [
                "export_review_labels.py",
                "--db", str(db_path),
                "--out", str(out),
                "--chip", str(CHIP),
            ],
        )
        rc = export_mod.main()
    assert rc == 0
    return {"out": out, "confirmed": confirmed, "rejected": rejected}


def test_chips_written_for_both_verdicts(exported: dict[str, Any]) -> None:
    images = sorted((exported["out"] / "images").glob("*.png"))
    assert {p.stem for p in images} == {exported["confirmed"].id, exported["rejected"].id}
    for path in images:
        with Image.open(path) as img:
            assert img.size == (CHIP, CHIP)
            assert img.mode == "L"


def test_confirmed_label_is_normalized_yolo_box(exported: dict[str, Any]) -> None:
    confirmed = exported["confirmed"]
    text = (exported["out"] / "labels" / f"{confirmed.id}.txt").read_text().strip()
    parts = text.split()
    assert len(parts) == 5, f"expected one YOLO line, got {text!r}"
    assert int(parts[0]) == CLASS_TO_ID[confirmed.cls]
    cx, cy, bw, bh = (float(v) for v in parts[1:])
    assert all(0.0 < v <= 1.0 for v in (cx, cy, bw, bh))
    # The chip is centred on the box, so the box centre sits mid-chip.
    assert cx == pytest.approx(0.5, abs=0.05)
    assert cy == pytest.approx(0.5, abs=0.05)
    # Labels carry 6 decimals, so compare at that precision.
    assert bw == pytest.approx((confirmed.pixel.col1 - confirmed.pixel.col0 + 1) / CHIP, abs=1e-5)
    assert bh == pytest.approx((confirmed.pixel.ping1 - confirmed.pixel.ping0 + 1) / CHIP, abs=1e-5)


def test_rejected_label_is_empty_hard_negative(exported: dict[str, Any]) -> None:
    label = exported["out"] / "labels" / f"{exported['rejected'].id}.txt"
    assert label.is_file()
    assert label.read_text() == ""


def test_data_yaml_has_frozen_class_map(exported: dict[str, Any]) -> None:
    text = (exported["out"] / "data.yaml").read_text(encoding="utf-8")
    assert "names:" in text
    assert "0: ghost_net" in text
    assert f"{CLASS_TO_ID['container']}: container" in text
    assert "fliplr" in text  # the no-mirror warning must travel with the data
