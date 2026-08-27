"""Brain B: masks must land exactly on seeded net footprints, the U-Net must
learn to segment them on a tiny CPU budget, and refinement must tighten only
foreground (net/rope) boxes. Training is module-scoped to keep the suite fast."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sonar_core.preprocess.pipeline import preprocess
from sonar_core.synth.scene import SceneConfig, SynthTarget, make_scene
from tridentnet.detector import Detection
from tridentnet.segdata import build_mask_dataset
from tridentnet.segmenter import Segmenter, UNet, train_segmenter

# One deterministic scene profile shared by every fixture: small enough for a
# busy CPU, wide enough that nets sit fully inside the swath.
SCENE_KW = dict(
    n_pings_range=(240, 240),
    n_samples=384,
    altitude_range=(8.0, 8.0),
    slant_range_range=(40.0, 40.0),
)
GROUND_RES = 40.0 / 384  # slant res == ground res (finest side rule)


def _known_targets(cfg: SceneConfig, rng: np.random.Generator) -> list[SynthTarget]:
    """One high-contrast net + one drum (the empty-mask control chip)."""
    return [
        SynthTarget(
            "ghost_net", "port", 120, 15.0, 8.0, 3.0, 1.2,
            reflectivity=3.6, shape="irregular",
        ),
        SynthTarget(
            "cylinder_drum", "starboard", 80, 20.0, 1.4, 0.9, 0.9,
            reflectivity=6.5, shape="ellipse",
        ),
    ]


def _train_targets(cfg: SceneConfig, rng: np.random.Generator) -> list[SynthTarget]:
    """Four rng-placed high-contrast nets + one drum negative per scene."""
    targets = [
        SynthTarget(
            "ghost_net",
            ("port", "starboard")[int(rng.integers(2))],
            int(rng.integers(40, cfg.n_pings - 40)),
            float(rng.uniform(8.0, 30.0)),
            length=float(rng.uniform(6.0, 10.0)),
            width=float(rng.uniform(2.5, 4.0)),
            height=1.2,
            reflectivity=float(rng.uniform(3.2, 4.0)),
            shape="irregular",
        )
        for _ in range(4)
    ]
    targets.append(
        SynthTarget(
            "cylinder_drum", "starboard", int(rng.integers(40, cfg.n_pings - 40)),
            float(rng.uniform(8.0, 30.0)), 1.4, 0.9, 0.9,
            reflectivity=6.5, shape="ellipse",
        )
    )
    return targets


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path)) > 127


def _find_chip(img_dir: Path, pattern: str) -> Path:
    matches = sorted(img_dir.glob(pattern))
    assert matches, f"no chip matching {pattern} in {img_dir}"
    return matches[0]


@pytest.fixture(scope="module")
def mask_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_mask_dataset(
        tmp_path_factory.mktemp("segdata"),
        n_scenes=1, chip=128, seed=3, val_frac=0.0, net_weight=1,
        targets_fn=_known_targets, **SCENE_KW,
    )


@pytest.fixture(scope="module")
def trained_seg(tmp_path_factory: pytest.TempPathFactory) -> tuple[Segmenter, dict]:
    """Tiny CPU training: 4 scenes (3 train / 1 val), ~1 minute."""
    data = build_mask_dataset(
        tmp_path_factory.mktemp("segtrain"),
        n_scenes=4, chip=96, seed=11, val_frac=0.25, net_weight=1,
        targets_fn=_train_targets, **SCENE_KW,
    )
    out = tmp_path_factory.mktemp("weights") / "segmenter.pt"
    train_segmenter(
        data, out_path=out,
        config={"epochs": 50, "batch": 4, "lr": 2.0e-3},
        device="cpu", seed=0,
    )
    payload = torch.load(out, map_location="cpu", weights_only=False)
    return Segmenter(weights=out, device="cpu"), payload


def test_mask_matches_seeded_footprint(mask_dataset: Path) -> None:
    """The net chip's mask centroid must sit on the seeded net; the drum chip
    (non-foreground class) must have an entirely empty mask."""
    img_dir = mask_dataset / "images" / "train"
    msk_dir = mask_dataset / "masks" / "train"

    net_chip = _find_chip(img_dir, "s000_port_t00_ghost_net_*.png")
    r_off, c_off = (int(v) for v in re.search(r"_r(\d+)_c(\d+)$", net_chip.stem).groups())
    mask = _load_mask(msk_dir / net_chip.name)
    assert mask.any(), "net chip mask has no foreground pixels"

    ys, xs = np.nonzero(mask)
    net = _known_targets(None, None)[0]
    expected_col = net.ground_range / GROUND_RES - 0.5
    assert abs((ys.mean() + r_off) - net.ping) < 4.0, "mask centroid off along-track"
    assert abs((xs.mean() + c_off) - expected_col) < 5.0, "mask centroid off across-track"
    # Footprint extent sanity: rows within ping +/- length/(2*speed*dt) = 20.
    assert ys.min() + r_off >= net.ping - 21 and ys.max() + r_off <= net.ping + 21

    drum_chip = _find_chip(img_dir, "s000_starboard_t01_cylinder_drum_*.png")
    assert not _load_mask(msk_dir / drum_chip.name).any(), "drum chip mask must be empty"


def test_unet_forward_shapes_and_odd_sizes() -> None:
    net = UNet(base=4)
    net.eval()
    with torch.no_grad():
        odd = net(torch.zeros(1, 1, 97, 113))
        even = net(torch.zeros(2, 1, 64, 48))
    assert odd.shape == (1, 1, 97, 113)
    assert even.shape == (2, 1, 64, 48)


def test_training_learns_and_val_dice(trained_seg: tuple[Segmenter, dict]) -> None:
    seg, payload = trained_seg
    losses = payload["train_losses"]
    assert losses[-1] < losses[0], f"training loss did not decrease: {losses}"
    assert payload["val_dice"] > 0.3, f"val Dice too low: {payload['val_dice']:.3f}"
    assert seg.val_dice == pytest.approx(payload["val_dice"])


def test_predict_mask_nan_handling(trained_seg: tuple[Segmenter, dict]) -> None:
    seg, _ = trained_seg
    img = np.random.default_rng(0).random((80, 72)).astype(np.float32)
    img[:, -8:] = np.nan
    mask = seg.predict_mask(img)
    assert mask.shape == img.shape and mask.dtype == np.bool_
    assert not mask[:, -8:].any(), "out-of-swath pixels must never be net"


def test_refine_detections_tightens_net_and_passes_wreck(
    trained_seg: tuple[Segmenter, dict],
) -> None:
    seg, _ = trained_seg
    cfg = SceneConfig(n_pings=200, n_samples=384, slant_range=40.0, altitude=8.0, seed=21)
    net = SynthTarget(
        "ghost_net", "starboard", 100, 15.0, 8.0, 3.5, 1.2,
        reflectivity=3.8, shape="irregular",
    )
    pa, _ = make_scene(cfg, [net])
    pre = preprocess(pa, config={"tiler": {"tile_size": 256}})
    tiles = [t for t in pre.tiles if t.side == "starboard"]

    col_c = net.ground_range / pre.ground.ground_res - 0.5
    half_len = net.length / (2.0 * cfg.speed * cfg.ping_interval)  # 20 pings
    half_w = (net.width / 2.0) / pre.ground.ground_res
    loose = 15  # deliberately sloppy box, as a weak Brain A would draw
    det = Detection(
        side="starboard",
        ping0=int(net.ping - half_len - loose), ping1=int(net.ping + half_len + loose),
        col0=int(col_c - half_w - loose), col1=int(col_c + half_w + loose),
        cls="ghost_net", score=0.7, brain="A",
    )
    wreck = Detection("starboard", 20, 40, 200, 260, "wreck", 0.9, brain="A")

    out = seg.refine_detections([det, wreck], tiles)
    assert len(out) == 2

    refined, mask = out[0]
    assert refined.brain == "AB", "refinement must append B to the brain provenance"
    orig_area = (det.ping1 - det.ping0 + 1) * (det.col1 - det.col0 + 1)
    ref_area = (refined.ping1 - refined.ping0 + 1) * (refined.col1 - refined.col0 + 1)
    assert ref_area < orig_area, "refined box must be tighter than the loose input box"
    margin = int(seg.cfg["refine_margin_px"])
    assert refined.ping0 >= det.ping0 - margin and refined.ping1 <= det.ping1 + margin
    assert refined.col0 >= det.col0 - margin and refined.col1 <= det.col1 + margin
    assert mask is not None and mask.dtype == np.bool_
    assert mask.shape == (
        refined.ping1 - refined.ping0 + 1,
        refined.col1 - refined.col0 + 1,
    )
    assert mask.any()

    passthrough, no_mask = out[1]
    assert passthrough is wreck and no_mask is None, "non-foreground must pass unchanged"
