"""PS-named acquisition-artifact augmentations (sonar_core.synth.artifacts):
shape/dtype/shadow-direction preservation, per-seed determinism, and the
label-safe ``artifact_aug`` path through the dataset builder.

Dataset builds are tiny (1 scene, 300 pings, 512 slant samples, 256-px chips)
so the full render + preprocess + chip pipeline stays fast while exercising
real geometry."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from sonar_core.synth.artifacts import (
    LABEL_SAFE_ARTIFACTS,
    apply_artifacts,
    heave_banding,
    ping_dropout,
    pitch_stretch,
    resolution_jitter,
    roll_shear,
)
from sonar_core.synth.scene import SceneConfig, SynthTarget
from tridentnet.classes import CLASS_TO_ID
from tridentnet.data import build_synthetic_dataset

ALL_ARTIFACTS = [heave_banding, pitch_stretch, roll_shear, ping_dropout, resolution_jitter]


def _textured_chip(dtype: type = np.uint8) -> np.ndarray:
    """Deterministic speckle-like chip so resampling artifacts have texture
    to act on (a constant image is a fixed point of every resample)."""
    rng = np.random.default_rng(99)
    img = rng.uniform(20.0, 220.0, size=(64, 96))
    if np.issubdtype(np.dtype(dtype), np.integer):
        return np.round(img).astype(dtype)
    return img.astype(dtype)


# ---------------------------------------------------------------------------
# shared contracts: shape, dtype, determinism, input immutability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", ALL_ARTIFACTS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["u8", "f32"])
def test_shape_dtype_preserved_and_input_untouched(fn, dtype) -> None:
    img = _textured_chip(dtype)
    before = img.copy()
    out = fn(img, np.random.default_rng(5))
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    assert out is not img
    np.testing.assert_array_equal(img, before)  # input never modified


@pytest.mark.parametrize("fn", ALL_ARTIFACTS, ids=lambda f: f.__name__)
def test_deterministic_per_rng_seed(fn) -> None:
    img = _textured_chip()
    a = fn(img, np.random.default_rng(42))
    b = fn(img, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


def test_apply_artifacts_deterministic_and_seed_sensitive() -> None:
    img = _textured_chip()
    a = apply_artifacts(img, np.random.default_rng(7), p_each=1.0)
    b = apply_artifacts(img, np.random.default_rng(7), p_each=1.0)
    c = apply_artifacts(img, np.random.default_rng(8), p_each=1.0)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c), "different seeds produced identical artifacts"
    assert a.shape == img.shape and a.dtype == img.dtype


def test_apply_artifacts_p_zero_is_identity_copy() -> None:
    img = _textured_chip()
    out = apply_artifacts(img, np.random.default_rng(0), p_each=0.0)
    np.testing.assert_array_equal(out, img)
    assert out is not img


def test_apply_artifacts_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown artifact"):
        apply_artifacts(_textured_chip(), np.random.default_rng(0), names=("mirror_flip",))


def test_never_reverses_column_axis() -> None:
    """Full artifact stack, 20 draws, on a left-bright/right-dark chip: the
    left (near-nadir) half must stay brighter in every sample — proof that no
    artifact mirrors or reorders the across-track (shadow) axis."""
    img = np.full((64, 64), 30, dtype=np.uint8)
    img[:, :32] = 200
    for k in range(20):
        out = apply_artifacts(img, np.random.default_rng(k), p_each=1.0)
        assert float(out[:, :32].mean()) > float(out[:, 32:].mean())


# ---------------------------------------------------------------------------
# per-artifact physics
# ---------------------------------------------------------------------------


def test_heave_changes_row_means_but_not_column_ordering() -> None:
    """Constant-row column ramp: heave must vary the ROW means (banding) while
    each row keeps its exact within-row ordering (the gain is one positive
    scalar per row)."""
    ramp = np.tile(np.linspace(10.0, 200.0, 96, dtype=np.float32), (64, 1))
    out = heave_banding(ramp, np.random.default_rng(3))
    assert float(np.std(out.mean(axis=1))) > 0.0, "no banding: all row means identical"
    assert np.all(np.diff(out, axis=1) >= 0.0), "column ordering broken within a row"


def test_ping_dropout_full_rows_are_zero_and_bounded() -> None:
    img = np.full((100, 40), 100, dtype=np.uint8)
    out = ping_dropout(img, np.random.default_rng(11), partial_p=0.0)
    zero_rows = np.flatnonzero((out == 0).all(axis=1))
    assert zero_rows.size >= 1, "no row was blanked"
    assert zero_rows.size <= round(0.06 * 100), "blanked more rows than max_rows_frac allows"
    untouched = np.setdiff1d(np.arange(100), zero_rows)
    np.testing.assert_array_equal(out[untouched], img[untouched])


def test_resolution_jitter_same_shape_and_softens() -> None:
    """A hard downsample (scale 0.5 both axes) must keep the shape, keep the
    dtype, and actually lose high-frequency content (pixels change)."""
    img = _textured_chip(np.float32)
    out = resolution_jitter(img, np.random.default_rng(2), scale=(0.5, 0.5))
    assert out.shape == img.shape and out.dtype == img.dtype
    assert not np.array_equal(out, img), "downsample round trip changed nothing"


def test_roll_shear_shift_is_bounded() -> None:
    """Single bright column through roll_shear: every row's brightness must
    stay within max_px/2 (+1 interp px) of the original column — the documented
    bound that keeps highlight-shadow adjacency intact — and never up-range of
    physical possibility (no mirroring, just a small shift)."""
    img = np.zeros((64, 96), dtype=np.float32)
    img[:, 30] = 1.0
    max_px = 6.0
    out = roll_shear(img, np.random.default_rng(17), max_px=max_px)
    nz_rows, nz_cols = np.nonzero(out > 1e-6)
    assert nz_rows.size > 0
    assert np.all(np.abs(nz_cols - 30) <= max_px / 2.0 + 1.0)


def test_pitch_stretch_touches_rows_only() -> None:
    """A vertical (all-rows) bright column must survive pitch_stretch exactly
    in place: the transform resamples the ROW axis only."""
    img = np.zeros((64, 96), dtype=np.float32)
    img[:, 40] = 1.0
    out = pitch_stretch(img, np.random.default_rng(21))
    nz_cols = np.unique(np.nonzero(out > 1e-6)[1])
    np.testing.assert_array_equal(nz_cols, [40])


# ---------------------------------------------------------------------------
# dataset builder integration (label-safe path)
# ---------------------------------------------------------------------------

SLANT_RANGE = 40.0
N_SAMPLES = 512
KNOWN_TARGET = SynthTarget(
    "cylinder_drum", "starboard", 150, 20.0, 1.6, 1.0, 0.9,
    reflectivity=6.0, shape="ellipse",
)


def _one_target(cfg: SceneConfig, rng: np.random.Generator) -> list[SynthTarget]:
    return [KNOWN_TARGET]


def _build(root: Path, **extra) -> Path:
    yaml_path = build_synthetic_dataset(
        root,
        n_scenes=1,
        chip=256,
        seed=3,
        val_frac=0.0,
        backgrounds_per_scene=2,
        n_pings_range=(300, 300),
        n_samples=N_SAMPLES,
        altitude_range=(8.0, 8.0),
        slant_range_range=(SLANT_RANGE, SLANT_RANGE),
        targets_fn=_one_target,
        **extra,
    )
    return Path(yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["path"])


@pytest.fixture(scope="module")
def artifact_builds(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("artifact_builds")
    return {
        "default": _build(base / "default"),
        "off": _build(base / "off", artifact_aug=0.0),
        "on": _build(base / "on", artifact_aug=1.0),
    }


def _files(root: Path) -> list[Path]:
    return sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())


def test_default_build_byte_identical_to_artifact_aug_zero(
    artifact_builds: dict[str, Path],
) -> None:
    """artifact_aug defaults to 0.0 and 0.0 must be a true no-op: every file
    (chips included) byte-identical between the two builds."""
    default, off = artifact_builds["default"], artifact_builds["off"]
    rel = _files(default)
    assert rel == _files(off)
    for p in rel:
        if p.name == "data.yaml":
            continue  # embeds the absolute build path, differs by construction
        assert (default / p).read_bytes() == (off / p).read_bytes(), f"{p} differs"


def test_artifact_aug_changes_pixels_only(artifact_builds: dict[str, Path]) -> None:
    """artifact_aug=1.0 draws from per-chip generators, never the scene
    generator: file list and every label byte-identical to the clean build,
    but at least one chip's pixels must differ."""
    off, on = artifact_builds["off"], artifact_builds["on"]
    rel = _files(off)
    assert rel == _files(on), "artifact_aug changed windows/filenames"
    for p in (p for p in rel if p.suffix == ".txt"):
        assert (off / p).read_bytes() == (on / p).read_bytes(), f"label {p} changed"
    pngs = [p for p in rel if p.suffix == ".png"]
    assert pngs
    assert any(
        (off / p).read_bytes() != (on / p).read_bytes() for p in pngs
    ), "artifact_aug=1.0 left every chip untouched"


def test_artifact_build_labels_still_valid_geometry(
    artifact_builds: dict[str, Path],
) -> None:
    """The known target's label in the artifact_aug=1.0 build must still map
    back to its survey position (same tolerances as the clean-build geometry
    test): the label-safe set (heave/dropout/resolution jitter) displaces no
    pixel, so the boxes stay true."""
    assert set(LABEL_SAFE_ARTIFACTS) == {"heave_banding", "ping_dropout", "resolution_jitter"}
    root = artifact_builds["on"]
    img_path = next((root / "images" / "train").glob("*_t00_*.png"))
    m = re.search(r"_r(\d+)_c(\d+)\.png$", img_path.name)
    assert m is not None
    r_off, c_off = int(m.group(1)), int(m.group(2))
    with Image.open(img_path) as im:
        win_w, win_h = im.size

    label = root / "labels" / "train" / (img_path.stem + ".txt")
    line = label.read_text(encoding="utf-8").splitlines()[0].split()
    assert len(line) == 5
    cls_id, cx, cy = int(line[0]), float(line[1]), float(line[2])
    assert cls_id == CLASS_TO_ID["cylinder_drum"]
    assert all(0.0 <= float(v) <= 1.0 for v in line[1:])

    ground_res = SLANT_RANGE / N_SAMPLES
    expected_row = KNOWN_TARGET.ping + 0.5
    expected_col = KNOWN_TARGET.ground_range / ground_res
    assert abs(r_off + cy * win_h - expected_row) <= 2.0
    col_offset = (c_off + cx * win_w) - expected_col
    assert 0.0 <= col_offset <= 4.0, f"label centre off across-track by {col_offset:.1f} px"
