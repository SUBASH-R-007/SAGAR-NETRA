"""YOLO-format detection dataset built entirely from the synthetic scene simulator.

SAGAR-NETRA is offline-first: every TridentNet brain must be trainable with no
internet and no proprietary survey data, so this module renders physics-consistent
surveys (:mod:`sonar_core.synth.scene`), runs them through the real M2 preprocessing
chain (:func:`sonar_core.preprocess.pipeline.preprocess`), and cuts detector chips
with YOLO labels from the *enhanced* ground-range imagery — exactly the imagery the
detector will see at inference time. Real public datasets are optional extras
(``scripts/download_datasets.py``) layered on later for domain realism.

Sonar-specific design decisions, all deliberate:

* **Labels include the near shadow edge.** The acoustic shadow is the strongest
  side-scan classification cue (shadow length encodes target height), so every
  label box is padded a few columns *down-range* (toward increasing column) —
  enough for the detector to see the highlight-to-shadow transition without
  swallowing the whole shadow, which can be tens of metres long at low grazing
  angles and would drown the highlight in background pixels.
* **Never mirror-augment across columns.** Both sides are stored nadir-first
  (column 0 at nadir) and shadows always extend toward increasing column; a
  left-right flip would put shadows up-range of their highlights — geometry no
  sonar can produce. ``data.yaml`` carries a warning; training must set
  ``fliplr: 0.0``. Vertical (along-track) flips remain physically valid.
* **Train/val split is by scene, not by chip.** All chips from one scene share
  the same speckle realization, seabed patch field and TVG residue; splitting
  chips across train and val would leak that texture and inflate val mAP.
* **Rare-class oversampling.** ``ghost_net`` — the primary mission class and the
  hardest to spot (low reflectivity, irregular pile) — gets 3x the sampling
  weight of every other class.
* **Hard negatives.** ``rock_cluster`` is the only *sampled* hard negative:
  ``sand_ripple`` and ``reef`` are excluded from :data:`CLASS_SPECS` because
  ripple texture comes free from the scene renderer's ripple band and is not a
  paste target (see :data:`PASTE_EXCLUDED`); those classes keep their ids in
  ``data.yaml`` so the frozen class map is preserved for later real-data
  fine-tuning.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sonar_core.preprocess.pipeline import preprocess
from sonar_core.preprocess.slant_range import GroundImage
from sonar_core.synth.scene import SceneConfig, SynthTarget, make_scene
from tridentnet.classes import CLASS_NAMES, CLASS_TO_ID

SIDES: tuple[str, str] = ("port", "starboard")

#: Classes never pasted as simulator targets: sand ripples are rendered by the
#: scene generator's ripple band (they are a *texture*, not an object with a
#: single highlight/shadow pair), and reef structure is likewise an extended
#: seabed facies that a metre-scale paste target cannot represent. Both keep
#: their label ids for later fine-tuning on real surveys.
PASTE_EXCLUDED: frozenset[str] = frozenset({"sand_ripple", "reef"})


@dataclass(frozen=True)
class ClassSpec:
    """Per-class sampling ranges for simulator paste targets.

    Extents are metres on the seabed: ``length_m`` along-track, ``width_m``
    across-track (ground range), ``height_m`` proud of the seabed — height is
    what casts the shadow, so these ranges drive shadow length realism.
    ``reflectivity`` is the highlight multiplier over local background: hollow
    steel (drums, containers, mines) rings back far harder than water-soaked
    netting or rubber. ``weight`` scales class-sampling probability.
    """

    length_m: tuple[float, float]
    width_m: tuple[float, float]
    height_m: tuple[float, float]
    reflectivity: tuple[float, float]
    shapes: tuple[str, ...]
    weight: float = 1.0
    natural: bool = False


#: Realistic dimension/height/reflectivity ranges per class. ``ghost_net`` is
#: oversampled 3x (rare mission-critical class); ``rock_cluster`` is the only
#: sampled hard negative (see module docstring / :data:`PASTE_EXCLUDED`).
CLASS_SPECS: dict[str, ClassSpec] = {
    "ghost_net": ClassSpec(
        length_m=(4.0, 20.0), width_m=(2.0, 8.0), height_m=(0.4, 2.0),
        reflectivity=(2.4, 4.0), shapes=("irregular",), weight=3.0,
    ),
    "wreck": ClassSpec(
        length_m=(12.0, 35.0), width_m=(4.0, 10.0), height_m=(2.0, 6.0),
        reflectivity=(4.5, 6.5), shapes=("rect", "irregular"),
    ),
    "aircraft": ClassSpec(
        length_m=(8.0, 25.0), width_m=(5.0, 18.0), height_m=(1.5, 4.0),
        reflectivity=(4.0, 6.0), shapes=("rect", "ellipse"),
    ),
    "pipeline": ClassSpec(
        length_m=(20.0, 60.0), width_m=(0.5, 1.2), height_m=(0.3, 1.0),
        reflectivity=(5.0, 7.0), shapes=("rect",),
    ),
    "cylinder_drum": ClassSpec(
        length_m=(0.8, 2.0), width_m=(0.6, 1.4), height_m=(0.6, 1.2),
        reflectivity=(5.0, 7.5), shapes=("ellipse",),
    ),
    "tire": ClassSpec(
        length_m=(0.8, 1.4), width_m=(0.8, 1.4), height_m=(0.2, 0.5),
        reflectivity=(3.0, 4.5), shapes=("ellipse",),
    ),
    "container": ClassSpec(
        length_m=(4.0, 8.0), width_m=(2.0, 3.0), height_m=(1.8, 2.9),
        reflectivity=(5.0, 7.0), shapes=("rect",),
    ),
    "human_body": ClassSpec(
        length_m=(1.4, 2.0), width_m=(0.4, 0.8), height_m=(0.2, 0.45),
        reflectivity=(2.5, 3.5), shapes=("ellipse",),
    ),
    "mine_like": ClassSpec(
        length_m=(0.6, 1.4), width_m=(0.6, 1.2), height_m=(0.3, 0.8),
        reflectivity=(6.0, 8.0), shapes=("ellipse",),
    ),
    "rock_cluster": ClassSpec(
        length_m=(3.0, 8.0), width_m=(2.0, 6.0), height_m=(0.4, 1.5),
        reflectivity=(2.0, 3.0), shapes=("irregular",), natural=True,
    ),
}


def _check_specs() -> None:
    """Fail fast at import if the spec table drifts from the frozen class map."""
    missing = set(CLASS_NAMES) - PASTE_EXCLUDED - set(CLASS_SPECS)
    extra = set(CLASS_SPECS) - set(CLASS_NAMES)
    if missing or extra:
        raise RuntimeError(
            f"CLASS_SPECS out of sync with tridentnet.classes: "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )


_check_specs()


def _targets_overlap(
    a: SynthTarget, b: SynthTarget, cfg: SceneConfig, min_separation_m: float
) -> bool:
    """True when two same-side targets sit closer than *min_separation_m* in
    both along-track and across-track metres (footprint edge to edge)."""
    if a.side != b.side:
        return False
    along = abs(a.ping - b.ping) * cfg.speed * cfg.ping_interval
    across = abs(a.ground_range - b.ground_range)
    return (
        along < (a.length + b.length) / 2.0 + min_separation_m
        and across < (a.width + b.width) / 2.0 + min_separation_m
    )


def random_targets(
    cfg: SceneConfig,
    rng: np.random.Generator,
    n_min: int = 3,
    n_max: int = 8,
    *,
    min_ground_m: float = 4.0,
    max_ground_frac: float = 0.85,
    min_separation_m: float = 3.0,
    max_length_frac: float = 0.5,
    max_tries: int = 60,
) -> list[SynthTarget]:
    """Sample a physically plausible debris field for one scene.

    Placement constraints follow real-survey practice:

    * Ground range is confined to ``[min_ground_m, max_ground_frac * max_ground]``:
      the first few metres past nadir are dominated by the bright first bottom
      return, and the outermost swath has grazing angles too shallow for
      reliable highlights (and its far shadows would run off the swath).
    * ``max_ground`` uses the *largest* expected altitude (mean + wobble), the
      worst case that narrows the swath most.
    * Same-side targets are kept ``min_separation_m`` apart edge-to-edge so
      highlights and shadows never merge into ambiguous ground truth.
    * Along-track extent is capped at ``max_length_frac`` of the survey so a
      pipeline cannot span the whole scene, and each target's centre ping is
      kept far enough from the survey ends that its full extent renders.

    Class choice is weighted by :attr:`ClassSpec.weight` (ghost_net 3x); shape
    variants are drawn from each class's allowed set. A candidate violating the
    separation rule is re-drawn up to *max_tries* times, then dropped, so a
    crowded small scene degrades gracefully instead of looping forever.
    """
    names = [n for n in CLASS_NAMES if n in CLASS_SPECS]
    weights = np.array([CLASS_SPECS[n].weight for n in names], dtype=np.float64)
    weights /= weights.sum()

    alt_hi = cfg.altitude + cfg.altitude_wobble
    max_ground = float(np.sqrt(max(cfg.slant_range**2 - alt_hi**2, 0.0)))
    ground_hi = max_ground_frac * max_ground
    survey_len_m = cfg.n_pings * cfg.speed * cfg.ping_interval

    n = int(rng.integers(n_min, n_max + 1))
    placed: list[SynthTarget] = []
    for _ in range(n):
        for _attempt in range(max_tries):
            name = names[int(rng.choice(len(names), p=weights))]
            spec = CLASS_SPECS[name]
            length = min(float(rng.uniform(*spec.length_m)), max_length_frac * survey_len_m)
            width = float(rng.uniform(*spec.width_m))
            height = float(rng.uniform(*spec.height_m))
            refl = float(rng.uniform(*spec.reflectivity))
            shape = spec.shapes[int(rng.integers(len(spec.shapes)))]
            side = SIDES[int(rng.integers(len(SIDES)))]

            half_len_pings = max(length / (2.0 * cfg.speed * cfg.ping_interval), 1.0)
            margin = int(np.ceil(half_len_pings)) + 1
            if cfg.n_pings - margin <= margin:
                ping = cfg.n_pings // 2
            else:
                ping = int(rng.integers(margin, cfg.n_pings - margin))

            g_lo = min_ground_m + width / 2.0
            g_hi = ground_hi - width / 2.0
            if g_hi <= g_lo:
                continue  # target wider than the usable swath band: re-draw
            ground = float(rng.uniform(g_lo, g_hi))

            cand = SynthTarget(
                name, side, ping, ground, length, width, height,
                reflectivity=refl, natural=spec.natural, shape=shape,
            )
            if not any(_targets_overlap(cand, other, cfg, min_separation_m) for other in placed):
                placed.append(cand)
                break
    return placed


# ---------------------------------------------------------------------------
# chip geometry
# ---------------------------------------------------------------------------


def _target_bbox(
    gi: GroundImage, t: SynthTarget, cfg: SceneConfig, shadow_pad_cols: int
) -> tuple[float, float, float, float] | None:
    """Target footprint as a continuous pixel-space box on its side's image.

    Continuous pixel coordinates: pixel ``(i, j)`` spans rows ``[i, i+1)`` and
    columns ``[j, j+1)``, so ping ``p`` is centred at row ``p + 0.5`` and
    ground range ``g`` sits at column coordinate
    ``col_of_ground_range(g) + 0.5 == g / ground_res``.

    Rows span ``ping +/- length / (2 * speed * ping_interval)`` — the same
    along-track footprint the renderer illuminates. Columns span the physical
    across-track extent plus *shadow_pad_cols* extra columns down-range
    (increasing column) so the near shadow edge — the detector's height cue —
    is inside the box. Returns ``None`` when the clipped box is degenerate
    (target rendered entirely off this side's swath).
    """
    n_rows, n_cols = gi.side(t.side).shape
    half_len = max(t.length / (2.0 * cfg.speed * cfg.ping_interval), 1.0)
    r0 = float(np.clip(t.ping + 0.5 - half_len, 0.0, n_rows))
    r1 = float(np.clip(t.ping + 0.5 + half_len, 0.0, n_rows))
    c0 = float(gi.col_of_ground_range(t.ground_range - t.width / 2.0)) + 0.5
    c1 = float(gi.col_of_ground_range(t.ground_range + t.width / 2.0)) + 0.5 + shadow_pad_cols
    c0 = float(np.clip(c0, 0.0, n_cols))
    c1 = float(np.clip(c1, 0.0, n_cols))
    if r1 <= r0 or c1 <= c0:
        return None
    return r0, r1, c0, c1


def _axis_offset(
    rng: np.random.Generator, b0: float, b1: float, extent: int, win: int, jitter_px: float
) -> int:
    """Window origin along one axis: random within the interval that keeps the
    box ``[b0, b1]`` fully inside a *win*-pixel window, biased to the box centre
    by *jitter_px*. A box larger than the window falls back to centre-clamp."""
    lo = max(0, int(np.ceil(b1)) - win)
    hi = min(extent - win, int(np.floor(b0)))
    centre = int(round((b0 + b1) / 2.0 - win / 2.0))
    if hi < lo:
        return int(np.clip(centre, 0, extent - win))
    j = int(round(jitter_px))
    lo_j, hi_j = max(lo, centre - j), min(hi, centre + j)
    if hi_j < lo_j:
        lo_j, hi_j = lo, hi
    return int(rng.integers(lo_j, hi_j + 1))


def _chip_window(
    rng: np.random.Generator,
    bbox: tuple[float, float, float, float],
    shape: tuple[int, int],
    chip: int,
    jitter_frac: float,
) -> tuple[int, int, int, int]:
    """Jittered chip placement ``(r_off, c_off, win_h, win_w)`` around *bbox*.

    Jitter decorrelates target position from chip centre so the detector cannot
    learn a centre prior; the target box always stays fully inside the window.
    An image smaller than *chip* in an axis yields a smaller window (labels are
    normalized by the actual window size), never padding — synthetic borders
    would add phantom edges for the detector to fire on.
    """
    n_rows, n_cols = shape
    win_h, win_w = min(chip, n_rows), min(chip, n_cols)
    r0, r1, c0, c1 = bbox
    r_off = _axis_offset(rng, r0, r1, n_rows, win_h, jitter_frac * win_h)
    c_off = _axis_offset(rng, c0, c1, n_cols, win_w, jitter_frac * win_w)
    return r_off, c_off, win_h, win_w


def _intersects(
    bbox: tuple[float, float, float, float], window: tuple[int, int, int, int]
) -> bool:
    """True when *bbox* overlaps the window with positive area."""
    r0, r1, c0, c1 = bbox
    r_off, c_off, win_h, win_w = window
    return r1 > r_off and r0 < r_off + win_h and c1 > c_off and c0 < c_off + win_w


def _labels_in_window(
    entries: list[tuple[int, tuple[float, float, float, float]]],
    window: tuple[int, int, int, int],
    min_box_px: float,
) -> list[str]:
    """YOLO label lines for every box intersecting *window*, clipped to it.

    Labelling *all* intersecting targets — not just the one the chip was cut
    for — is essential: an unlabelled neighbour would be trained as background,
    directly punishing correct detections. Slivers thinner than *min_box_px*
    after clipping are dropped (nothing recognizable remains to learn from).
    """
    r_off, c_off, win_h, win_w = window
    lines: list[str] = []
    for cls_id, (r0, r1, c0, c1) in entries:
        ir0, ir1 = max(r0, float(r_off)), min(r1, float(r_off + win_h))
        ic0, ic1 = max(c0, float(c_off)), min(c1, float(c_off + win_w))
        if ir1 - ir0 < min_box_px or ic1 - ic0 < min_box_px:
            continue
        cx = ((ic0 + ic1) / 2.0 - c_off) / win_w
        cy = ((ir0 + ir1) / 2.0 - r_off) / win_h
        bw = (ic1 - ic0) / win_w
        bh = (ir1 - ir0) / win_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def _write_chip(view: np.ndarray, path: Path) -> None:
    """Enhanced ground pixels (finite values in [0, 1], NaN beyond the swath)
    to 8-bit grayscale PNG. NaN maps to 0: out-of-swath fill becomes black,
    matching the darkness of true acoustic shadow rather than inventing
    texture."""
    arr = np.clip(np.nan_to_num(view, nan=0.0), 0.0, 1.0)
    Image.fromarray(np.round(arr * 255.0).astype(np.uint8), mode="L").save(path)


def _write_data_yaml(out_dir: Path) -> Path:
    """Write ``data.yaml`` with the frozen class map and an absolute dataset
    path, plus a prominent no-mirror warning (see module docstring)."""
    lines = [
        "# SAGAR-NETRA synthetic side-scan YOLO dataset",
        "# generated by tridentnet.data.build_synthetic_dataset (simulator-only, offline).",
        "#",
        "# AUGMENTATION WARNING: never mirror across columns — set fliplr: 0.0.",
        "# Both sides are stored nadir-first (column 0 at nadir) and acoustic",
        "# shadows always extend toward increasing column; a left-right flip",
        "# puts shadows up-range of their highlights, which no sonar can produce.",
        f"path: {out_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)]
    path = out_dir / "data.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _safe_progress(
    progress: Callable[[str, float], None] | None, stage: str, fraction: float
) -> None:
    """Invoke the progress observer; a broken observer must never abort a build."""
    if progress is None:
        return
    try:
        progress(stage, fraction)
    except Exception:  # noqa: BLE001 - observer failures are deliberately swallowed
        pass


# ---------------------------------------------------------------------------
# dataset builder
# ---------------------------------------------------------------------------


def build_synthetic_dataset(
    out_dir: str | Path,
    n_scenes: int = 24,
    chip: int = 512,
    seed: int = 0,
    val_frac: float = 0.2,
    backgrounds_per_scene: int = 2,
    progress: Callable[[str, float], None] | None = None,
    *,
    n_pings_range: tuple[int, int] = (700, 1100),
    n_samples: int = 1024,
    altitude_range: tuple[float, float] = (6.0, 12.0),
    slant_range_range: tuple[float, float] = (40.0, 60.0),
    shadow_pad_cols: int = 3,
    jitter_frac: float = 0.25,
    n_min_targets: int = 3,
    n_max_targets: int = 8,
    min_box_px: float = 2.0,
    bg_min_finite: float = 0.5,
    bg_max_tries: int = 40,
    targets_fn: Callable[[SceneConfig, np.random.Generator], list[SynthTarget]] | None = None,
    preprocess_config: dict[str, Any] | None = None,
) -> Path:
    """Build a YOLO detection dataset from *n_scenes* simulated surveys.

    Per scene: draw a random :class:`SceneConfig` (altitude, slant range and
    survey length vary so the detector sees the full spread of ground
    resolutions and shadow geometries the simulator can produce), seed random
    targets, render with :func:`make_scene`, run the full default M2
    :func:`preprocess` chain, then cut one *chip* x *chip* PNG per target from
    the **enhanced** ground image of the target's side — jittered so the target
    is not always centred but always fully inside — plus
    *backgrounds_per_scene* target-free chips with empty label files (explicit
    negatives teach the detector that plain seabed, ripple texture and swath
    edges are background). Every label box is padded *shadow_pad_cols* columns
    down-range to include the near shadow edge, and every target intersecting a
    window is labelled, not just the one the chip was cut for.

    The train/val split assigns whole SCENES (chips from one scene share
    speckle and seabed texture — splitting them across sets would leak).
    ``val_frac`` of scenes (at least one when positive, and never all scenes
    when more than one exists) go to val via a seed-deterministic permutation.

    Chip filenames encode provenance: ``s{scene}_{side}_{t|bg}{k}_r{row}_c{col}``
    where ``r``/``c`` are the global (ping, ground-column) of the chip's
    top-left pixel, so any detection on a chip maps back to survey coordinates
    (and via :class:`GroundImage` to raw slant samples and navigation).

    The build is fully deterministic for a given *seed*: per-scene generators
    are spawned as ``default_rng([seed, scene_idx])`` and the scene renderer
    and preprocessing are themselves seed-stable.

    Parameters beyond the signature docstrings above: *targets_fn* overrides
    :func:`random_targets` (used by tests to place a single known target);
    *preprocess_config* is deep-merged over the M2 defaults; *min_box_px*
    drops clipped label slivers; *bg_min_finite* rejects background windows
    that are mostly out-of-swath NaN; *n_samples* is the per-side slant sample
    count of every rendered scene.

    Returns the path to the written ``data.yaml``.
    """
    out_dir = Path(out_dir).resolve()
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    if val_frac <= 0.0:
        n_val = 0
    else:
        n_val = max(1, round(val_frac * n_scenes))
        # The train split must never be empty: a single-scene build gets no
        # val scenes rather than sending its only scene to val.
        n_val = min(n_val, n_scenes - 1)
    val_scenes = {int(i) for i in np.random.default_rng(seed).permutation(n_scenes)[:n_val]}

    for scene_idx in range(n_scenes):
        rng = np.random.default_rng([seed, scene_idx])
        cfg = SceneConfig(
            n_pings=int(rng.integers(n_pings_range[0], n_pings_range[1] + 1)),
            n_samples=int(n_samples),
            slant_range=float(rng.uniform(*slant_range_range)),
            altitude=float(rng.uniform(*altitude_range)),
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        if targets_fn is not None:
            targets = targets_fn(cfg, rng)
        else:
            targets = random_targets(cfg, rng, n_min=n_min_targets, n_max=n_max_targets)

        pa, targets = make_scene(cfg, targets)
        result = preprocess(pa, config=preprocess_config)
        gi = result.ground

        split = "val" if scene_idx in val_scenes else "train"
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split

        # Padded pixel-space boxes per side, shared by label lookup (every
        # intersecting target gets a line) and background exclusion.
        boxes: dict[str, list[tuple[int, tuple[float, float, float, float]]]] = {
            side: [] for side in SIDES
        }
        for t in targets:
            bb = _target_bbox(gi, t, cfg, shadow_pad_cols)
            if bb is not None:
                boxes[t.side].append((CLASS_TO_ID[t.cls], bb))

        for k, t in enumerate(targets):
            bb = _target_bbox(gi, t, cfg, shadow_pad_cols)
            if bb is None:
                continue
            img = gi.side(t.side)
            window = _chip_window(rng, bb, img.shape, chip, jitter_frac)
            r_off, c_off, win_h, win_w = window
            lines = _labels_in_window(boxes[t.side], window, min_box_px)
            stem = f"s{scene_idx:03d}_{t.side}_t{k:02d}_r{r_off}_c{c_off}"
            _write_chip(img[r_off : r_off + win_h, c_off : c_off + win_w], img_dir / f"{stem}.png")
            (lbl_dir / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )

        for b in range(backgrounds_per_scene):
            for _attempt in range(bg_max_tries):
                side = SIDES[int(rng.integers(len(SIDES)))]
                img = gi.side(side)
                n_rows, n_cols = img.shape
                win_h, win_w = min(chip, n_rows), min(chip, n_cols)
                r_off = int(rng.integers(0, n_rows - win_h + 1))
                c_off = int(rng.integers(0, n_cols - win_w + 1))
                window = (r_off, c_off, win_h, win_w)
                if any(_intersects(bb, window) for _, bb in boxes[side]):
                    continue
                view = img[r_off : r_off + win_h, c_off : c_off + win_w]
                if float(np.isfinite(view).mean()) < bg_min_finite:
                    continue  # mostly out-of-swath fill: not a useful negative
                stem = f"s{scene_idx:03d}_{side}_bg{b:02d}_r{r_off}_c{c_off}"
                _write_chip(view, img_dir / f"{stem}.png")
                (lbl_dir / f"{stem}.txt").write_text("", encoding="utf-8")
                break

        _safe_progress(progress, f"scene {scene_idx + 1}/{n_scenes}", (scene_idx + 1) / n_scenes)

    yaml_path = _write_data_yaml(out_dir)
    _safe_progress(progress, "done", 1.0)
    return yaml_path
