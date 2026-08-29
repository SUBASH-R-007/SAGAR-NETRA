"""Does the advantage hold as the seabed gets messy? A clutter sweep.

Usage:
    python scripts/eval_clutter.py [--scenes 8] [--levels 0 6 12 24]
        [--out docs/clutter_sweep.md]

``scripts/eval_baseline.py`` compares SAGAR-NETRA with a classical CAD baseline
on the standard held-out scenes and finds them close — the baseline slightly
ahead. That result is real but it is not what it looks like, and this script
exists to show why.

**The confound.** In the scene simulator, ``rock_cluster`` — the only natural
clutter class — is given reflectivity 2.0-3.0, the lowest of any class, while
most man-made targets sit at 4.0-8.0. Brightness therefore *is* the man-made /
natural label for most of the catalogue. A detector that thresholds on
brightness gets the answer handed to it by the data generator. Real sonar
offers no such gap: a granite boulder and a steel drum can return comparable
amplitude, which is the entire reason the problem needs shape, shadow geometry
and learning rather than a threshold.

So this script sweeps clutter under **two** conditions:

- ``native`` — rocks keep their catalogue reflectivity, brightness gap intact.
  This is the regime the head-to-head comparison implicitly ran in.
- ``matched`` — each decoy rock is given the reflectivity of a randomly chosen
  *man-made* target from the same scene. The amplitude distributions become
  identical by construction, so brightness carries **zero** information about
  whether an object is debris, and only shape, shadow geometry and learned
  discrimination can separate them.

The debris field is held fixed and clutter levels are nested: same scene, same
targets, same seed, more rocks. Every rock is a false positive by construction
(truth boxes are man-made only), so the sweep measures exactly one thing: how
fast does precision fall as target-shaped natural objects accumulate?

If SAGAR-NETRA's advantage is real it should appear as a gentler slope under
``matched``. If it does not appear there either, that is worth knowing before
anyone claims it from a stage.

Everything here is SYNTHETIC; the caveat is written into the output markdown.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geoscribe.build import survey_stats  # noqa: E402
from physicheck.calibrate import PhysicsGate  # noqa: E402
from physicheck.verifier import PhysicsVerifier  # noqa: E402
from physicheck.verify import verify_detections  # noqa: E402
from sonar_core.preprocess.pipeline import preprocess  # noqa: E402
from sonar_core.synth.scene import SceneConfig, SynthTarget, make_scene  # noqa: E402
from tridentnet.baseline import ClassicalCAD, ClassicalConfig  # noqa: E402
from tridentnet.data import CLASS_SPECS, random_targets  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "docs" / "clutter_sweep.md"

TUNE_SEED_BASE = 11_000
DEPLOYED_FLOOR_PCT = 50.0

#: Extra rock clusters layered onto each scene. Nested: level N contains every
#: rock of level N-1 and the debris field never changes.
DEFAULT_LEVELS: tuple[int, ...] = (0, 6, 12, 24)

#: Clutter brightness conditions. See the module docstring — ``matched`` is the
#: one that answers the question; ``native`` is kept so the confound is visible
#: rather than merely asserted.
MODES: tuple[str, ...] = ("native", "matched")

K_SIGMA_GRID: tuple[float, ...] = (
    0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0, 25.0
)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ED = _load("eval_detector")
EB = _load("eval_baseline")


def _rock_pool(
    cfg: SceneConfig,
    rng: np.random.Generator,
    existing: list[SynthTarget],
    n_rocks: int,
    mode: str,
    *,
    min_ground_m: float = 4.0,
    max_ground_frac: float = 0.85,
    min_separation_m: float = 3.0,
    max_tries: int = 200,
) -> list[SynthTarget]:
    """Decoy rock clusters, placed under the same rules as the debris field.

    Rocks are kept clear of *man-made* targets so no truth box is corrupted by
    a natural object growing into it — a rock overlapping another rock is fine
    and realistic, but a rock merged into a drum would make the ground truth
    itself ambiguous and quietly poison both methods' scores.

    Under ``mode="matched"`` each rock borrows the reflectivity of a real
    target in this scene, which is what removes brightness as a shortcut.
    """
    spec = CLASS_SPECS["rock_cluster"]
    alt_hi = cfg.altitude + cfg.altitude_wobble
    max_ground = float(np.sqrt(max(cfg.slant_range**2 - alt_hi**2, 0.0)))
    ground_hi = max_ground_frac * max_ground
    manmade = [t for t in existing if not t.natural]
    donors = [t.reflectivity for t in manmade] or None

    rocks: list[SynthTarget] = []
    for _ in range(n_rocks):
        for _attempt in range(max_tries):
            length = float(rng.uniform(*spec.length_m))
            width = float(rng.uniform(*spec.width_m))
            side = ("port", "starboard")[int(rng.integers(2))]

            half_len_pings = max(length / (2.0 * cfg.speed * cfg.ping_interval), 1.0)
            margin = int(np.ceil(half_len_pings)) + 1
            if cfg.n_pings - margin <= margin:
                continue
            ping = int(rng.integers(margin, cfg.n_pings - margin))

            g_lo, g_hi = min_ground_m + width / 2.0, ground_hi - width / 2.0
            if g_hi <= g_lo:
                continue
            ground = float(rng.uniform(g_lo, g_hi))

            # Both draws happen in both modes, always in this order. Branching
            # on the draw instead would consume different amounts of the bit
            # stream per mode and silently move every subsequent rock, which
            # would destroy the one thing this experiment controls for: the two
            # modes must differ in brightness and in nothing else.
            native_refl = float(rng.uniform(*spec.reflectivity))
            donor_idx = int(rng.integers(len(donors))) if donors else 0
            reflectivity = (
                float(donors[donor_idx]) if (mode == "matched" and donors) else native_refl
            )

            cand = SynthTarget(
                cls="rock_cluster", side=side, ping=ping, ground_range=ground,
                length=length, width=width,
                height=float(rng.uniform(*spec.height_m)),
                reflectivity=reflectivity, natural=True, shape="irregular",
            )
            if _clear_of(cand, manmade, cfg, min_separation_m):
                rocks.append(cand)
                break
    return rocks


def _clear_of(
    cand: SynthTarget, others: list[SynthTarget], cfg: SceneConfig, sep_m: float
) -> bool:
    """True when *cand* overlaps none of *others* in both range and along-track."""
    for other in others:
        if other.side != cand.side:
            continue
        range_gap = abs(cand.ground_range - other.ground_range) - (
            cand.width + other.width
        ) / 2.0
        along_m = abs(cand.ping - other.ping) * cfg.speed * cfg.ping_interval
        along_gap = along_m - (cand.length + other.length) / 2.0
        if range_gap < sep_m and along_gap < sep_m:
            return False
    return True


def _classical_labels(pre, truths, cfg: ClassicalConfig, iou: float) -> list:
    found = ClassicalCAD(cfg).detect(pre)
    return ED._match_scene(
        [(d, 100.0 * float(d.score)) for d in found], truths, iou, any_class=True
    )


def _tune_classical(n_scenes: int, iou: float, k_grid) -> tuple[float, float]:
    """Pick (k_sigma, score cut) for the shadow-gated baseline on split A."""
    per_k: dict[float, list] = {k: [] for k in k_grid}
    n_truth, area = 0, 0.0
    for scene in ED.iter_scenes(n_scenes, TUNE_SEED_BASE):
        n_truth += len(scene.truths)
        area += scene.area_km2
        for k in k_grid:
            per_k[k].extend(
                _classical_labels(
                    scene.pre, scene.truths,
                    ClassicalConfig(k_sigma=k, require_shadow=True), iou,
                )
            )
    best = None
    for k, labels in per_k.items():
        m, cut = EB._sweep_floor(labels, n_truth, area)
        if best is None or m.f1 > best[0]:
            best = (m.f1, k, cut)
    assert best is not None
    print(f"tuned classical on split A: k_sigma={best[1]:g}, score>={best[2]:.1f}")
    return best[1], best[2]


def run_sweep(
    n_scenes: int = 8,
    levels: tuple[int, ...] = DEFAULT_LEVELS,
    seed_base: int = ED.SEED_BASE,
    iou_thresh: float = 0.30,
    out_path: str | Path = DEFAULT_OUT,
    detector=None,
    *,
    raw_score_floor: float = 0.25,
    k_grid: tuple[float, ...] = K_SIGMA_GRID,
    n_pings_range: tuple[int, int] = (500, 800),
    n_samples: int = 1024,
    shadow_pad_cols: int = 3,
) -> dict:
    """Measure both methods at each clutter level, under both brightness modes."""
    start = time.perf_counter()
    detector = detector if detector is not None else ED.deployed_detector()
    verifier = PhysicsVerifier.load()
    gate = PhysicsGate()
    k_sel, cut_sel = _tune_classical(n_scenes, iou_thresh, k_grid)
    cad_cfg = ClassicalConfig(k_sigma=k_sel, require_shadow=True)

    classical: dict[tuple[str, int], list] = {
        (m, level): [] for m in MODES for level in levels
    }
    sagar: dict[tuple[str, int], list] = {
        (m, level): [] for m in MODES for level in levels
    }
    n_truth, area_km2 = 0, 0.0

    rng = np.random.default_rng(seed_base)
    for i in range(n_scenes):
        cfg = SceneConfig(
            n_pings=int(rng.integers(*n_pings_range)),
            n_samples=int(n_samples),
            slant_range=float(rng.uniform(40.0, 60.0)),
            altitude=float(rng.uniform(6.0, 12.0)),
            seed=seed_base + i,
        )
        base = random_targets(cfg, rng)
        # One pool per (scene, mode), drawn once and sliced so levels nest.
        # Same generator seed per mode: the rocks sit in the same places and
        # differ only in brightness, which is the variable under test.
        pools = {
            mode: _rock_pool(
                cfg, np.random.default_rng(seed_base + 500 + i), base,
                max(levels), mode,
            )
            for mode in MODES
        }

        for mode in MODES:
            for level in levels:
                if level == 0 and mode != MODES[0]:
                    continue  # no rocks: identical to the first mode, filled in below
                pa, targets = make_scene(cfg, base + pools[mode][:level])
                pre = preprocess(pa)
                truths = ED._truth_boxes(pre, targets, cfg, shadow_pad_cols)
                if mode == MODES[0] and level == levels[0]:
                    n_truth += len(truths)
                    area_km2 += float(survey_stats(pre)["area_surveyed_sqkm"])

                classical[(mode, level)].extend(
                    _classical_labels(pre, truths, cad_cfg, iou_thresh)
                )
                dets = [
                    d
                    for d in detector.detect_tiles(pre.tiles)
                    if d.score >= raw_score_floor
                ]
                sagar[(mode, level)].extend(
                    ED._match_scene(
                        [
                            (v.det, v.confidence_pct)
                            for v in verify_detections(
                                dets, pre, gate=gate, verifier=verifier
                            )
                        ],
                        truths, iou_thresh, any_class=True,
                    )
                )
        print(
            f"scene {i + 1}/{n_scenes}: {len(base)} debris targets, "
            f"{len(pools[MODES[0]])} decoy rocks available"
        )

    # Level 0 is rock-free, so it is the same measurement in every mode.
    if 0 in levels:
        for mode in MODES[1:]:
            classical[(mode, 0)] = classical[(MODES[0], 0)]
            sagar[(mode, 0)] = sagar[(MODES[0], 0)]

    results = {
        mode: {
            level: {
                "classical": ED._config_metrics(
                    classical[(mode, level)], n_truth, area_km2, cut_sel
                ),
                "sagar": ED._config_metrics(
                    sagar[(mode, level)], n_truth, area_km2, DEPLOYED_FLOOR_PCT
                ),
            }
            for level in levels
        }
        for mode in MODES
    }
    written = _write(
        Path(out_path), results, levels=levels, n_scenes=n_scenes, n_truth=n_truth,
        area_km2=area_km2, seed_base=seed_base, iou_thresh=iou_thresh,
        k_sel=k_sel, cut_sel=cut_sel, elapsed_s=time.perf_counter() - start,
    )
    print(f"\nwrote {written}\n")
    print(written.read_text(encoding="utf-8"))
    return results


MODE_BLURB = {
    "native": "rocks keep catalogue reflectivity 2.0-3.0 — the simulator's "
              "brightness gap is intact and a threshold can exploit it",
    "matched": "each rock borrows a real target's reflectivity — brightness "
               "carries no information about whether an object is debris",
}


def _write(
    out_path: Path,
    results: dict,
    *,
    levels,
    n_scenes: int,
    n_truth: int,
    area_km2: float,
    seed_base: int,
    iou_thresh: float,
    k_sel: float,
    cut_sel: float,
    elapsed_s: float,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first, last = levels[0], levels[-1]

    lines = [
        "# Clutter sweep — what happens when the seabed stops being clean",
        "",
        f"Generated {datetime.now(tz=UTC).isoformat(timespec='seconds')} by "
        f"`scripts/eval_clutter.py` in {elapsed_s:.0f} s.",
        "",
        "**SYNTHETIC.** Physics-simulated scenes only; no real sonar data. Treat as",
        "a *relative* comparison between methods on identical pixels.",
        "",
        "## Why this table exists",
        "",
        "`docs/baseline_comparison.md` finds a tuned classical CAD baseline slightly",
        "*ahead* of SAGAR-NETRA on the standard held-out scenes. That comparison is",
        "confounded, and this table is the evidence.",
        "",
        "In the scene simulator `rock_cluster` — the only natural clutter class — has",
        "reflectivity **2.0-3.0**, the lowest of any class, while most man-made",
        "targets sit at **4.0-8.0**. Brightness therefore *is* the man-made/natural",
        "label for most of the catalogue, and a detector that thresholds on",
        "brightness is handed the answer by the data generator. Real sonar offers no",
        "such gap: a boulder and a steel drum can return comparable amplitude, which",
        "is precisely why the problem needs shape, shadow geometry and learning.",
        "",
        "So clutter is swept under two conditions, identical in every other respect —",
        "same scenes, same debris, same seeds, rocks in the same positions:",
        "",
        f"- **native** — {MODE_BLURB['native']}.",
        f"- **matched** — {MODE_BLURB['matched']}.",
        "",
        "Levels are nested; every rock is a false positive by construction (truth",
        "boxes are man-made only). Recall should stay near-flat because the debris",
        "field never changes; what moves is precision.",
        "",
        "## Protocol",
        "",
        f"- {n_scenes} scenes, {n_truth} man-made truth boxes, {area_km2:.4f} km², "
        f"seed base {seed_base}.",
        f"- Classical baseline tuned on a separate split (seed base {TUNE_SEED_BASE}): "
        f"`k_sigma={k_sel:g}`, `score>={cut_sel:.1f}`, shadow gate on (its stronger form).",
        f"- SAGAR-NETRA at its shipped {DEPLOYED_FLOOR_PCT:.0f}% floor, tuned against nothing.",
        f"- TP: IoU >= {iou_thresh}, **class match not required** (localization only).",
        "- Decoy rocks are placed clear of man-made targets so no truth box is corrupted.",
        "",
    ]

    for mode in MODES:
        lines += [
            f"## {mode.upper()} — {MODE_BLURB[mode]}",
            "",
            "| extra rocks | classical P | classical R | classical F1 "
            "| SAGAR P | SAGAR R | SAGAR F1 |",
            "|---|---|---|---|---|---|---|",
        ]
        for level in levels:
            c = results[mode][level]["classical"]
            s = results[mode][level]["sagar"]
            lines.append(
                f"| +{level} | {c.precision:.3f} | {c.recall:.3f} | {c.f1:.3f} "
                f"| {s.precision:.3f} | {s.recall:.3f} | {s.f1:.3f} |"
            )
        dc = results[mode][last]["classical"].precision - results[mode][first]["classical"].precision
        ds = results[mode][last]["sagar"].precision - results[mode][first]["sagar"].precision
        lines += [
            "",
            f"Precision change from +{first} to +{last} rocks: "
            f"**classical {dc:+.3f}**, **SAGAR-NETRA {ds:+.3f}**.",
            "",
        ]

    lines += [
        "## Reading it",
        "",
        "- Compare the two `matched` slopes, not the headline numbers. The question",
        "  is which method degrades more slowly when brightness stops being a",
        "  shortcut, because that is the only condition resembling real seabed.",
        "- `rock_cluster` is a trained hard negative for the classifier and a feature",
        "  in the Stage-2 verifier's cue vector; the classical detector has no way to",
        "  express \"bright, shadowed, and not debris\".",
        "- If the two methods degrade alike under `matched`, the honest conclusion is",
        "  that this simulator cannot separate them, and the claim needs real data —",
        "  not a louder version of this table.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")

    out_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "generated": datetime.now(tz=UTC).isoformat(timespec="seconds"),
                "synthetic": True,
                "seed_base": seed_base,
                "n_scenes": n_scenes,
                "n_truth": n_truth,
                "area_km2": area_km2,
                "classical_k_sigma": k_sel,
                "classical_score_cut": cut_sel,
                "modes": {
                    mode: [
                        {
                            "extra_rocks": level,
                            "classical": {
                                "precision": results[mode][level]["classical"].precision,
                                "recall": results[mode][level]["classical"].recall,
                                "f1": results[mode][level]["classical"].f1,
                                "fp_per_km2": results[mode][level]["classical"].fp_per_km2,
                            },
                            "sagar": {
                                "precision": results[mode][level]["sagar"].precision,
                                "recall": results[mode][level]["sagar"].recall,
                                "f1": results[mode][level]["sagar"].f1,
                                "fp_per_km2": results[mode][level]["sagar"].fp_per_km2,
                            },
                        }
                        for level in levels
                    ]
                    for mode in MODES
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=8)
    parser.add_argument("--levels", type=int, nargs="+", default=list(DEFAULT_LEVELS))
    parser.add_argument("--seed-base", type=int, default=ED.SEED_BASE)
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run_sweep(
        n_scenes=args.scenes,
        levels=tuple(args.levels),
        seed_base=args.seed_base,
        iou_thresh=args.iou,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
