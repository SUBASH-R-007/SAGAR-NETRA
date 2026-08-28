"""Tests for the clutter sweep's decoy placement.

The sweep's entire claim rests on one control: the ``native`` and ``matched``
conditions must differ in decoy *brightness* and in nothing else. If the two
modes place rocks differently, the comparison silently becomes two unrelated
experiments and the conclusion drawn from it is worthless. These tests pin that
control, and pin the reflectivity semantics that make ``matched`` meaningful.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from sonar_core.synth.scene import SceneConfig, SynthTarget
from tridentnet.data import CLASS_SPECS

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EC = _load("eval_clutter")

CFG = SceneConfig(n_pings=800, n_samples=1024, slant_range=50.0, altitude=8.0, seed=1)

#: A debris field with deliberately varied, all-bright reflectivities, so
#: "borrowed a real target's brightness" is distinguishable from "kept its own".
DEBRIS = [
    SynthTarget("cylinder_drum", "port", 200, 15.0, 2.0, 1.5, 0.9, 6.2, False, "ellipse"),
    SynthTarget("container", "starboard", 400, 22.0, 6.0, 2.4, 2.4, 5.4, False, "rect"),
    SynthTarget("mine_like", "port", 600, 30.0, 1.2, 1.0, 0.6, 7.1, False, "ellipse"),
]
DONORS = {t.reflectivity for t in DEBRIS}


def _pool(mode: str, n: int = 8, seed: int = 4):
    return EC._rock_pool(CFG, np.random.default_rng(seed), DEBRIS, n, mode)


def _geometry(rocks):
    """Everything about a rock except how bright it is."""
    return [
        (r.side, r.ping, r.ground_range, r.length, r.width, r.height, r.shape)
        for r in rocks
    ]


def test_modes_place_identical_rocks() -> None:
    """The control: same seed, same positions, brightness the only difference.

    Branching on an RNG draw would desynchronise the two streams and move every
    subsequent rock — a failure that produces perfectly plausible tables.
    """
    native, matched = _pool("native"), _pool("matched")
    assert len(native) == len(matched) > 0
    assert _geometry(native) == _geometry(matched)
    assert [r.reflectivity for r in native] != [r.reflectivity for r in matched]


def test_native_keeps_catalogue_reflectivity() -> None:
    lo, hi = CLASS_SPECS["rock_cluster"].reflectivity
    assert all(lo <= r.reflectivity <= hi for r in _pool("native"))


def test_matched_borrows_a_real_target_brightness() -> None:
    """Under `matched`, brightness must carry no man-made/natural information."""
    assert all(r.reflectivity in DONORS for r in _pool("matched"))


def test_matched_is_brighter_than_native() -> None:
    """Sanity: the simulator's gap is real, so matching it must raise decoys."""
    native = float(np.mean([r.reflectivity for r in _pool("native")]))
    matched = float(np.mean([r.reflectivity for r in _pool("matched")]))
    assert matched > native


def test_rocks_are_natural_and_named() -> None:
    for rock in _pool("matched"):
        assert rock.natural is True
        assert rock.cls == "rock_cluster"


def test_rocks_never_touch_a_real_target() -> None:
    """A rock merged into a drum would corrupt the truth box for both methods."""
    for rock in _pool("matched", n=24):
        assert EC._clear_of(rock, DEBRIS, CFG, 3.0)


def test_clear_of_rejects_an_overlap() -> None:
    intruder = SynthTarget(
        "rock_cluster", "port", 200, 15.0, 2.0, 1.5, 0.9, 2.5, True, "irregular"
    )
    assert not EC._clear_of(intruder, DEBRIS, CFG, 3.0)

    # Same position, other side: no overlap, because the swaths are disjoint.
    other_side = SynthTarget(
        "rock_cluster", "starboard", 200, 15.0, 2.0, 1.5, 0.9, 2.5, True, "irregular"
    )
    assert EC._clear_of(other_side, [DEBRIS[0]], CFG, 3.0)


def test_pool_is_nested_by_construction() -> None:
    """Levels slice one pool, so level N must contain every rock of level N-1."""
    pool = _pool("matched", n=12)
    assert _geometry(pool[:6]) == _geometry(pool)[:6]
