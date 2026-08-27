"""Shared fixtures: a small deterministic scene, and the bundled sample XTF
(generated once per test session — byte-stable for a given seed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sonar_core.parsers.base import PingArray
from sonar_core.synth.scene import SceneConfig, SynthTarget, make_scene


@pytest.fixture(scope="session")
def small_scene() -> tuple[PingArray, list[SynthTarget]]:
    cfg = SceneConfig(n_pings=200, n_samples=256, slant_range=40.0, seed=7)
    targets = [
        SynthTarget("cylinder_drum", "starboard", 100, 22.0, 1.4, 0.9, 0.9, reflectivity=6.0),
        SynthTarget("tire", "port", 60, 14.0, 1.1, 1.1, 0.35, reflectivity=3.5),
    ]
    return make_scene(cfg, targets)


@pytest.fixture(scope="session")
def sample_xtf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from sonar_core.synth.sample import make_sample

    out = tmp_path_factory.mktemp("samples")
    return make_sample(out, n_pings=600)
