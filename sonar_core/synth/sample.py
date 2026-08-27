"""Bundled-sample builder: deterministic synthetic survey as XTF + truth JSON.

Lives in the package (not scripts/) so tests and the API can regenerate the
sample programmatically; ``scripts/make_sample_xtf.py`` is the CLI wrapper.
"""

from __future__ import annotations

import json
from pathlib import Path

from sonar_core.parsers.xtf_writer import write_xtf
from sonar_core.synth.scene import SceneConfig, make_scene
from sonar_core.waterfall import save_waterfall_png


def make_sample(out_dir: str | Path, n_pings: int = 1200, seed: int = 26057) -> Path:
    """Write survey_alpha.xtf, its ground-truth JSON, and a raw waterfall PNG."""
    out_dir = Path(out_dir)
    cfg = SceneConfig(n_pings=n_pings, seed=seed)
    pa, targets = make_scene(cfg)

    xtf_path = write_xtf(pa, out_dir / "survey_alpha.xtf")
    truth = {
        "scene": {
            "n_pings": cfg.n_pings,
            "n_samples": cfg.n_samples,
            "slant_range_m": cfg.slant_range,
            "mean_altitude_m": cfg.altitude,
            "seed": cfg.seed,
        },
        "targets": [t.to_dict() for t in targets],
    }
    (out_dir / "survey_alpha.truth.json").write_text(json.dumps(truth, indent=2))
    save_waterfall_png(pa, out_dir / "survey_alpha.raw.png")
    return xtf_path
