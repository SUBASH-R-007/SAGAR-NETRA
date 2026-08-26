"""Generate the bundled sample survey: a deterministic, physics-consistent
synthetic XTF plus its ground-truth JSON and a raw waterfall quick-look.

Usage:
    python scripts/make_sample_xtf.py [--out data/samples] [--pings 1200]

Output is byte-stable for a given seed, so the sample never needs to be
committed to git — regenerating yields the identical survey.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sonar_core.synth.sample import make_sample

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "samples")
    parser.add_argument("--pings", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=26057)
    args = parser.parse_args()

    path = make_sample(args.out, n_pings=args.pings, seed=args.seed)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print("wrote survey_alpha.truth.json and survey_alpha.raw.png alongside")


if __name__ == "__main__":
    main()
