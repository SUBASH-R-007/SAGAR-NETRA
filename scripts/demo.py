"""SAGAR-NETRA demo rehearsal: the full story on one command.

    python scripts/demo.py [--input data/samples/survey_alpha.xtf] [--serve]

Narrates the complete flow — parse, preprocess, TridentNet, PhysiCheck,
GeoScribe — on the bundled sample survey, prints the contact table and where
every report landed, and (with ``--serve``) starts the DRISHTI console with
the results loaded.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def narrate(stage: str = "", fraction: float = 0.0, message: str = "", **_: object) -> None:
    bar = "#" * int(fraction * 30)
    print(f"\r[{bar:<30}] {fraction * 100:5.1f}%  {message or stage:<40}", end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=REPO_ROOT / "data" / "samples" / "survey_alpha.xtf"
    )
    parser.add_argument("--serve", action="store_true", help="launch the console after")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print("=" * 72)
    print("SAGAR-NETRA — AI-powered marine debris detection from side-scan sonar")
    print("=" * 72)

    if not args.input.exists():
        print(f"\nSample survey missing — generating {args.input} ...")
        from sonar_core.synth.sample import make_sample

        make_sample(args.input.parent)

    from api.db import ContactRepo
    from api.processing import process_survey

    repo = ContactRepo(REPO_ROOT / "data" / "contacts.db")
    print(f"\nProcessing {args.input.name} ...")
    start = time.perf_counter()
    summary = process_survey(args.input, repo, update=narrate)
    elapsed = time.perf_counter() - start
    print()

    print(f"\nDone in {elapsed:.1f} s: {summary['n_pings']} pings -> "
          f"{summary['n_tiles']} tiles -> {summary['n_detections']} raw detections -> "
          f"{summary['n_contacts']} verified contacts\n")

    contacts = repo.query(survey=summary["survey"], limit=100)
    if contacts:
        header = f"{'ID':<20} {'class':<14} {'conf%':>6} {'sev':>5} {'H(m)':>6}  position"
        print(header)
        print("-" * len(header))
        for c in contacts:
            height = "-" if c.dims.height_m is None else f"{c.dims.height_m:.1f}"
            flag = " [PHYSICS!]" if c.physics.physics_violation else ""
            print(f"{c.id:<20} {c.cls:<14} {c.confidence:>6.1f} {c.severity:>5.1f} "
                  f"{height:>6}  {c.lat:.5f}, {c.lon:.5f}{flag}")
    else:
        print("No contacts survived verification — with untrained smoke weights this "
              "can happen; train with scripts/train_detector.py first.")

    print("\nReports:")
    for fmt, path in summary["reports"].items():
        print(f"  {fmt:<8} {path}")
    print(f"  imagery  {summary['outputs_dir']}\\waterfall.png (+ evidence cards)")

    if args.serve:
        print(f"\nStarting DRISHTI console at http://127.0.0.1:{args.port} — Ctrl+C to stop")
        import uvicorn

        uvicorn.run("api.main:app", host="127.0.0.1", port=args.port)
    else:
        print("\nNext: python scripts/demo.py --serve   (opens the dashboard with these results)")


if __name__ == "__main__":
    main()
