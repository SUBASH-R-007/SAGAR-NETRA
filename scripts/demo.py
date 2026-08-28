"""SAGAR-NETRA demo rehearsal: the full story on one command.

    python scripts/demo.py [--input data/samples/survey_alpha.xtf] [--serve]

Narrates the complete flow — parse, preprocess, TridentNet, PhysiCheck,
GeoScribe — on the bundled sample survey, prints the contact table and where
every report landed, and (with ``--serve``) starts the DRISHTI console with
the results loaded.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ``python scripts/demo.py`` puts scripts/ on sys.path -- not the repo root --
# so the package imports below only resolve if SAGAR-NETRA happens to be
# pip-installed. Put the checkout first so it always wins over an installed copy.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Third-party packages the demo cannot run without, checked up front so the
#: usual mistake -- a shell where the virtualenv was never activated -- reports
#: itself in one readable line instead of as a traceback from deep inside the
#: pipeline, halfway through a presentation.
REQUIRED = ("pyxtf", "torch", "ultralytics", "cv2", "sklearn", "reportlab", "simplekml")

#: Where ``python -m venv .venv`` puts the interpreter on this platform.
VENV_RELPATH = Path(".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")


def check_environment() -> None:
    """Abort with the exact command to run when the interpreter is the wrong one."""
    missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
    if not missing:
        return

    print()
    print(f"Cannot run: missing {', '.join(missing)}")
    print(f"Interpreter: {sys.executable}")
    print()

    if (REPO_ROOT / VENV_RELPATH).exists():
        argv = " ".join(sys.argv[1:])
        print("That is not the project virtualenv. From the repo root, run:")
        print()
        print(f"  {VENV_RELPATH} scripts/demo.py {argv}".rstrip())
    else:
        print("The virtualenv does not exist yet. From the repo root, run:")
        print()
        print("  python -m venv .venv")
        print(f"  {VENV_RELPATH.parent / 'pip'} install -e .[ml,api,dev]")
    raise SystemExit(1)


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

    check_environment()

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
