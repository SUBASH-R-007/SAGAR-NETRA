"""Best-effort downloader for public sonar datasets (OPTIONAL extras).

SAGAR-NETRA trains entirely from the synthetic scene simulator
(``tridentnet.data.build_synthetic_dataset``); nothing in the pipeline requires
any of the datasets below. They are optional add-ons for domain realism,
fine-tuning and honest evaluation on real acoustics. Licenses differ — some are
non-commercial — so ``--list`` prints license and attribution for each entry
and the decision to mix a dataset into training stays with the operator.

Usage:
    python scripts/download_datasets.py --list
    python scripts/download_datasets.py --get uatd
    python scripts/download_datasets.py --all

Every failure mode (offline, moved URL, registration-gated dataset) warns and
continues; this script never crashes a build.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "datasets"

USER_AGENT = "sagar-netra-dataset-fetch/0.1"
DEFAULT_TIMEOUT_S = 60.0
DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB read granularity for progress updates
BYTES_PER_MB = 1e6


@dataclass(frozen=True)
class DatasetSpec:
    """One public dataset: where it lives, what it may be used for, and how."""

    name: str
    url: str
    license: str
    expected_use: str
    direct: bool = True  # False: landing page — registration/manual download
    note: str = ""

    @property
    def target_dir(self) -> Path:
        return DATA_ROOT / self.name


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="marine-debris-fls",
        url="https://zenodo.org/api/records/15101686/files-archive",
        license="CC BY-NC-SA 4.0",
        expected_use=(
            "forward-looking sonar marine-debris images; classifier pretraining "
            "and qualitative eval (non-commercial license - never ship in a paid build)"
        ),
        note="DOI 10.5281/zenodo.15101686",
    ),
    DatasetSpec(
        name="uatd",
        url="https://ndownloader.figshare.com/articles/21331143/versions/3",
        license="CC BY 4.0",
        expected_use=(
            "underwater acoustic target detection benchmark (cube, cylinder, tyre "
            "and other shapes); detector pretraining and shape-class realism"
        ),
        note="DOI 10.6084/m9.figshare.21331143.v3",
    ),
    DatasetSpec(
        name="sctd",
        url="https://github.com/MingqiangNing/SCTD/archive/refs/heads/main.zip",
        license="unspecified (academic use; cite the SCTD paper)",
        expected_use=(
            "sonar common target detection: aircraft, ship, human classes for "
            "fine-tuning wreck/aircraft/human_body heads"
        ),
    ),
    DatasetSpec(
        name="klsg",
        url=(
            "https://github.com/huoguanying/"
            "SeabedObjects-Ship-and-Airplane-dataset/archive/refs/heads/master.zip"
        ),
        license="released for research by the authors (see repo)",
        expected_use="KLSG side-scan wreck and aircraft chips; wreck/aircraft class realism",
    ),
    DatasetSpec(
        name="ai4shipwrecks",
        url="https://umfieldrobotics.github.io/ai4shipwrecks/",
        license="CC BY-NC-SA 4.0 (see site)",
        expected_use=(
            "Thunder Bay side-scan shipwreck segmentation benchmark; Brain B "
            "(segmenter) evaluation on real surveys"
        ),
        direct=False,
        note=(
            "download is registration-gated: request access on the site, then "
            "extract into data/datasets/ai4shipwrecks"
        ),
    ),
    DatasetSpec(
        name="swdd",
        url="https://zenodo.org/api/records/13692547/files-archive",
        license="see Zenodo record page",
        expected_use="side-scan wreck detection dataset; wreck detector evaluation",
        note="DOI 10.5281/zenodo.13692547",
    ),
    DatasetSpec(
        name="marine-pulse",
        url="https://zenodo.org/api/records/7922705/files-archive",
        license="CC BY 4.0",
        expected_use=(
            "Marine PULSE: pipelines, cables and other seabed infrastructure "
            "imagery; pipeline class realism"
        ),
        note="DOI 10.5281/zenodo.7922705",
    ),
    DatasetSpec(
        name="uxo",
        url="https://zenodo.org/api/records/13778485/files-archive",
        license="BSD-3-Clause",
        expected_use="unexploded-ordnance sonar views; mine_like class realism",
        note="DOI 10.5281/zenodo.13778485",
    ),
)


def list_datasets() -> None:
    """Print the dataset table with licenses and attribution notes."""
    width = max(len(s.name) for s in DATASETS)
    print(f"{'name':<{width}}  {'license':<45}  url")
    print("-" * (width + 2 + 45 + 2 + 40))
    for s in DATASETS:
        print(f"{s.name:<{width}}  {s.license:<45}  {s.url}")
        print(f"{'':<{width}}  use: {s.expected_use}")
        if s.note:
            print(f"{'':<{width}}  note: {s.note}")
        if not s.direct:
            print(f"{'':<{width}}  (manual download; --get prints instructions)")
    print()
    print("All datasets are OPTIONAL: the full pipeline trains from the synthetic")
    print("simulator alone. Respect each license; cite the original authors in any")
    print("publication or demo that uses their data.")


def _download(url: str, dest: Path, timeout_s: float) -> None:
    """Stream *url* to *dest* with a coarse progress line."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp, open(dest, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = resp.read(DOWNLOAD_CHUNK_BYTES)
            if not block:
                break
            fh.write(block)
            done += len(block)
            if total:
                pct = 100.0 * done / total
                print(
                    f"\r  {dest.name}: {done / BYTES_PER_MB:8.1f} MB"
                    f" / {total / BYTES_PER_MB:.1f} MB ({pct:5.1f}%)",
                    end="",
                    flush=True,
                )
            else:
                print(f"\r  {dest.name}: {done / BYTES_PER_MB:8.1f} MB", end="", flush=True)
        print()


def fetch(spec: DatasetSpec, timeout_s: float = DEFAULT_TIMEOUT_S) -> bool:
    """Download and unpack one dataset; warn and return False on any failure."""
    if not spec.direct:
        print(f"[{spec.name}] manual download required: {spec.url}")
        if spec.note:
            print(f"  note: {spec.note}")
        return False

    spec.target_dir.mkdir(parents=True, exist_ok=True)
    fname = Path(urllib.parse.urlparse(spec.url).path).name or f"{spec.name}.bin"
    dest = spec.target_dir / fname
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[{spec.name}] archive already present: {dest}")
    else:
        # Download to a temp name and rename on success so an interrupted run
        # (Ctrl+C included) can never leave a truncated file that a later run
        # would mistake for a complete archive.
        part = dest.with_suffix(dest.suffix + ".part")
        print(f"[{spec.name}] downloading {spec.url}")
        try:
            _download(spec.url, part, timeout_s)
            part.replace(dest)
        except BaseException as exc:
            part.unlink(missing_ok=True)
            if not isinstance(exc, (urllib.error.URLError, OSError, ValueError)):
                raise
            print(f"[{spec.name}] WARNING: download failed ({exc}); skipping")
            return False

    if dest.suffix.lower() == ".zip" and not zipfile.is_zipfile(dest):
        print(f"[{spec.name}] WARNING: {dest.name} is not a valid zip (corrupt "
              f"download?) — delete it and rerun")
        return False
    if zipfile.is_zipfile(dest):
        print(f"[{spec.name}] extracting {dest.name}")
        try:
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(spec.target_dir)
        except (zipfile.BadZipFile, OSError) as exc:
            print(f"[{spec.name}] WARNING: extraction failed ({exc})")
            return False
    print(f"[{spec.name}] ready under {spec.target_dir}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print the dataset table")
    parser.add_argument(
        "--get", action="append", metavar="NAME", default=[],
        help="download one dataset by name (repeatable)",
    )
    parser.add_argument("--all", action="store_true", help="download every direct dataset")
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="per-request timeout, seconds"
    )
    args = parser.parse_args(argv)

    if args.list:
        list_datasets()
    by_name = {s.name: s for s in DATASETS}
    wanted: list[DatasetSpec] = []
    for name in args.get:
        if name not in by_name:
            print(f"WARNING: unknown dataset {name!r}; known: {', '.join(sorted(by_name))}")
            continue
        wanted.append(by_name[name])
    if args.all:
        wanted.extend(s for s in DATASETS if s not in wanted)
    for spec in wanted:
        fetch(spec, timeout_s=args.timeout)
    if not (args.list or args.get or args.all):
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
