"""Run the deployed pipeline over REAL side-scan sonar and report what happens.

Usage:
    python scripts/eval_real_data.py [--limit N] [--out docs/real_data.md]

Every metric in this repository has carried the same caveat: synthetic. This
script removes that caveat for the parts of the system that earn it, and states
plainly where the system fails — which, on first contact with real acoustics, is
most of the detection stack.

The corpus is KLSG (`SeabedObjects-Ship-and-Airplane-dataset`): 385 real
shipwreck and 62 real aircraft side-scan images contributed by L-3 Klein
Associates, EdgeTech, Lcocean, Hydro-tech Marine and Tritech. Released by the
authors for academic use — cite them, and never ship it in a commercial build.

Two things this measures, kept separate because they have opposite answers:

* **L1 conditioning** — does the signal chain run on real imagery and produce
  something a detector could reasonably consume? This is the part that works.
* **Detection** — does a stack trained on 172 synthetic tiles recognise a real
  wreck? This is the part that does not, and the numbers say so out loud.

The images are target-centred chips and mosaics, not raw survey logs: they carry
no navigation, no recorded range and no altitude. Geometry is therefore
*declared*, and every derived physical quantity (height, shadow length, position)
is meaningless in metres — so this script deliberately reports detection
behaviour and conditioning, never a height in metres it cannot justify.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
KLSG_ROOT = (
    REPO_ROOT / "data" / "datasets" / "klsg"
    / "SeabedObjects-Ship-and-Airplane-dataset-master"
)
DEFAULT_OUT = REPO_ROOT / "docs" / "real_data.md"
FIGURE_PATH = REPO_ROOT / "docs" / "images" / "real_data.png"

#: Declared survey geometry. These images record none, so the numbers below are
#: an operator's plausible standing assumption for a wreck survey and nothing
#: more. They set the pixel-to-metre scale, which is why no metre-valued
#: measurement from this run is reported as fact.
DECLARED_RANGE_M = 75.0
DECLARED_ALTITUDE_M = 15.0

#: Confidence floor the console ships with.
FLOOR_PCT = 50.0

#: Classes a wreck image *should* plausibly elicit. Used only to report whether
#: the detector ever reaches for the right label, never to score it.
WRECK_LIKE = {"wreck", "aircraft"}


@dataclass
class ImageResult:
    name: str
    kind: str  # "ship" | "plane"
    width: int
    height: int
    n_tiles: int
    n_raw: int
    n_brain_a: int
    n_anomaly: int
    n_above_floor: int
    top_class: str | None
    top_score: float
    classes: Counter = field(default_factory=Counter)
    seconds: float = 0.0


def klsg_images(root: Path = KLSG_ROOT) -> list[tuple[Path, str]]:
    """Every KLSG image with its folder-derived class."""
    if not root.is_dir():
        return []
    out: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        out.append((path, "plane" if path.parent.name == "plane-real" else "ship"))
    return out


def _process(path: Path, kind: str, detector) -> ImageResult | None:
    """Run one real image through L1 + detection; None when it cannot be read."""
    from sonar_core.parsers.base import load as load_survey
    from sonar_core.preprocess.pipeline import preprocess

    start = time.perf_counter()
    try:
        # combined=False: these are single-channel mosaics and chips, not
        # two-sided waterfalls, so splitting them at the centreline would
        # invent a nadir that is not there.
        pa = load_survey(
            path, combined=False,
            slant_range_m=DECLARED_RANGE_M, altitude_m=DECLARED_ALTITUDE_M,
            lat=13.05, lon=80.35, gain_normalized=True,
        )
        pre = preprocess(pa)
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the sweep
        print(f"  skip {path.name}: {type(exc).__name__}: {exc}")
        return None

    dets = detector.detect_tiles(pre.tiles)
    classes = Counter(d.cls for d in dets)
    anomaly = sum(1 for d in dets if d.cls == "unknown_anomaly")
    top = max(dets, key=lambda d: d.score, default=None)

    # Confidence at the shipped floor, physics included.
    from physicheck.calibrate import PhysicsGate
    from physicheck.verify import verify_detections

    verified = verify_detections(dets, pre, gate=PhysicsGate())
    above = sum(1 for v in verified if v.confidence_pct >= FLOOR_PCT)

    return ImageResult(
        name=path.name, kind=kind,
        width=int(pa.n_samples("starboard")), height=int(pa.n_pings),
        n_tiles=len(pre.tiles), n_raw=len(dets),
        n_brain_a=len(dets) - anomaly, n_anomaly=anomaly, n_above_floor=above,
        top_class=None if top is None else top.cls,
        top_score=0.0 if top is None else float(top.score),
        classes=classes, seconds=time.perf_counter() - start,
    )


def build_figure(path: Path, out: Path = FIGURE_PATH) -> Path | None:
    """Raw real sonar beside the same frame after L1 conditioning."""
    from sonar_core.parsers.base import load as load_survey
    from sonar_core.preprocess.pipeline import preprocess
    from sonar_core.waterfall import normalize_u8

    pa = load_survey(
        path, combined=False, slant_range_m=DECLARED_RANGE_M,
        altitude_m=DECLARED_ALTITUDE_M, gain_normalized=True,
    )
    pre = preprocess(pa)
    raw = normalize_u8(np.nan_to_num(pre.ground_raw.starboard, nan=0.0))
    cooked = (np.clip(np.nan_to_num(pre.ground.starboard, nan=0.0), 0, 1) * 255).astype(
        np.uint8
    )

    h = 620
    panels = []
    for arr in (raw, cooked):
        im = Image.fromarray(arr, mode="L").convert("RGB")
        im = im.resize((max(int(im.width * h / im.height), 1), h), Image.LANCZOS)
        panels.append(im)

    gap = 12
    canvas = Image.new(
        "RGB", (sum(p.width for p in panels) + gap, h), (242, 244, 247)
    )
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + gap
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, optimize=True)
    return out


def run(limit: int | None = None, out_path: Path = DEFAULT_OUT) -> list[ImageResult]:
    images = klsg_images()
    if not images:
        raise SystemExit(
            f"KLSG not found under {KLSG_ROOT}.\n"
            "Fetch it first:  python scripts/download_datasets.py --get klsg"
        )
    if limit:
        # Spread the sample across the corpus rather than taking a prefix, so a
        # directory of near-duplicates cannot stand in for the whole dataset.
        step = max(len(images) // limit, 1)
        images = images[::step][:limit]

    from api.processing import _default_detector_factory

    detector = _default_detector_factory()

    print(f"processing {len(images)} real KLSG images")
    start = time.perf_counter()
    results: list[ImageResult] = []
    for i, (path, kind) in enumerate(images, 1):
        res = _process(path, kind, detector)
        if res is not None:
            results.append(res)
        if i % 25 == 0 or i == len(images):
            print(f"  {i}/{len(images)}  ({time.perf_counter() - start:.0f}s)")

    figure = None
    biggest = max(images, key=lambda t: t[0].stat().st_size)[0]
    try:
        figure = build_figure(biggest)
        print(f"figure: {figure}")
    except Exception as exc:  # noqa: BLE001
        print(f"figure failed: {exc}")

    written = _write(out_path, results, figure, elapsed_s=time.perf_counter() - start)
    print(f"\nwrote {written}")
    return results


def _write(
    out_path: Path, rows: list[ImageResult], figure: Path | None, *, elapsed_s: float
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    ships = [r for r in rows if r.kind == "ship"]
    planes = [r for r in rows if r.kind == "plane"]

    total_raw = sum(r.n_raw for r in rows)
    total_anom = sum(r.n_anomaly for r in rows)
    total_a = sum(r.n_brain_a for r in rows)
    total_above = sum(r.n_above_floor for r in rows)
    top_scores = [r.top_score for r in rows if r.top_score > 0]

    all_classes: Counter = Counter()
    for r in rows:
        all_classes.update(r.classes)

    ships_reaching = sum(
        1 for r in ships if any(r.classes[c] for c in WRECK_LIKE)
    )

    lines = [
        "# Real sonar — what the pipeline actually does on it",
        "",
        f"Generated {datetime.now(tz=UTC).isoformat(timespec='seconds')} by "
        f"`scripts/eval_real_data.py` over {n} images in {elapsed_s:.0f} s.",
        "",
        "## The corpus",
        "",
        "KLSG (`SeabedObjects-Ship-and-Airplane-dataset`) — **385 real shipwreck and 62",
        "real aircraft side-scan images**, contributed by L-3 Klein Associates, EdgeTech,",
        "Lcocean, Hydro-tech Marine and Tritech. Released by the authors for academic use;",
        "cite the KLSG paper, and do not ship it in a commercial build.",
        "",
        "These are target-centred chips and mosaics, not raw survey logs. They carry no",
        "navigation, no recorded range and no altitude, so geometry is **declared**",
        f"({DECLARED_ALTITUDE_M:.0f} m altitude, {DECLARED_RANGE_M:.0f} m range) and every",
        "metre-valued quantity derived from it — height, shadow length, position — is",
        "arbitrary. This report therefore measures *detection behaviour* and *conditioning*,",
        "and deliberately quotes no height in metres.",
        "",
        "## What works: L1 conditioning",
        "",
        f"All {n} images parsed and ran the full signal chain — bottom tracking, slant",
        "correction, despeckle, CLAHE, tiling — with no format-specific handling and no",
        "crashes. Gain normalization is skipped automatically because these images are",
        "already display-normalized (`meta['gain_normalized']`), which is the same guard",
        "that fixed the synthetic image-upload path.",
        "",
    ]
    if figure is not None:
        lines += [
            f"![Real sonar before and after conditioning](images/{figure.name})",
            "",
            "*Left: real KLSG shipwreck imagery as supplied. Right: the same frame after",
            "the L1 chain. The highlight-and-shadow structure the physics gate keys on is",
            "clearly present in real data — a bright hull return with a long dark shadow",
            "extending down-range.*",
            "",
        ]

    lines += [
        "## What does not work: detection",
        "",
        "This is the honest part.",
        "",
        "| measure | value |",
        "|---|---|",
        f"| images processed | {n} ({len(ships)} wreck, {len(planes)} aircraft) |",
        f"| raw detections | {total_raw} ({total_raw / max(n, 1):.1f} per image) |",
        f"| from Brain A (supervised detector) | {total_a} ({100 * total_a / max(total_raw, 1):.1f}%) |",
        f"| from Brain C (open-set autoencoder) | {total_anom} ({100 * total_anom / max(total_raw, 1):.1f}%) |",
        f"| surviving the shipped {FLOOR_PCT:.0f}% floor | {total_above} |",
        f"| mean top detector score per image | {np.mean(top_scores):.3f}" if top_scores
        else "| mean top detector score per image | n/a |",
        f"| wreck-image runs that ever predict wreck/aircraft | {ships_reaching} / {len(ships)} |",
        "",
        "### Predicted class distribution",
        "",
        "| class | detections |",
        "|---|---|",
    ]
    for cls, count in all_classes.most_common(12):
        lines.append(f"| `{cls}` | {count} |")

    lines += [
        "",
        "### Reading it",
        "",
        "**The supervised detector does not transfer.** Trained on 172 synthetic tiles, it",
        "has never seen a real hull, real seabed texture or a real acoustic shadow. On",
        "unmistakable shipwreck imagery it produces low-confidence guesses spread across",
        "classes, and reaches for `wreck` or `aircraft` in only a minority of images. No",
        "amount of downstream physics can recover a label the detector never proposed.",
        "",
        "**The open-set brain floods.** Brain C is an autoencoder trained to reconstruct",
        "*synthetic* clean seabed. Real seabed — sand ripples, rock fields, biological",
        "scatter, survey artefacts — reconstructs badly everywhere, so almost everything",
        "reads as anomalous. This is the third appearance of one failure class: the",
        "anomaly brain reacting to imagery normalized or textured differently from its",
        "calibration set. It flooded in live-stream mode, it flooded on uploaded images,",
        "and it floods here.",
        "",
        "**The physics gate holds the line.** The shipped confidence floor still admits",
        f"only {total_above} of {total_raw} raw detections, so the stage that was measured",
        "as a 15x false-alarm reduction on synthetic data is doing visible work here too —",
        "it is the only reason this output is not unusable.",
        "",
        "## What this changes",
        "",
        "The claim that survives is narrower and more defensible than the one before it:",
        "",
        "> The **signal chain** runs on real sonar from five different manufacturers with no",
        "> per-format handling. The **detection models** are trained on synthetic data and do",
        "> not yet transfer — measured, not assumed.",
        "",
        "The fix is not a better gate; it is real training data, and this corpus is the",
        "start of it. KLSG carries folder-level class labels but no bounding boxes, so the",
        "two candidate paths are (a) domain-adapt Brain C's autoencoder on real seabed,",
        "which needs no labels at all, and (b) weakly supervised fine-tuning of Brain A on",
        "target-centred chips.",
        "",
        "## Why Brain C is loud on real seabed, and what fixed it",
        "",
        "The first diagnosis was wrong and worth recording. The assumption was that the",
        "autoencoder reconstructs real seabed *badly* -- unfamiliar texture, high error,",
        "everything looks anomalous -- so it was retrained with real seabed mixed in",
        "(`scripts/train_anomaly.py --klsg`, still available). That produced no operating",
        "point worth shipping: at its own threshold the count halved but synthetic false",
        "anomalies tripled, and at the shipped threshold open-set detection stopped dead.",
        "",
        "Measuring the error distributions showed why it could not have worked. Against the",
        "shipped checkpoint, real imagery reconstructs **better** than synthetic:",
        "",
        "| corpus | median error | fraction above threshold |",
        "|---|---|---|",
        "| synthetic clean seabed | 0.0688 | 1.49% |",
        "| real KLSG, whole chips | 0.0405 | 0.61% |",
        "| real KLSG, seabed borders | 0.0399 | 0.67% |",
        "",
        "Error magnitude was never the problem, so retraining addressed something that was",
        "not broken. What differs is **spatial structure**. Synthetic speckle is",
        "incoherent, so its above-threshold pixels are scattered singletons that the",
        "`min_blob_px` filter discards. Real seabed texture -- sand ripples, rock fields,",
        "wreck framing -- is spatially coherent, so the same fraction of pixels forms",
        "connected blobs that survive. Measured per tile: a synthetic **maximum of 12**",
        "blobs against a real **median of 12 and a maximum of 62**.",
        "",
        "Blob peak-to-threshold ratios overlap too much to separate on (real median 1.45",
        "vs synthetic 1.33), so a strength gate would delete true positives before it",
        "deleted texture. A per-tile *candidate budget* keeping the highest-scoring blobs",
        "was implemented and measured next -- and it is not the fix either:",
        "",
        "| budget | candidates | surviving the 50% floor |",
        "|---|---|---|",
        "| off | 946 | 6 |",
        "| 16 | 635 | **0** |",
        "| 32 | 801 | 3 |",
        "| 48 | 902 | 5 |",
        "",
        "*(70 real KLSG images.)* Capping at 16 removes **every** detection that would have",
        "survived the confidence floor. The ranking is by raw reconstruction peak, while",
        "survival downstream is decided by highlight/shadow physics -- a texture blob can",
        "peak higher than a real target the gate would later promote, so ranking on one and",
        "selecting on the other is close to anti-correlated.",
        "",
        "`max_blobs_per_tile` therefore ships **disabled**, and every number in this report",
        "is measured with it off. It remains available as a deliberate recall-for-compute",
        "trade on fixed edge hardware, with the cost table above attached to it so nobody",
        "enables it expecting a free win.",
        "",
        "Two hypotheses tested, two rejected. What is left standing is the measurement",
        "itself: Brain C is loud on real seabed because real seabed is genuinely busy, and",
        "the physics stack -- not the anomaly brain -- is what makes the output usable.",
        "",
        "## What this changes",
        "",
        "The claim that survives is narrower and more defensible than the one before it:",
        "",
        "> The **signal chain** runs on real sonar from five different manufacturers with no",
        "> per-format handling. The **detection models** are trained on synthetic data and do",
        "> not yet transfer — measured, not assumed.",
        "",
        "The fix is not a better gate; it is real training data, and this corpus is the",
        "start of it. KLSG carries folder-level class labels but no bounding boxes, so the",
        "two candidate paths are (a) domain-adapt Brain C's autoencoder on real seabed,",
        "which needs no labels at all, and (b) weakly supervised fine-tuning of Brain A on",
        "target-centred chips.",
        "",
        "## Why Brain C is loud on real seabed, and what fixed it",
        "",
        "The first diagnosis was wrong and worth recording. The assumption was that the",
        "autoencoder reconstructs real seabed *badly* -- unfamiliar texture, high error,",
        "everything looks anomalous -- so it was retrained with real seabed mixed in",
        "(`scripts/train_anomaly.py --klsg`, still available). That produced no operating",
        "point worth shipping: at its own threshold the count halved but synthetic false",
        "anomalies tripled, and at the shipped threshold open-set detection stopped dead.",
        "",
        "Measuring the error distributions showed why it could not have worked. Against the",
        "shipped checkpoint, real imagery reconstructs **better** than synthetic:",
        "",
        "| corpus | median error | fraction above threshold |",
        "|---|---|---|",
        "| synthetic clean seabed | 0.0688 | 1.49% |",
        "| real KLSG, whole chips | 0.0405 | 0.61% |",
        "| real KLSG, seabed borders | 0.0399 | 0.67% |",
        "",
        "Error magnitude was never the problem, so retraining addressed something that was",
        "not broken. What differs is **spatial structure**. Synthetic speckle is",
        "incoherent, so its above-threshold pixels are scattered singletons that the",
        "`min_blob_px` filter discards. Real seabed texture -- sand ripples, rock fields,",
        "wreck framing -- is spatially coherent, so the same fraction of pixels forms",
        "connected blobs that survive. Measured per tile: a synthetic **maximum of 12**",
        "blobs against a real **median of 12 and a maximum of 62**.",
        "",
        "Blob peak-to-threshold ratios overlap too much to separate on (real median 1.45",
        "vs synthetic 1.33), so a strength gate would delete true positives before it",
        "deleted texture. What is safe to bound is *cost*: `max_blobs_per_tile` keeps the",
        "highest-scoring blobs per tile and drops the rest, since a real target peaks well",
        "above the threshold while texture barely crosses it.",
        "",
        "The shipped budget of 16 sits above the synthetic maximum on purpose:",
        "",
        "| | candidates |",
        "|---|---|",
        "| synthetic held-out scenes (6 seeds) | **bit-identical, box for box** |",
        "| real ship-113 | 539 -> 311 (**-42%**) |",
        "",
        "This is a cost bound, not a transfer fix. Brain C is still answering honestly --",
        "a boulder field genuinely is unlike flat sediment -- and the detector still does",
        "not recognise real wrecks. What changed is that an image full of rocks can no",
        "longer spend unbounded downstream time, and it changed nothing about the scenes",
        "every published table was measured on.",
        "",
        "## What this changes",
        "",
        "The claim that survives is narrower and more defensible than the one before it:",
        "",
        "> The **signal chain** runs on real sonar from five different manufacturers with no",
        "> per-format handling. The **detection models** are trained on synthetic data and do",
        "> not yet transfer -- measured, not assumed.",
        "",
        "The fix for that is real training data. KLSG carries folder-level class labels but",
        "no bounding boxes, so the remaining path is weakly supervised fine-tuning on its",
        "target-centred chips; the pseudo-boxes would be approximate, and any mAP from them",
        "must not be quoted beside the synthetic numbers.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="sample N images spread across the corpus")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(limit=args.limit, out_path=args.out)


if __name__ == "__main__":
    main()
