"""Ultralytics detector wrapper — TridentNet Brain A.

Brain A is the supervised object detector of the TridentNet ensemble. It
ingests the SAHI-style tiles produced by the M2 preprocessing pipeline
(:mod:`sonar_core.preprocess.tiler`): float32 chips in ``[0, 1]`` with NaN
marking out-of-swath fill, cut from a ground-range image whose rows are pings
and whose columns are ground-range distance from nadir (column 0 at nadir on
*both* sides, acoustic shadows always extending toward increasing column).

Design points driven by the sonar geometry:

* **Grayscale to 3 channels by replication.** Side-scan imagery has a single
  intensity channel; replicating it satisfies the RGB stem of pretrained
  backbones without inventing chroma. NaN fill becomes 0 — the same level as
  the blanked water column — so swath edges read as dark, textureless seabed
  absence rather than synthetic texture a detector could fire on.
* **Global coordinates immediately.** Every box is mapped from tile pixels to
  global inclusive ``(ping0..ping1, col0..col1)`` via the tile's recorded
  ``(row0, col0)`` origin, so a detection stays traceable through
  :meth:`GroundImage.to_slant_sample` back to the raw ping and its NAV record
  for georeferencing.
* **Cross-tile dedup in global space.** Tiles overlap (default 25%) so small
  debris is never split by every boundary — the flip side is that one target
  near a boundary is detected in up to four tiles. Greedy NMS per
  ``(side, cls)`` in global coordinates keeps the highest-confidence copy.
  Port and starboard are physically distinct seabed strips that merely share
  a column index space, so boxes on different sides must never merge.
* **No mirror augmentation across columns, ever.** Highlight-then-shadow
  order along the column axis is fixed by the acoustics; flipping it would
  train on physically impossible imagery. This wrapper performs no
  augmentation, and training code must respect the same rule.

When no trained sonar checkpoint exists (``weights/detector.pt``), the
wrapper falls back to the COCO-pretrained model named in the config so the
plumbing can be smoke-tested end to end; class names then come from the
pretrained model verbatim and a warning records that sonar classes are
unavailable.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from sonar_core.preprocess.tiler import Tile

if TYPE_CHECKING:  # pragma: no cover - typing only; ultralytics import is lazy
    from ultralytics import YOLO

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: YAML mirror of :data:`DEFAULTS`; deep-merged over it when present.
CONFIG_PATH: Path = _REPO_ROOT / "configs" / "detector.yaml"

#: Trained sonar checkpoint used when ``Detector(weights=None)`` and it exists.
DEFAULT_WEIGHTS_PATH: Path = _REPO_ROOT / "weights" / "detector.pt"

#: Full-scale value of an 8-bit image; tiles arrive normalized to [0, 1].
_UINT8_MAX = 255.0

#: Single source of truth for every tunable; ``configs/detector.yaml`` mirrors
#: this structure (with units in its comments) and user config dicts are
#: deep-merged on top, so YAML and code can never disagree about a default.
DEFAULTS: dict[str, Any] = {
    "model": "yolov8n.pt",  # pretrained fallback when weights/detector.pt is absent
    "imgsz": 640,  # inference size (pixels)
    "conf": 0.25,  # per-tile minimum confidence
    "iou": 0.5,  # in-tile NMS IoU (Ultralytics)
    "dedup_iou": 0.45,  # cross-tile global NMS IoU
    "max_det": 100,  # detection cap per tile
    "batch": 8,  # tiles per predict() call
}


@dataclass(frozen=True)
class Detection:
    """One detector find, in global ground-image coordinates.

    Extents are **inclusive** pixel indices: ``ping0..ping1`` are rows of the
    side's ground image (ping indices, 1:1 with NAV records) and
    ``col0..col1`` are ground-range columns (column 0 at nadir). Inclusive
    coordinates make a single-pixel target a zero-width-free box, which keeps
    IoU well defined for the smallest debris (a drum can be under ten pixels
    across at survey range scales).
    """

    side: str  # "port" | "starboard"
    ping0: int  # global ground-image row extent, inclusive
    ping1: int
    col0: int  # global ground-range column extent, inclusive
    col1: int
    cls: str  # name from tridentnet.classes (COCO names pass through verbatim
    # when running on pretrained fallback weights)
    score: float  # model confidence 0..1
    brain: str = "A"
    tile_index: int = -1  # Tile.index the box came from (-1: unknown)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively overlay *override* onto *base*; neither input is mutated.

    Same contract as the preprocessing pipeline's merge, so partial YAML files
    and partial config dicts compose identically across modules.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _safe_progress(
    progress: Callable[[str, float], None] | None, stage: str, fraction: float
) -> None:
    """Invoke the progress observer; a broken observer must never abort a run.

    Same ``(stage_name, fraction)`` contract as
    :func:`sonar_core.preprocess.pipeline.preprocess`: fractions are
    monotonically nondecreasing in ``[0, 1]`` and the final call is
    ``("done", 1.0)``.
    """
    if progress is None:
        return
    try:
        progress(stage, fraction)
    except Exception:  # noqa: BLE001 - observer failures are deliberately swallowed
        pass


def load_config(
    override: dict[str, Any] | None = None, path: Path = CONFIG_PATH
) -> dict[str, Any]:
    """Resolve the effective detector config: DEFAULTS < YAML file < override."""
    cfg = dict(DEFAULTS)
    if path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a mapping, got {type(data).__name__}")
        cfg = _deep_merge(cfg, data)
    return _deep_merge(cfg, override or {})


def box_iou(a: Detection, b: Detection) -> float:
    """Intersection-over-union of two detections in inclusive coordinates.

    With inclusive extents a box spans ``(ping1 - ping0 + 1)`` rows and
    ``(col1 - col0 + 1)`` columns, so a single-pixel detection has area 1 and
    two identical single-pixel boxes score IoU 1.0 — essential for deduping
    the smallest debris returns, which corner-based float IoU would zero out.
    Boxes on different sides never overlap physically; callers group by side
    before comparing (this function only measures geometry).
    """
    inter_h = min(a.ping1, b.ping1) - max(a.ping0, b.ping0) + 1
    inter_w = min(a.col1, b.col1) - max(a.col0, b.col0) + 1
    if inter_h <= 0 or inter_w <= 0:
        return 0.0
    inter = float(inter_h * inter_w)
    area_a = float((a.ping1 - a.ping0 + 1) * (a.col1 - a.col0 + 1))
    area_b = float((b.ping1 - b.ping0 + 1) * (b.col1 - b.col0 + 1))
    return inter / (area_a + area_b - inter)


def merge_detections(dets: Iterable[Detection], iou_thresh: float) -> list[Detection]:
    """Cross-tile dedup: greedy NMS per ``(side, cls)`` in global coordinates.

    Tile overlap re-detects one physical target in up to four tiles; those
    copies land on (nearly) the same global footprint, so per-group greedy
    NMS — keep the highest score, suppress every remaining box whose IoU with
    it is ``>= iou_thresh`` — collapses them to the most confident copy.
    Suppression is non-transitive (classic NMS): a box is only ever compared
    against *kept* boxes, so a chain A-B-C where only adjacent pairs overlap
    keeps A and C. Different classes or different sides are never compared:
    port and starboard are distinct seabed strips sharing a column index
    space, and cross-class merging would hide genuine co-located returns.

    Output is sorted by descending score (stable for ties).
    """
    if not 0.0 < iou_thresh <= 1.0:
        raise ValueError(f"iou_thresh must be in (0, 1], got {iou_thresh}")

    groups: dict[tuple[str, str], list[Detection]] = defaultdict(list)
    for d in dets:
        groups[(d.side, d.cls)].append(d)

    kept: list[Detection] = []
    for group in groups.values():
        if len(group) == 1:
            kept.extend(group)
            continue
        boxes = np.array(
            [(d.ping0, d.ping1, d.col0, d.col1) for d in group], dtype=np.float64
        )
        scores = np.array([d.score for d in group], dtype=np.float64)
        p0, p1, c0, c1 = boxes.T
        areas = (p1 - p0 + 1.0) * (c1 - c0 + 1.0)
        order = np.argsort(-scores, kind="stable")
        while order.size:
            i = order[0]
            kept.append(group[int(i)])
            rest = order[1:]
            if rest.size == 0:
                break
            ih = np.minimum(p1[i], p1[rest]) - np.maximum(p0[i], p0[rest]) + 1.0
            iw = np.minimum(c1[i], c1[rest]) - np.maximum(c0[i], c0[rest]) + 1.0
            inter = np.clip(ih, 0.0, None) * np.clip(iw, 0.0, None)
            iou = inter / (areas[i] + areas[rest] - inter)
            order = rest[iou < iou_thresh]
    return sorted(kept, key=lambda d: -d.score)


def _tile_to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Float [0,1] tile (NaN swath fill) -> HxWx3 uint8 for the detector.

    NaN marks ground range a ping's slant range and altitude cannot reach —
    acoustically empty, so it maps to 0, the same level as the blanked water
    column. The single sonar intensity channel is replicated across RGB so
    ImageNet/COCO-pretrained stems accept it without inventing chroma.
    """
    arr = np.nan_to_num(np.asarray(image, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    u8 = np.round(np.clip(arr, 0.0, 1.0) * _UINT8_MAX).astype(np.uint8)
    return np.repeat(u8[:, :, None], 3, axis=2)


def _to_numpy(x: Any) -> np.ndarray:
    """Torch tensor (any device) or array-like -> numpy array."""
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)


def _result_class_name(names: Any, class_id: int) -> str:
    """Class-id -> name from an Ultralytics result's ``names`` mapping."""
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _boxes_to_detections(tile: Tile, result: Any) -> list[Detection]:
    """Map one tile's Ultralytics result to global-coordinate detections.

    ``result.boxes.xyxy`` is in the ORIGINAL tile pixel space (Ultralytics
    un-letterboxes before returning), with x along ground-range columns and y
    along pings. A continuous ``[x1, x2)`` extent becomes the inclusive pixel
    run ``floor(x1) .. ceil(x2) - 1``, clamped to the tile so the global box
    can never leave the tile's footprint (whose every pixel is a real ground
    sample by tiler construction — the last tile is shifted, never padded).
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = _to_numpy(boxes.xyxy).astype(np.float64).reshape(-1, 4)
    if xyxy.size == 0:
        return []
    conf = _to_numpy(boxes.conf).astype(np.float64).ravel()
    cls_ids = _to_numpy(boxes.cls).astype(np.int64).ravel()
    names = getattr(result, "names", None)

    h, w = tile.image.shape
    c0 = np.clip(np.floor(xyxy[:, 0]), 0, w - 1).astype(np.int64)
    r0 = np.clip(np.floor(xyxy[:, 1]), 0, h - 1).astype(np.int64)
    c1 = np.clip(np.ceil(xyxy[:, 2]) - 1, 0, w - 1).astype(np.int64)
    r1 = np.clip(np.ceil(xyxy[:, 3]) - 1, 0, h - 1).astype(np.int64)
    c1 = np.maximum(c1, c0)  # degenerate (sub-pixel) boxes stay 1 pixel wide
    r1 = np.maximum(r1, r0)

    return [
        Detection(
            side=tile.side,
            ping0=int(tile.row0 + r0[k]),
            ping1=int(tile.row0 + r1[k]),
            col0=int(tile.col0 + c0[k]),
            col1=int(tile.col0 + c1[k]),
            cls=_result_class_name(names, int(cls_ids[k])),
            score=float(conf[k]),
            tile_index=int(tile.index),
        )
        for k in range(xyxy.shape[0])
    ]


class Detector:
    """TridentNet Brain A: Ultralytics YOLO over preprocessed sonar tiles.

    The underlying model is loaded lazily on first use, so constructing a
    ``Detector`` (and resolving its config/weights) needs neither network nor
    torch — important on edge boxes where Brain A may be disabled entirely.
    """

    def __init__(
        self,
        weights: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Resolve config and weights; the model itself loads on first use.

        Parameters
        ----------
        weights:
            Path to a trained ``.pt`` checkpoint. ``None`` selects
            :data:`DEFAULT_WEIGHTS_PATH` (``weights/detector.pt``) if it
            exists, else falls back to the COCO-pretrained asset named by
            ``config["model"]`` — fine for smoke tests; class names then come
            from the pretrained model and a warning records that sonar
            classes are unavailable.
        config:
            Partial override deep-merged over ``configs/detector.yaml``
            (which is itself merged over :data:`DEFAULTS`).
        """
        self.config: dict[str, Any] = load_config(config)
        self.using_pretrained_fallback: bool = False
        if weights is not None:
            self.weights: str = str(weights)
        elif DEFAULT_WEIGHTS_PATH.is_file():
            self.weights = str(DEFAULT_WEIGHTS_PATH)
        else:
            self.weights = str(self.config["model"])
            self.using_pretrained_fallback = True
            logger.warning(
                "weights/detector.pt not found; falling back to COCO-pretrained %r "
                "— sonar classes are unavailable (class names come from the "
                "pretrained model)",
                self.weights,
            )
        self._model: YOLO | None = None

    @property
    def model(self) -> YOLO:
        """The Ultralytics model, loaded (and possibly auto-downloaded) lazily."""
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.weights)
        return self._model

    def class_names(self) -> list[str]:
        """Model class names ordered by class id (sonar classes for a trained
        checkpoint; COCO names on the pretrained fallback)."""
        names = self.model.names
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names)]
        return [str(n) for n in names]

    def _device(self) -> str:
        """Inference device: CUDA when available, else CPU."""
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _predict(self, images: list[np.ndarray]) -> list[Any]:
        """Run one batched Ultralytics predict; returns one Results per image.

        Ultralytics letterboxes each tile to ``imgsz`` internally and maps
        boxes back, so ``results[i].boxes.xyxy`` is already in the original
        tile pixel space (verified by the deterministic mapping test).
        """
        cfg = self.config
        return self.model.predict(
            source=images,
            imgsz=int(cfg["imgsz"]),
            conf=float(cfg["conf"]),
            iou=float(cfg["iou"]),
            max_det=int(cfg["max_det"]),
            device=self._device(),
            verbose=False,
        )

    def detect_tiles(
        self,
        tiles: Sequence[Tile],
        progress: Callable[[str, float], None] | None = None,
    ) -> list[Detection]:
        """Detect debris in preprocessed tiles; return deduped global boxes.

        Each :class:`Tile` image (float ``[0, 1]``, NaN swath fill) is
        converted to replicated-channel uint8, predicted in batches of
        ``config["batch"]``, mapped to global inclusive ``(ping, column)``
        extents via the tile origin, then cross-tile deduplicated per
        ``(side, cls)`` with greedy NMS at ``config["dedup_iou"]`` — see
        :func:`merge_detections` for why overlap-tiling makes this mandatory.

        *progress* follows the pipeline observer contract:
        ``(stage_name, fraction)`` with nondecreasing fractions in
        ``[0, 1]``, ending with ``("done", 1.0)``; its exceptions are
        swallowed so a UI glitch never kills a survey run.
        """
        tiles = list(tiles)
        if not tiles:
            _safe_progress(progress, "done", 1.0)
            return []

        batch = int(self.config["batch"])
        if batch < 1:
            raise ValueError(f"config['batch'] must be >= 1, got {batch}")
        n_batches = (len(tiles) + batch - 1) // batch
        total_stages = n_batches + 1  # + the dedup pass

        raw: list[Detection] = []
        for bi in range(n_batches):
            _safe_progress(progress, "predict", bi / total_stages)
            chunk = tiles[bi * batch : (bi + 1) * batch]
            images = [_tile_to_uint8_rgb(t.image) for t in chunk]
            results = self._predict(images)
            for tile, result in zip(chunk, results, strict=True):
                raw.extend(_boxes_to_detections(tile, result))

        _safe_progress(progress, "dedup", n_batches / total_stages)
        merged = merge_detections(raw, float(self.config["dedup_iou"]))
        _safe_progress(progress, "done", 1.0)
        return merged
