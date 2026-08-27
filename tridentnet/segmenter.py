"""TridentNet Brain B: compact U-Net net/rope segmentation (blueprint N-01/L2).

Brain A draws boxes; Brain B draws *footprints*. Ghost nets and pipelines are
filamentous — thin, sinuous, sprawling — so an axis-aligned box around one is
mostly background: its area overestimates the hazard, its centroid can miss
the material entirely, and its geo-corners overstate the recovery zone. A
pixel mask fixes all three, which is why the ensemble contract
(:mod:`tridentnet.ensemble`) reserves provenance ``"B"`` for mask-refined
candidates. This module deliberately exposes only clean, standalone APIs
(:class:`Segmenter.predict_mask`, :class:`Segmenter.refine_detections`);
wiring into the ensemble happens later.

Design points driven by the sonar imagery:

* **Pure-torch compact U-Net** (4 encoder stages, base 16, ~1.9M params): a
  net highlight against seabed speckle is a texture/contrast problem, not a
  semantics problem — a small model trains in minutes on CPU and fits edge
  boxes, and skip connections preserve the pixel-thin structures that
  downsampling alone would erase. No torchvision/segformer dependency.
* **Single channel in [0, 1], NaN -> 0.** Tiles arrive as float chips with
  NaN marking out-of-swath fill; 0 is the blanked-water-column level, so
  swath edges read as dark seabed absence (same convention as Brains A and C).
  Out-of-swath pixels are forced False in the output mask — nothing beyond
  the swath was ever ensonified, so nothing there can be net.
* **BCE + Dice loss.** A net covers a small fraction of a chip; plain BCE is
  dominated by background and converges to all-empty masks. The Dice term
  scores overlap ratio directly, making foreground pixels count regardless of
  class imbalance; BCE keeps per-pixel gradients smooth early on.
* **Never mirror across columns.** Highlight-then-shadow order along the
  column axis is fixed by the acoustics. No augmentation is performed here;
  any future augmentation must respect the same rule.

Weights are trained by :func:`train_segmenter` (see
``scripts/train_segmenter.py``); like Brain C, :class:`Segmenter` raises
``FileNotFoundError`` when no checkpoint exists rather than inventing masks.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.nn import functional as F  # noqa: N812 - torch's canonical alias

from sonar_core.preprocess.tiler import Tile
from tridentnet.detector import Detection
from tridentnet.segdata import FOREGROUND_CLASSES

_REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = _REPO_ROOT / "configs" / "segmenter.yaml"
DEFAULT_WEIGHTS_PATH = _REPO_ROOT / "weights" / "segmenter.pt"

#: Encoder downsampling stages; inputs are padded to a multiple of 2**stages
#: so every pooled resolution divides evenly and the decoder inverts exactly.
N_STAGES = 4
PAD_MULTIPLE = 2**N_STAGES

DEFAULTS: dict[str, Any] = {
    "base_channels": 16,  # first encoder width; doubles per stage (4 stages)
    "epochs": 20,  # passes over the chip set
    "lr": 1.0e-3,  # Adam learning rate
    "batch": 8,  # chips per step
    "threshold": 0.5,  # sigmoid cut turning probabilities into mask pixels
    "dice_eps": 1.0,  # soft-Dice smoothing (stabilizes empty-mask chips)
    "refine_margin_px": 4,  # box expansion when intersecting a mask with a detection
    "min_mask_frac": 0.05,  # min mask fraction inside a box to accept refinement
}


class _DoubleConv(nn.Module):
    """(conv3x3 -> BN -> ReLU) x2 — the classic U-Net stage block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Compact U-Net: 4 encoder stages (base 16 -> 128), bottleneck, skips.

    ``forward(x)`` takes ``(B, 1, H, W)`` intensities in ``[0, 1]`` and
    returns per-pixel logits ``(B, 1, H, W)`` at the *input* size: arbitrary
    (odd) sizes are padded internally to a multiple of :data:`PAD_MULTIPLE`
    (reflect where possible, so the pad continues real seabed texture instead
    of a synthetic edge) and the output is cropped back.
    """

    def __init__(self, base: int = 16) -> None:
        super().__init__()
        widths = [base * 2**i for i in range(N_STAGES)]  # 16, 32, 64, 128
        self.enc = nn.ModuleList()
        in_ch = 1
        for w in widths:
            self.enc.append(_DoubleConv(in_ch, w))
            in_ch = w
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _DoubleConv(widths[-1], widths[-1] * 2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        up_in = widths[-1] * 2
        for w in reversed(widths):
            self.up.append(nn.ConvTranspose2d(up_in, w, 2, stride=2))
            self.dec.append(_DoubleConv(w * 2, w))
            up_in = w
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = int(x.shape[-2]), int(x.shape[-1])
        pad_h, pad_w = (-h) % PAD_MULTIPLE, (-w) % PAD_MULTIPLE
        if pad_h or pad_w:
            mode = "reflect" if pad_h < h and pad_w < w else "constant"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        skips: list[torch.Tensor] = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips), strict=True):
            x = up(x)
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(x)[..., :h, :w]


def _load_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Effective config: DEFAULTS < configs/segmenter.yaml < override."""
    merged = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        merged.update(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {})
    merged.update(config or {})
    return merged


def _safe_progress(
    progress: Callable[[str, float], None] | None, stage: str, fraction: float
) -> None:
    """Invoke the progress observer; a broken observer must never abort a run."""
    if progress is None:
        return
    try:
        progress(stage, fraction)
    except Exception:  # noqa: BLE001 - observer failures are deliberately swallowed
        pass


def _load_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one split of a :func:`tridentnet.segdata.build_mask_dataset` tree.

    Returns ``(images, masks)`` as ``(n, 1, H, W)`` float32 in [0, 1] and
    ``(n, 1, H, W)`` float32 in {0, 1}. Chips are zero-padded (image 0 =
    blanked-water level, mask 0 = background) up to the split's maximum
    extent rounded to :data:`PAD_MULTIPLE`, so mixed edge-clamped chip sizes
    still stack into one batchable array.
    """
    img_dir = data_dir / "images" / split
    msk_dir = data_dir / "masks" / split
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for img_path in sorted(img_dir.glob("*.png")):
        msk_path = msk_dir / img_path.name
        if not msk_path.exists():
            continue
        img = np.asarray(Image.open(img_path), dtype=np.float32) / 255.0
        msk = (np.asarray(Image.open(msk_path)) > 127).astype(np.float32)
        pairs.append((img, msk))
    if not pairs:
        return (
            np.zeros((0, 1, 0, 0), dtype=np.float32),
            np.zeros((0, 1, 0, 0), dtype=np.float32),
        )
    max_h = max(img.shape[0] for img, _ in pairs)
    max_w = max(img.shape[1] for img, _ in pairs)
    max_h += (-max_h) % PAD_MULTIPLE
    max_w += (-max_w) % PAD_MULTIPLE
    images = np.zeros((len(pairs), 1, max_h, max_w), dtype=np.float32)
    masks = np.zeros((len(pairs), 1, max_h, max_w), dtype=np.float32)
    for i, (img, msk) in enumerate(pairs):
        images[i, 0, : img.shape[0], : img.shape[1]] = img
        masks[i, 0, : msk.shape[0], : msk.shape[1]] = msk
    return images, masks


def _dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    """Soft Dice loss per sample, averaged over the batch.

    Directly optimizes overlap ratio, so a net covering 5% of a chip weighs
    as much as its background — the imbalance-immunity BCE lacks. *eps*
    smooths the ratio so an all-background chip with an all-background
    prediction scores a perfect 0 loss instead of 0/0.
    """
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def train_segmenter(
    data_dir: str | Path,
    out_path: str | Path = DEFAULT_WEIGHTS_PATH,
    config: dict[str, Any] | None = None,
    device: str | None = None,
    seed: int = 0,
    progress: Callable[[str, float], None] | None = None,
) -> Path:
    """Train the U-Net on a mask-chip dataset and save weights + honest val Dice.

    Loss is BCE-with-logits plus soft Dice (see :func:`_dice_loss`), Adam,
    with ``epochs``/``lr``/``batch`` from the merged config. After training,
    validation Dice is computed the honest way — hard masks at
    ``config["threshold"]`` on the held-out val scenes, micro-averaged over
    all val pixels (``2 * |pred & gt| / (|pred| + |gt|)``) — and stored in
    the checkpoint as ``val_dice`` next to the per-epoch mean train losses
    (``train_losses``), so any consumer can audit training instead of
    trusting it. Batching is manual (no DataLoader workers — Windows-safe).

    Raises ``ValueError`` when either split is empty: a model without a
    validation score would ship an unverifiable ``val_dice``.
    """
    cfg = _load_config(config)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    data_dir = Path(data_dir)
    train_imgs, train_msks = _load_split(data_dir, "train")
    val_imgs, val_msks = _load_split(data_dir, "val")
    if len(train_imgs) == 0:
        raise ValueError(f"no training chips under {data_dir}; run build_mask_dataset first")
    if len(val_imgs) == 0:
        raise ValueError(
            f"no validation chips under {data_dir}; build the dataset with val_frac > 0"
        )

    train_x = torch.from_numpy(train_imgs)
    train_y = torch.from_numpy(train_msks)

    model = UNet(int(cfg["base_channels"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    bce = nn.BCEWithLogitsLoss()
    batch = int(cfg["batch"])
    eps = float(cfg["dice_eps"])
    epochs = int(cfg["epochs"])

    train_losses: list[float] = []
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(train_x))
        epoch_losses: list[float] = []
        for i in range(0, len(train_x), batch):
            idx = perm[i : i + batch]
            xb = train_x[idx].to(device)
            yb = train_y[idx].to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = bce(logits, yb) + _dice_loss(logits, yb, eps)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        train_losses.append(float(np.mean(epoch_losses)))
        _safe_progress(progress, "train_segmenter", (epoch + 1) / (epochs + 1))

    # Honest held-out score: hard masks, micro-averaged Dice over all val pixels.
    _safe_progress(progress, "validate", epochs / (epochs + 1))
    model.eval()
    threshold = float(cfg["threshold"])
    inter = pred_sum = gt_sum = 0.0
    with torch.no_grad():
        for i in range(0, len(val_imgs), batch):
            xb = torch.from_numpy(val_imgs[i : i + batch]).to(device)
            gt = val_msks[i : i + batch] > 0.5
            pred = (torch.sigmoid(model(xb)).cpu().numpy() > threshold)
            inter += float(np.logical_and(pred, gt).sum())
            pred_sum += float(pred.sum())
            gt_sum += float(gt.sum())
    denom = pred_sum + gt_sum
    val_dice = (2.0 * inter / denom) if denom > 0 else 1.0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "base_channels": int(cfg["base_channels"]),
            "config": cfg,
            "val_dice": float(val_dice),
            "train_losses": train_losses,
        },
        out_path,
    )
    _safe_progress(progress, "done", 1.0)
    return out_path


class Segmenter:
    """Inference wrapper: tiles in, boolean net/rope masks out.

    Same weight-loading contract as Brain C's ``AnomalyDetector``: missing
    weights raise ``FileNotFoundError`` immediately — a segmenter that
    silently predicts nothing would masquerade as "no nets found".
    """

    def __init__(
        self,
        weights: str | Path | None = None,
        config: dict[str, Any] | None = None,
        device: str | None = None,
    ) -> None:
        path = Path(weights) if weights is not None else DEFAULT_WEIGHTS_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"segmenter weights not found at {path}; run scripts/train_segmenter.py"
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        # Merge order for inference: DEFAULTS < frozen training config <
        # live configs/segmenter.yaml < caller override. The checkpoint's
        # frozen config seeds training-time values, but the YAML must stay
        # live-tunable for inference knobs (threshold, refine margins) —
        # _load_config applies the YAML and caller on top of this base.
        base = {**DEFAULTS, **payload.get("config", {})}
        merged = dict(base)
        if CONFIG_PATH.exists():
            merged.update(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {})
        merged.update(config or {})
        self.cfg = merged
        self.val_dice = float(payload.get("val_dice", float("nan")))
        self.model = UNet(int(payload["base_channels"]))
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device).eval()

    def predict_mask(self, tile_img: np.ndarray) -> np.ndarray:
        """Boolean net/rope mask for one tile image ((H, W) float [0, 1], NaN ok).

        NaN (out-of-swath fill) becomes 0 for the network — the blanked-water
        level, matching training chips — and is forced ``False`` in the output:
        un-ensonified ground can never contain a detectable net. The U-Net pads
        internally to a multiple of :data:`PAD_MULTIPLE` and crops back, so any
        tile size works. Threshold comes from ``config["threshold"]``.
        """
        arr = np.asarray(tile_img, dtype=np.float32)
        clean = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        with torch.no_grad():
            x = torch.from_numpy(clean[None, None]).to(self.device)
            prob = torch.sigmoid(self.model(x))[0, 0].cpu().numpy()
        mask = prob > float(self.cfg["threshold"])
        mask[~np.isfinite(arr)] = False
        return mask

    def _tile_for(self, det: Detection, tiles: Sequence[Tile], ping: int, col: int) -> Tile | None:
        """The tile holding (*ping*, *col*): the detection's own source tile
        when it still contains the centre, else the first same-side tile that
        does (overlap-tiling guarantees interior points at least one home)."""
        for tile in tiles:
            if tile.index == det.tile_index and tile.side == det.side:
                if tile.contains(ping, col):
                    return tile
                break
        for tile in tiles:
            if tile.side == det.side and tile.contains(ping, col):
                return tile
        return None

    def refine_detections(
        self,
        detections: Sequence[Detection],
        tiles: Sequence[Tile],
    ) -> list[tuple[Detection, np.ndarray | None]]:
        """Tighten foreground-class boxes to their predicted mask footprint.

        For each detection of a :data:`FOREGROUND_CLASSES` class: find the
        tile containing the box centre, predict its mask (cached per tile —
        several boxes can share one), intersect the mask with the detection
        box expanded by ``config["refine_margin_px"]`` (the margin forgives
        slightly-tight Brain A boxes without letting the mask wander), and if
        the mask covers at least ``config["min_mask_frac"]`` of that region,
        return a refined :class:`Detection` — the tight bounding box of the
        mask, ``"B"`` appended to its brain provenance via
        ``dataclasses.replace`` — plus the local mask cropped to the refined
        box (rows/cols align 1:1 with the refined extents). Otherwise (below
        the fraction, off-tile centre, or a non-foreground class) the original
        detection passes through with ``None``: a wreck box is Brain A/C
        business, and a mask too sparse to trust must not shrink a real find.
        """
        margin = int(self.cfg["refine_margin_px"])
        min_frac = float(self.cfg["min_mask_frac"])
        mask_cache: dict[int, np.ndarray] = {}
        out: list[tuple[Detection, np.ndarray | None]] = []
        for det in detections:
            if det.cls not in FOREGROUND_CLASSES:
                out.append((det, None))
                continue
            centre_ping = (det.ping0 + det.ping1) // 2
            centre_col = (det.col0 + det.col1) // 2
            tile = self._tile_for(det, tiles, centre_ping, centre_col)
            if tile is None:
                out.append((det, None))
                continue
            mask = mask_cache.get(tile.index)
            if mask is None:
                mask = self.predict_mask(tile.image)
                mask_cache[tile.index] = mask
            h, w = mask.shape
            r0 = max(det.ping0 - tile.row0 - margin, 0)
            r1 = min(det.ping1 - tile.row0 + margin, h - 1)
            c0 = max(det.col0 - tile.col0 - margin, 0)
            c1 = min(det.col1 - tile.col0 + margin, w - 1)
            if r1 < r0 or c1 < c0:
                out.append((det, None))
                continue
            region = mask[r0 : r1 + 1, c0 : c1 + 1]
            if not region.any() or float(region.mean()) < min_frac:
                out.append((det, None))
                continue
            ys, xs = np.nonzero(region)
            rr0, rr1 = r0 + int(ys.min()), r0 + int(ys.max())
            cc0, cc1 = c0 + int(xs.min()), c0 + int(xs.max())
            refined = dataclasses.replace(
                det,
                ping0=int(tile.row0 + rr0),
                ping1=int(tile.row0 + rr1),
                col0=int(tile.col0 + cc0),
                col1=int(tile.col0 + cc1),
                brain=det.brain if "B" in det.brain else det.brain + "B",
                tile_index=int(tile.index),
            )
            out.append((refined, mask[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()))
        return out
