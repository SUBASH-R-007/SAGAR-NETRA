"""TridentNet Brain C: convolutional-autoencoder anomaly detection.

Trained ONLY on clean seabed tiles, the autoencoder learns to reconstruct
"normal" texture — speckle statistics, ripples, reflectivity patches. Any
structure it has never seen (a man-made object, an unusual seabed feature)
reconstructs poorly, so the per-pixel reconstruction error is an open-set
anomaly map: connected blobs of high error become ``unknown_anomaly``
candidates without any labeled examples. This is what catches debris classes
the supervised detector was never trained on.

The network is fully convolutional (the 128-channel latent is a spatial
bottleneck at 1/8 resolution, not a flat vector), so tiles of any size flow
through unchanged. The detection threshold is calibrated at training time:
the ``thresh_q`` quantile of the smoothed reconstruction error over held-out
clean tiles is stored with the weights, so inference needs no tuning.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.ndimage import gaussian_filter, label
from torch import nn

from sonar_core.preprocess.tiler import Tile
from tridentnet.classes import ANOMALY_CLASS
from tridentnet.detector import Detection, merge_detections

_REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = _REPO_ROOT / "configs" / "anomaly.yaml"
DEFAULT_WEIGHTS_PATH = _REPO_ROOT / "weights" / "anomaly.pt"

DEFAULTS: dict[str, Any] = {
    "base_channels": 16,  # first encoder width; doubles per stage
    "latent_channels": 128,  # spatial bottleneck depth at 1/8 resolution
    "epochs": 12,
    "lr": 1.0e-3,
    "batch": 8,
    "patch": 128,  # training crop side (pixels)
    "thresh_q": 0.995,  # clean-error quantile that defines "anomalous"
    "smooth_sigma": 2.0,  # error-map smoothing before thresholding (pixels)
    "min_blob_px": 40,  # discard smaller error blobs (speckle flukes)
    "dedup_iou": 0.4,  # cross-tile NMS for anomaly boxes
    "score_scale": 2.0,  # error/threshold ratio mapped to score 0..1 at this ratio
    # Ground columns adjacent to nadir are excluded from anomaly detection:
    # the slant->ground resampling stretches a handful of samples across many
    # columns there, producing smooth streaks whose reconstruction error is
    # systematically high on perfectly normal seabed. The supervised detector
    # still covers that strip (~1.3 m at 4 cm resolution).
    "nadir_guard_cols": 32,
}


class ConvAE(nn.Module):
    """Six-conv fully-convolutional autoencoder (3 down, 3 up)."""

    def __init__(self, base: int = 16, latent: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, base, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, latent, 3, stride=1, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent, base * 4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _load_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        merged.update(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {})
    merged.update(config or {})
    return merged


def _tiles_to_patches(
    tiles: list[np.ndarray], patch: int, rng: np.random.Generator, per_tile: int = 4
) -> np.ndarray:
    """Random clean crops, NaN-free, stacked as (n, 1, patch, patch) float32."""
    crops = []
    for img in tiles:
        h, w = img.shape
        if h < patch or w < patch:
            continue
        for _ in range(per_tile):
            r = int(rng.integers(0, h - patch + 1))
            c = int(rng.integers(0, w - patch + 1))
            crop = img[r : r + patch, c : c + patch]
            # Keep patches straddling the nadir/swath NaN edges (filled to 0,
            # exactly as inference fills them): the autoencoder must learn
            # that context or it will flag every swath edge as anomalous.
            if np.isfinite(crop).mean() < 0.5:
                continue
            crops.append(np.nan_to_num(crop, nan=0.0))
    if not crops:
        raise ValueError("no clean patches could be sampled from the given tiles")
    return np.stack(crops)[:, None, :, :].astype(np.float32)


def train_anomaly(
    background_tiles: list[np.ndarray],
    out_path: str | Path = DEFAULT_WEIGHTS_PATH,
    config: dict[str, Any] | None = None,
    device: str | None = None,
    seed: int = 0,
    progress: Callable[[str, float], None] | None = None,
) -> Path:
    """Train on clean seabed tiles ([0,1] float arrays) and save weights +
    the calibrated error threshold."""
    cfg = _load_config(config)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    patches = _tiles_to_patches(background_tiles, int(cfg["patch"]), rng)
    n_val = max(len(patches) // 10, 1)
    train_x = torch.from_numpy(patches[n_val:])
    val_x = torch.from_numpy(patches[:n_val])

    model = ConvAE(int(cfg["base_channels"]), int(cfg["latent_channels"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    loss_fn = nn.MSELoss()
    batch = int(cfg["batch"])

    model.train()
    epochs = int(cfg["epochs"])
    for epoch in range(epochs):
        perm = torch.randperm(len(train_x))
        for i in range(0, len(train_x), batch):
            xb = train_x[perm[i : i + batch]].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            optimizer.step()
        if progress is not None:
            progress("train_anomaly", (epoch + 1) / epochs)

    # Calibrate the anomaly threshold on held-out clean patches.
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(0, len(val_x), batch):
            xb = val_x[i : i + batch].to(device)
            err = (model(xb) - xb).abs().squeeze(1).cpu().numpy()
            for e in err:
                errors.append(gaussian_filter(e, sigma=float(cfg["smooth_sigma"])))
    threshold = float(np.quantile(np.stack(errors), float(cfg["thresh_q"])))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "base_channels": int(cfg["base_channels"]),
            "latent_channels": int(cfg["latent_channels"]),
            "threshold": threshold,
            "config": cfg,
        },
        out_path,
    )
    return out_path


@dataclass
class AnomalyMap:
    """Per-tile smoothed reconstruction error, for the dashboard heat view."""

    tile_index: int
    error: np.ndarray


class AnomalyDetector:
    """Inference wrapper: tiles in, ``unknown_anomaly`` Detections out."""

    def __init__(
        self,
        weights: str | Path | None = None,
        config: dict[str, Any] | None = None,
        device: str | None = None,
    ) -> None:
        path = Path(weights) if weights is not None else DEFAULT_WEIGHTS_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"anomaly weights not found at {path}; run scripts/train_anomaly.py"
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.cfg = _load_config({**payload.get("config", {}), **(config or {})})
        self.threshold = float(payload["threshold"])
        self.model = ConvAE(int(payload["base_channels"]), int(payload["latent_channels"]))
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device).eval()

    def error_map(self, tile_img: np.ndarray) -> np.ndarray:
        """Smoothed |x - reconstruction| for one tile ([0,1], NaN allowed)."""
        clean = np.nan_to_num(tile_img, nan=0.0).astype(np.float32)
        h, w = clean.shape
        # Pad to a multiple of 8 so the three stride-2 stages invert exactly.
        pad_h, pad_w = (-h) % 8, (-w) % 8
        padded = np.pad(clean, ((0, pad_h), (0, pad_w)), mode="reflect")
        with torch.no_grad():
            x = torch.from_numpy(padded[None, None]).to(self.device)
            err = (self.model(x) - x).abs()[0, 0].cpu().numpy()
        err = gaussian_filter(err[:h, :w], sigma=float(self.cfg["smooth_sigma"]))
        # Mask AFTER smoothing so no neighbouring error bleeds back into the
        # out-of-swath zone.
        err[~np.isfinite(tile_img)] = 0.0
        return err

    def detect_tiles(
        self,
        tiles: list[Tile],
        progress: Callable[[str, float], None] | None = None,
    ) -> list[Detection]:
        candidates: list[Detection] = []
        n = max(len(tiles), 1)
        nadir_guard = int(self.cfg["nadir_guard_cols"])
        for i, tile in enumerate(tiles):
            err = self.error_map(tile.image)
            guard_local = nadir_guard - tile.col0
            if guard_local > 0:
                err[:, : min(guard_local, err.shape[1])] = 0.0
            mask = err > self.threshold
            if mask.any():
                labeled, n_blobs = label(mask)
                for blob_id in range(1, n_blobs + 1):
                    ys, xs = np.nonzero(labeled == blob_id)
                    if len(ys) < int(self.cfg["min_blob_px"]):
                        continue
                    peak = float(err[ys, xs].max())
                    ratio = peak / self.threshold
                    score = float(min(ratio / float(self.cfg["score_scale"]), 0.99))
                    ping0, col0 = tile.to_global(int(ys.min()), int(xs.min()))
                    ping1, col1 = tile.to_global(int(ys.max()), int(xs.max()))
                    candidates.append(
                        Detection(
                            side=tile.side,
                            ping0=int(ping0), ping1=int(ping1),
                            col0=int(col0), col1=int(col1),
                            cls=ANOMALY_CLASS, score=score,
                            brain="C", tile_index=tile.index,
                        )
                    )
            if progress is not None:
                progress("anomaly", (i + 1) / n)
        return merge_detections(candidates, float(self.cfg["dedup_iou"]))
