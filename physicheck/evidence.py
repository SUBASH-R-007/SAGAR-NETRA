"""Evidence Cards: one image + cue list per detection, so every contact the
system reports can be audited by a human at a glance.

The card shows the enhanced imagery crop with the detection box, the measured
shadow extent, an optional EigenCAM attention overlay (gradient-free class
activation map on the detector backbone — best-effort, never required), and a
caption with class, calibrated confidence, height and the acoustic cues.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from physicheck.verify import VerifiedDetection
from sonar_core.preprocess.pipeline import PreprocessResult

BOX_COLOR = (255, 210, 40)
SHADOW_COLOR = (255, 80, 80)
TEXT_BG = (12, 20, 30)
TEXT_FG = (235, 240, 245)
CAM_ALPHA = 0.35


def try_eigencam(
    torch_model: Any, target_layers: list[Any], chip_rgb_u8: np.ndarray
) -> np.ndarray | None:
    """Gradient-free EigenCAM heatmap in [0, 1], or None on any failure.

    EigenCAM projects backbone activations onto their first principal
    component, so it needs no gradients, no class targets, and works with the
    YOLO head attached — the robust choice for an offline evidence overlay.
    """
    try:
        import torch
        from pytorch_grad_cam import EigenCAM

        tensor = (
            torch.from_numpy(chip_rgb_u8.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        cam = EigenCAM(model=torch_model, target_layers=target_layers)
        gray = cam(input_tensor=tensor)[0]
        lo, hi = float(gray.min()), float(gray.max())
        return (gray - lo) / (hi - lo) if hi > lo else None
    except Exception:  # noqa: BLE001 - evidence overlay is strictly best-effort
        return None


def _to_u8(img: np.ndarray) -> np.ndarray:
    """Enhanced ground image ([0,1] floats, NaN swath) -> display uint8."""
    return (np.nan_to_num(img, nan=0.0) * 255.0 + 0.5).astype(np.uint8)


def _overlay_cam(base_rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import cm

    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (base_rgb.shape[1], base_rgb.shape[0]), Image.BILINEAR
    )
    heat = cm.get_cmap("jet")(np.asarray(cam_img) / 255.0)[..., :3]
    blended = (1 - CAM_ALPHA) * base_rgb / 255.0 + CAM_ALPHA * heat
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def render_evidence_card(
    pre: PreprocessResult,
    verified: VerifiedDetection,
    out_path: str | Path,
    cam: np.ndarray | None = None,
    pad_pings: int = 40,
    pad_cols: int = 40,
    upscale: int = 2,
) -> Path:
    """Write the evidence PNG (and a sibling ``.json`` cue list); returns PNG path."""
    det = verified.det
    gi = pre.ground
    img = gi.side(det.side)
    n_pings, n_cols = img.shape

    # Crop covers the box, the measured shadow, and context padding.
    shadow_cols = 0
    if np.isfinite(verified.analysis.x_end_m):
        shadow_cols = int(
            np.ceil(gi.col_of_ground_range(verified.analysis.x_end_m)) - det.col1
        )
    r0 = max(det.ping0 - pad_pings, 0)
    r1 = min(det.ping1 + pad_pings, n_pings - 1)
    c0 = max(det.col0 - pad_cols, 0)
    c1 = min(det.col1 + max(shadow_cols, 0) + pad_cols, n_cols - 1)

    crop = _to_u8(img[r0 : r1 + 1, c0 : c1 + 1])
    rgb = np.repeat(crop[:, :, None], 3, axis=2)
    if cam is not None:
        rgb = _overlay_cam(rgb, cam)

    pil = Image.fromarray(rgb).resize(
        (rgb.shape[1] * upscale, rgb.shape[0] * upscale), Image.NEAREST
    )
    draw = ImageDraw.Draw(pil)

    def to_canvas(ping: int, col: int) -> tuple[int, int]:
        return (col - c0) * upscale, (ping - r0) * upscale

    x0, y0 = to_canvas(det.ping0, det.col0)
    x1, y1 = to_canvas(det.ping1 + 1, det.col1 + 1)
    draw.rectangle([x0, y0, x1, y1], outline=BOX_COLOR, width=2)
    if shadow_cols > 0:
        sx0, sy0 = to_canvas(det.ping0, det.col1 + 1)
        sx1, sy1 = to_canvas(det.ping1 + 1, min(det.col1 + shadow_cols, c1) + 1)
        draw.rectangle([sx0, sy0, sx1, sy1], outline=SHADOW_COLOR, width=1)
        draw.line([sx0, (sy0 + sy1) // 2, sx1, (sy0 + sy1) // 2], fill=SHADOW_COLOR, width=1)

    # Caption strip.
    cues = verified.cues()
    height = cues.get("height_m")
    lines = [
        f"{det.cls}  {verified.confidence_pct:.1f}%  [{'+'.join(cues['brains'])}]",
        f"highlight {'yes' if cues['highlight'] else 'NO'}"
        f" ({cues['highlight_ratio']}x) | shadow {'yes' if cues['shadow'] else 'NO'}"
        f" ({cues['shadow_len_m']} m)",
        f"height {'n/a' if height is None else f'{height} m'}"
        + (f" | VIOLATION: {cues['violation_reason']}" if cues["physics_violation"] else ""),
    ]
    strip_h = 16 * len(lines) + 10
    canvas = Image.new("RGB", (pil.width, pil.height + strip_h), TEXT_BG)
    canvas.paste(pil, (0, 0))
    caption = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        caption.text((6, pil.height + 5 + 16 * i), line, fill=TEXT_FG)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    out_path.with_suffix(".json").write_text(json.dumps(cues, indent=2), encoding="utf-8")
    return out_path
