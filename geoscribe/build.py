"""Assemble verified detections into :class:`Contact` records: geotag,
dimensions, severity, evidence rendering, review bookkeeping.

Hard-negative classes (rocks, ripples, reef) are dropped here — they exist
only to absorb detector confusions and are never reported to the operator.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from geoscribe.contact import Contact, Dimensions, PhysicsEvidence, PixelRef
from geoscribe.geotag import bbox_to_geo
from geoscribe.severity import Layer, severity_score
from physicheck.evidence import render_evidence_card
from physicheck.verify import VerifiedDetection
from sonar_core.preprocess.pipeline import PreprocessResult
from tridentnet.classes import is_reportable


def _iso_time(epoch: float) -> str | None:
    if not np.isfinite(epoch):
        return None
    return datetime.fromtimestamp(float(epoch), tz=UTC).isoformat()


def _thumbnail(pre: PreprocessResult, det, out_path: Path, pad: int = 24) -> Path:
    """Small enhanced-imagery crop of the box for tables and map popups."""
    from PIL import Image

    img = pre.ground.side(det.side)
    n_pings, n_cols = img.shape
    r0, r1 = max(det.ping0 - pad, 0), min(det.ping1 + pad, n_pings - 1)
    c0, c1 = max(det.col0 - pad, 0), min(det.col1 + pad, n_cols - 1)
    crop = (np.nan_to_num(img[r0 : r1 + 1, c0 : c1 + 1], nan=0.0) * 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop, mode="L").save(out_path)
    return out_path


def build_contacts(
    verified: list[VerifiedDetection],
    pre: PreprocessResult,
    survey: str,
    layers: list[Layer] | None = None,
    evidence_dir: str | Path | None = None,
    layback_m: float | None = None,
    id_prefix: str = "SN",
) -> list[Contact]:
    """Turn physics-verified detections into reportable contacts.

    Contacts are numbered in confidence order (the input order from
    :func:`physicheck.verify.verify_detections`). ``evidence_dir`` enables
    per-contact evidence cards + thumbnails; without it the paths stay None
    (useful for fast tests and the API's preview pass).
    """
    date_part = "unknown-date"
    times = pre.ground.nav["time"]
    finite_times = times[np.isfinite(times)]
    if finite_times.size:
        date_part = datetime.fromtimestamp(float(finite_times[0]), tz=UTC).strftime("%Y%m%d")

    contacts: list[Contact] = []
    seq = 0
    for v in verified:
        det = v.det
        if not is_reportable(det.cls):
            continue
        seq += 1
        contact_id = f"{id_prefix}-{date_part}-{seq:04d}"

        geo = bbox_to_geo(
            pre.ground, det.side, det.ping0, det.ping1, det.col0, det.col1, layback_m
        )
        centre = (det.ping0 + det.ping1) // 2
        rec = pre.ground.nav[centre]
        depth = None
        if np.isfinite(rec["sensor_depth"]) and np.isfinite(pre.bottom.altitude_m[centre]):
            depth = float(rec["sensor_depth"]) + float(pre.bottom.altitude_m[centre])

        height = None if not np.isfinite(v.analysis.height_m) else float(v.analysis.height_m)
        dims = Dimensions(
            length_m=round(geo.along_m, 2),
            width_m=round(geo.across_m, 2),
            height_m=None if height is None else round(height, 2),
        )
        score, breakdown = severity_score(
            det.cls,
            area_m2=geo.along_m * geo.across_m,
            height_m=height,
            depth_m=depth,
            lat=geo.lat,
            lon=geo.lon,
            layers=layers,
        )

        evidence_png = thumb_png = None
        if evidence_dir is not None:
            evidence_dir = Path(evidence_dir)
            evidence_png = str(
                render_evidence_card(pre, v, evidence_dir / f"{contact_id}_evidence.png")
            )
            thumb_png = str(_thumbnail(pre, det, evidence_dir / f"{contact_id}_thumb.png"))

        contacts.append(
            Contact(
                id=contact_id,
                cls=det.cls,
                confidence=v.confidence_pct,
                brains=sorted(set(getattr(det, "brain", "A"))),
                lat=round(geo.lat, 7),
                lon=round(geo.lon, 7),
                corners=[(round(la, 7), round(lo, 7)) for la, lo in geo.corners],
                dims=dims,
                physics=PhysicsEvidence(
                    highlight=v.analysis.has_highlight,
                    shadow=v.analysis.has_shadow,
                    highlight_ratio=_round_finite(v.analysis.highlight_ratio),
                    shadow_ratio=_round_finite(v.analysis.shadow_ratio),
                    shadow_len_m=_round_finite(v.analysis.shadow_len_m),
                    height_m=None if height is None else round(height, 2),
                    physics_violation=v.gate.violation,
                    violation_reason=v.gate.reason,
                ),
                severity=score,
                severity_breakdown=breakdown,
                pixel=PixelRef(
                    side=det.side,
                    ping0=int(det.ping0),
                    ping1=int(det.ping1),
                    col0=int(det.col0),
                    col1=int(det.col1),
                ),
                depth_m=None if depth is None else round(depth, 1),
                detected_at=_iso_time(float(rec["time"])),
                survey=survey,
                evidence_png=evidence_png,
                thumbnail_png=thumb_png,
            )
        )
    return contacts


def _round_finite(value: float, digits: int = 3) -> float | None:
    return round(float(value), digits) if np.isfinite(value) else None
