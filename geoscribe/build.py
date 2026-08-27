"""Assemble verified detections into :class:`Contact` records: geotag,
dimensions, severity, evidence rendering, review bookkeeping.

Hard-negative classes (rocks, ripples, reef) are dropped here — they exist
only to absorb detector confusions and are never reported to the operator.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pyproj import Geod

from geoscribe.contact import Contact, Dimensions, PhysicsEvidence, PixelRef
from geoscribe.geotag import bbox_to_geo
from geoscribe.report import priority_for, recommended_action_for
from geoscribe.severity import Layer, severity_score
from physicheck.evidence import render_evidence_card
from physicheck.verify import VerifiedDetection
from sonar_core.preprocess.pipeline import PreprocessResult
from tridentnet.classes import is_reportable

_GEOD = Geod(ellps="WGS84")

#: Where the reporting tunables live; mirrors the keys read below.
GEOSCRIBE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "geoscribe.yaml"

#: Fallback surface-GPS fix error when the config file or key is absent.
DEFAULT_NAV_UNCERTAINTY_M = 2.0


def nav_fix_uncertainty_m(config_path: str | Path = GEOSCRIBE_CONFIG) -> float:
    """Surface GPS antenna fix error charged to every contact position.

    Read from ``position_accuracy.nav_fix_uncertainty_m`` in
    ``configs/geoscribe.yaml``; a missing file or key falls back to
    :data:`DEFAULT_NAV_UNCERTAINTY_M` (2.0 m, typical DGPS horizontal error)
    so a stripped-down deployment still reports an honest budget.
    """
    path = Path(config_path)
    if path.exists():
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        value = (doc.get("position_accuracy") or {}).get("nav_fix_uncertainty_m")
        if value is not None:
            return float(value)
    return DEFAULT_NAV_UNCERTAINTY_M


def position_accuracy(
    ground_res: float,
    layback_known: bool,
    nav_uncertainty_m: float = DEFAULT_NAV_UNCERTAINTY_M,
    layback_uncertainty_m: float = 5.0,
) -> float:
    """Honest per-contact position error budget, metres::

        accuracy = 2 * ground_res + layback_term + nav_uncertainty_m

    * ``2 * ground_res`` — pixel picking and slant-to-ground resampling can
      each be off by about one ground column;
    * ``layback_term`` — 0 when the towfish offset astern of the antenna is
      known (recorded in nav or supplied by the operator), else
      ``layback_uncertainty_m``: an unmodelled cable layback shifts the whole
      swath along-track by the full missing offset;
    * ``nav_uncertainty_m`` — the surface GPS fix error itself
      (``configs/geoscribe.yaml``, default 2.0 m).

    Terms add linearly rather than in quadrature on purpose: each is a bias,
    not independent noise, and a recovery diver wants the conservative number.
    """
    layback_term = 0.0 if layback_known else float(layback_uncertainty_m)
    return 2.0 * float(ground_res) + layback_term + float(nav_uncertainty_m)


def survey_stats(pre: PreprocessResult) -> dict[str, Any]:
    """Coverage and sonar-geometry numbers for the contacts.json summary block.

    Track length is the geodesic sum over consecutive finite nav fixes. The
    usable swath is the mean count of finite (in-swath, non-blanked) ground
    columns per ping, in metres, summed over both sides — NaN gaps beyond a
    ping's own swath therefore shrink the area honestly instead of counting
    the full image width. Area = track length x usable swath.
    """
    gi = pre.ground
    lats = np.asarray(gi.nav["lat"], dtype=np.float64)
    lons = np.asarray(gi.nav["lon"], dtype=np.float64)
    finite = np.isfinite(lats) & np.isfinite(lons)
    track_m = 0.0
    if int(finite.sum()) >= 2:
        la, lo = lats[finite], lons[finite]
        _, _, dists = _GEOD.inv(lo[:-1], la[:-1], lo[1:], la[1:])
        track_m = float(np.sum(dists))

    swath_m = 0.0  # both sides: port usable width + starboard usable width
    for side in ("port", "starboard"):
        img = gi.side(side)
        if img.size:
            swath_m += float(np.isfinite(img).sum(axis=1).mean()) * gi.ground_res

    slant = np.asarray(gi.nav["slant_range"], dtype=np.float64)
    slant = slant[np.isfinite(slant) & (slant > 0)]
    alt = np.asarray(gi.altitude_m, dtype=np.float64)
    alt = alt[np.isfinite(alt)]
    return {
        "track_length_km": round(track_m / 1e3, 3),
        "swath_width_m": round(swath_m, 1),
        "area_surveyed_sqkm": round(track_m * swath_m / 1e6, 4),
        "range_m": None if slant.size == 0 else round(float(np.median(slant)), 1),
        "altitude_m": None if alt.size == 0 else round(float(np.mean(alt)), 1),
        "n_pings": int(gi.n_pings),
    }


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
    hazard_table: dict[str, float] | None = None,
    nav_uncertainty_m: float | None = None,
    layback_uncertainty_m: float = 5.0,
) -> list[Contact]:
    """Turn physics-verified detections into reportable contacts.

    Contacts are numbered in confidence order (the input order from
    :func:`physicheck.verify.verify_detections`). ``evidence_dir`` enables
    per-contact evidence cards + thumbnails; without it the paths stay None
    (useful for fast tests and the API's preview pass). ``hazard_table``
    passes a mission profile's per-class hazard weights straight through to
    :func:`geoscribe.severity.severity_score` (None keeps the defaults), so
    disaster-mode profiles re-rank contacts without touching detection or
    physics evidence.

    Triage fields (priority / recommended action) derive from the severity
    bands in :mod:`geoscribe.report` — a single threshold table. The position
    accuracy budget uses :func:`position_accuracy`; ``nav_uncertainty_m``
    None reads ``configs/geoscribe.yaml`` (default 2.0 m) and
    ``layback_uncertainty_m`` is charged only for pings whose towfish layback
    is unrecorded (NaN) with no operator override.
    """
    date_part = "unknown-date"
    times = pre.ground.nav["time"]
    finite_times = times[np.isfinite(times)]
    if finite_times.size:
        date_part = datetime.fromtimestamp(float(finite_times[0]), tz=UTC).strftime("%Y%m%d")

    if nav_uncertainty_m is None:
        nav_uncertainty_m = nav_fix_uncertainty_m()

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
            hazard_table=hazard_table,
        )

        layback_known = layback_m is not None or bool(np.isfinite(rec["layback"]))
        accuracy = position_accuracy(
            pre.ground.ground_res,
            layback_known,
            nav_uncertainty_m=nav_uncertainty_m,
            layback_uncertainty_m=layback_uncertainty_m,
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
                priority=priority_for(score),
                recommended_action=recommended_action_for(det.cls, score),
                position_accuracy_m=round(accuracy, 2),
            )
        )
    return contacts


def _round_finite(value: float, digits: int = 3) -> float | None:
    return round(float(value), digits) if np.isfinite(value) else None
