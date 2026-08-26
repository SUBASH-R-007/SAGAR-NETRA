"""Entanglement Severity Index (0–100).

A weighted, saturating blend of: class hazard, footprint area, height above
seabed, depth band (shallow objects entangle gear and hit hulls), and
geodesic proximity to sensitive GeoJSON layers (shipping lanes, turtle
nesting zones, marine protected areas). Every term is exposed in a
breakdown so operators can see *why* a contact ranks high.

No shapely dependency: point-in-polygon runs on lon/lat ray casting and
point-to-edge distances are computed on a local azimuthal-equidistant
projection centred at the contact (exact enough below a few hundred km).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyproj import Proj

from geoscribe.contact import SeverityBreakdown

DEFAULT_HAZARD: dict[str, float] = {
    "ghost_net": 1.0,
    "human_body": 1.0,
    "mine_like": 0.95,
    "container": 0.80,
    "cylinder_drum": 0.75,
    "wreck": 0.60,
    "aircraft": 0.60,
    "pipeline": 0.50,
    "tire": 0.30,
    "unknown_anomaly": 0.65,
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "hazard": 0.40,
    "size": 0.15,
    "height": 0.10,
    "depth": 0.15,
    "proximity": 0.20,
}


@dataclass
class Layer:
    """One sensitive-area GeoJSON layer."""

    name: str
    weight: float  # 0..1 importance multiplier
    features: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_geojson(cls, path: str | Path, weight: float = 1.0, name: str | None = None) -> Layer:
        path = Path(path)
        doc = json.loads(path.read_text(encoding="utf-8"))
        features = doc.get("features", [doc] if doc.get("type") == "Feature" else [])
        return cls(name=name or path.stem, weight=weight, features=features)


def _rings_of(geometry: dict[str, Any]) -> list[list[tuple[float, float]]]:
    """All coordinate rings/lines of a geometry as [(lon, lat), ...] lists."""
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon":
        return [[(c[0], c[1]) for c in ring] for ring in coords]
    if gtype == "MultiPolygon":
        return [[(c[0], c[1]) for c in ring] for poly in coords for ring in poly]
    if gtype == "LineString":
        return [[(c[0], c[1]) for c in coords]]
    if gtype == "MultiLineString":
        return [[(c[0], c[1]) for c in line] for line in coords]
    return []


def _point_in_polygon(lon: float, lat: float, geometry: dict[str, Any]) -> bool:
    if geometry.get("type") not in ("Polygon", "MultiPolygon"):
        return False
    polygons = (
        [geometry["coordinates"]]
        if geometry["type"] == "Polygon"
        else geometry["coordinates"]
    )
    for poly in polygons:
        if not poly:
            continue
        outer = poly[0]
        inside = False
        j = len(outer) - 1
        for i in range(len(outer)):
            xi, yi = outer[i][0], outer[i][1]
            xj, yj = outer[j][0], outer[j][1]
            if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
                inside = not inside
            j = i
        if inside:
            return True
    return False


def _distance_to_geometry_m(lon: float, lat: float, geometry: dict[str, Any]) -> float:
    """Geodesic-grade distance (m) from a point to a geometry's edges; 0 inside."""
    if _point_in_polygon(lon, lat, geometry):
        return 0.0
    proj = Proj(proj="aeqd", lat_0=lat, lon_0=lon, ellps="WGS84")
    best = math.inf
    for ring in _rings_of(geometry):
        if not ring:
            continue
        xs, ys = proj([c[0] for c in ring], [c[1] for c in ring])
        for i in range(len(xs) - 1):
            best = min(best, _point_segment_dist(0.0, 0.0, xs[i], ys[i], xs[i + 1], ys[i + 1]))
        if len(xs) == 1:
            best = min(best, math.hypot(xs[0], ys[0]))
    return best


def _point_segment_dist(px: float, py: float, x0: float, y0: float, x1: float, y1: float) -> float:
    dx, dy = x1 - x0, y1 - y0
    seg2 = dx * dx + dy * dy
    if seg2 == 0:
        return math.hypot(px - x0, py - y0)
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / seg2))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def severity_score(
    cls: str,
    area_m2: float,
    height_m: float | None,
    depth_m: float | None,
    lat: float,
    lon: float,
    layers: list[Layer] | None = None,
    hazard_table: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
    area_scale_m2: float = 40.0,
    height_scale_m: float = 2.0,
    shallow_m: float = 20.0,
    deep_m: float = 100.0,
    proximity_decay_m: float = 2000.0,
) -> tuple[float, SeverityBreakdown]:
    """Score one contact; returns (0-100 score, per-term breakdown)."""
    hazard_table = hazard_table or DEFAULT_HAZARD
    weights = weights or DEFAULT_WEIGHTS

    hazard = hazard_table.get(cls, hazard_table.get("unknown_anomaly", 0.5))
    size = 1.0 - math.exp(-max(area_m2, 0.0) / area_scale_m2)
    height = 0.3 if height_m is None else 1.0 - math.exp(-max(height_m, 0.0) / height_scale_m)

    if depth_m is None or not math.isfinite(depth_m):
        depth = 0.5
    elif depth_m <= shallow_m:
        depth = 1.0
    elif depth_m >= deep_m:
        depth = 0.2
    else:
        depth = 1.0 - 0.8 * (depth_m - shallow_m) / (deep_m - shallow_m)

    proximity = 0.0
    nearest_name: str | None = None
    nearest_dist: float | None = None
    for layer in layers or []:
        for feature in layer.features:
            geometry = feature.get("geometry") or {}
            dist = _distance_to_geometry_m(lon, lat, geometry)
            contribution = layer.weight * math.exp(-dist / proximity_decay_m)
            if contribution > proximity:
                proximity = contribution
                nearest_name = str(
                    feature.get("properties", {}).get("name", layer.name)
                )
                nearest_dist = dist

    total = (
        weights["hazard"] * hazard
        + weights["size"] * size
        + weights["height"] * height
        + weights["depth"] * depth
        + weights["proximity"] * proximity
    ) / sum(weights.values())
    score = round(100.0 * min(max(total, 0.0), 1.0), 1)

    breakdown = SeverityBreakdown(
        hazard=round(hazard, 3),
        size=round(size, 3),
        height=round(height, 3),
        depth=round(depth, 3),
        proximity=round(proximity, 3),
        nearest_layer=nearest_name,
        nearest_layer_distance_m=None if nearest_dist is None else round(nearest_dist, 1),
    )
    return score, breakdown


def load_layers(layer_dir: str | Path, config: dict[str, float] | None = None) -> list[Layer]:
    """Load every ``*.geojson`` in *layer_dir*; weights from *config* by stem."""
    layer_dir = Path(layer_dir)
    config = config or {}
    layers = []
    if layer_dir.is_dir():
        for path in sorted(layer_dir.glob("*.geojson")):
            layers.append(Layer.from_geojson(path, weight=float(config.get(path.stem, 1.0))))
    return layers
