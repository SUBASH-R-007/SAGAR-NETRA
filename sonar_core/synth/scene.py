"""Physics-consistent synthetic side-scan scene generator.

Renders a survey the way a real towfish records one: per-ping slant-range
sample vectors with a water-column gap sized by altitude, a bright first
bottom return, grazing-angle range falloff, imperfect-TVG gain banding,
Rayleigh (multiplicative) speckle, sand-ripple texture, and seeded targets
whose acoustic shadows are ray-traced from target height — a shadow ends at
ground range ``x_end = x_far * A / (A - H)``, so the classic slant-range
height estimate ``H = L*A/R`` recovers the seeded height. This makes the
synthetic data usable both to exercise the parser stack and to validate the
PhysiCheck shadow-physics module against known ground truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from sonar_core.parsers.base import NAV_DTYPE, PingArray

EARTH_M_PER_DEG_LAT = 111_320.0


@dataclass
class SynthTarget:
    """Ground truth for one seeded object."""

    cls: str
    side: str  # "port" | "starboard"
    ping: int  # centre ping index
    ground_range: float  # m from nadir to object centre
    length: float  # along-track extent, m
    width: float  # ground-range (across-track) extent, m
    height: float  # proud height above seabed, m
    reflectivity: float = 4.0  # highlight multiplier over background
    natural: bool = False  # hard negative (rock etc.), not man-made
    shape: str = "rect"  # footprint: "rect" | "ellipse" | "irregular"

    def half_width_at(self, dp: float) -> float:
        """Across-track half-extent at normalized along-track offset dp in [-1, 1]."""
        if self.shape == "rect":
            return self.width / 2.0
        return (self.width / 2.0) * float(np.sqrt(max(1.0 - dp * dp, 0.0)))

    def shadow_extent(self, altitude: float) -> tuple[float, float]:
        """(shadow start, shadow end) in ground range, from ray geometry."""
        x_far = self.ground_range + self.width / 2.0
        h = min(self.height, 0.95 * altitude)  # object taller than altitude: cap
        x_end = x_far * altitude / (altitude - h)
        return x_far, x_end

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SceneConfig:
    n_pings: int = 1200
    n_samples: int = 1024  # per side
    slant_range: float = 50.0  # m
    altitude: float = 8.0  # mean towfish height above seabed, m
    altitude_wobble: float = 0.5  # m, slow sinusoidal heave
    sensor_depth: float = 22.0  # m
    sound_velocity: float = 1500.0
    speed: float = 2.0  # m/s over ground
    ping_interval: float = 0.1  # s -> 0.2 m along-track spacing
    start_lat: float = 13.05  # off Chennai / Bay of Bengal
    start_lon: float = 80.35
    heading: float = 90.0  # due east
    mean_level: float = 9000.0  # uint16 counts at mid-range
    noise_floor: float = 250.0  # water-column ambient level
    shadow_level: float = 0.12  # residual reverberation inside shadows
    seed: int = 26057


def default_targets(cfg: SceneConfig) -> list[SynthTarget]:
    """A representative debris field spanning the class map.

    Along-track placement is fractional so any survey length renders every
    target (positions stay proportional for short test surveys).
    """
    def at(frac: float) -> int:
        return int(frac * cfg.n_pings)

    return [
        SynthTarget("wreck", "starboard", at(0.16), 27.0, 24.0, 6.0, 4.2, reflectivity=5.5),
        SynthTarget(
            "ghost_net", "port", at(0.29), 18.0, 6.0, 3.2, 1.3,
            reflectivity=3.2, shape="irregular",
        ),
        SynthTarget(
            "cylinder_drum", "starboard", at(0.43), 33.0, 1.4, 0.9, 0.9,
            reflectivity=6.5, shape="ellipse",
        ),
        SynthTarget(
            "tire", "port", at(0.53), 12.5, 1.1, 1.1, 0.35, reflectivity=3.5, shape="ellipse"
        ),
        SynthTarget("container", "starboard", at(0.67), 21.0, 6.1, 2.4, 2.4, reflectivity=6.0),
        SynthTarget(
            "mine_like", "port", at(0.78), 26.0, 0.8, 0.8, 0.5, reflectivity=7.0, shape="ellipse"
        ),
        SynthTarget(
            "rock_cluster", "starboard", at(0.33), 15.0, 4.0, 3.0, 0.8,
            reflectivity=2.6, natural=True, shape="irregular",
        ),
        SynthTarget(
            "rock_cluster", "port", at(0.88), 30.0, 5.0, 3.5, 1.0,
            reflectivity=2.4, natural=True, shape="irregular",
        ),
    ]


def _hash_noise(ping: int, ground: np.ndarray, salt: int) -> np.ndarray:
    """Deterministic pseudo-noise in [0, 1) per (ping, ground-position) —
    stable across runs so irregular targets render identically for a seed."""
    x = np.sin(ground * 12.9898 + ping * 78.233 + salt * 37.719) * 43758.5453
    return x - np.floor(x)


def _smooth_noise(shape: tuple[int, int], sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean, unit-ish amplitude smooth random field (seabed reflectivity patches)."""
    coarse = rng.standard_normal((max(shape[0] // 32, 4), max(shape[1] // 32, 4)))
    coarse = gaussian_filter(coarse, sigma=sigma)
    fine = zoom(coarse, (shape[0] / coarse.shape[0], shape[1] / coarse.shape[1]), order=1)
    fine = fine[: shape[0], : shape[1]]
    pad = ((0, shape[0] - fine.shape[0]), (0, shape[1] - fine.shape[1]))
    fine = np.pad(fine, pad, mode="edge")
    peak = np.abs(fine).max()
    return fine / peak if peak > 0 else fine


def _make_nav(cfg: SceneConfig, altitudes: np.ndarray) -> np.ndarray:
    nav = np.zeros(cfg.n_pings, dtype=NAV_DTYPE)
    spacing = cfg.speed * cfg.ping_interval
    heading_rad = np.deg2rad(cfg.heading)
    t0 = 1_767_225_600.0  # 2026-01-01T00:00:00Z, arbitrary but stable epoch

    lat = cfg.start_lat
    lon = cfg.start_lon
    for i in range(cfg.n_pings):
        nav[i]["time"] = t0 + i * cfg.ping_interval
        nav[i]["lat"] = lat
        nav[i]["lon"] = lon
        nav[i]["heading"] = cfg.heading
        nav[i]["altitude"] = altitudes[i]
        nav[i]["sensor_depth"] = cfg.sensor_depth
        nav[i]["sound_velocity"] = cfg.sound_velocity
        nav[i]["slant_range"] = cfg.slant_range
        nav[i]["speed"] = cfg.speed
        nav[i]["layback"] = 0.0
        dlat = spacing * np.cos(heading_rad) / EARTH_M_PER_DEG_LAT
        dlon = spacing * np.sin(heading_rad) / (
            EARTH_M_PER_DEG_LAT * np.cos(np.deg2rad(lat))
        )
        lat += dlat
        lon += dlon
    return nav


def make_scene(
    cfg: SceneConfig | None = None,
    targets: list[SynthTarget] | None = None,
) -> tuple[PingArray, list[SynthTarget]]:
    """Render the survey; returns the ping data and the seeded ground truth."""
    cfg = cfg or SceneConfig()
    targets = default_targets(cfg) if targets is None else targets
    rng = np.random.default_rng(cfg.seed)

    ping_idx = np.arange(cfg.n_pings)
    altitudes = (
        cfg.altitude
        + cfg.altitude_wobble * np.sin(2 * np.pi * ping_idx / 300.0)
        + gaussian_filter(rng.standard_normal(cfg.n_pings), sigma=25) * 0.15
    ).astype(np.float32)

    nav = _make_nav(cfg, altitudes)
    slant_res = cfg.slant_range / cfg.n_samples
    slant = (np.arange(cfg.n_samples) + 0.5) * slant_res  # slant range per sample
    along_m = ping_idx * cfg.speed * cfg.ping_interval

    # Large-scale seabed reflectivity patches, one field per side.
    patches = {
        side: 1.0 + 0.22 * _smooth_noise((cfg.n_pings, cfg.n_samples), 2.0, rng)
        for side in ("port", "starboard")
    }

    # Sand-ripple band: along-track metres 60..140 on both sides.
    ripple_lambda = 1.6  # m crest spacing
    ripple_theta = np.deg2rad(15.0)  # crest rotation vs across-track

    sides: dict[str, np.ndarray] = {}
    for side in ("port", "starboard"):
        img = np.empty((cfg.n_pings, cfg.n_samples), dtype=np.float32)
        side_targets = [t for t in targets if t.side == side]
        for i in range(cfg.n_pings):
            alt = float(altitudes[i])
            ground = np.sqrt(np.maximum(slant**2 - alt**2, 0.0))
            in_water = slant < alt

            # Grazing-angle falloff and residual TVG banding (for EGN to remove).
            grazing = np.where(in_water, 0.0, (alt / np.maximum(slant, alt)) ** 0.7)
            tvg_residual = 1.0 + 0.18 * np.sin(2 * np.pi * slant / 17.0) + 0.12 * (
                slant / cfg.slant_range
            )

            reflect = patches[side][i].copy()
            # Ripples live in a band of along-track distance, fade at edges.
            band = np.exp(-(((along_m[i] - 100.0) / 45.0) ** 4))
            if band > 1e-3:
                phase = 2 * np.pi * (
                    ground * np.cos(ripple_theta) + along_m[i] * np.sin(ripple_theta)
                ) / ripple_lambda
                reflect *= 1.0 + 0.30 * band * np.sin(phase)

            # Seeded targets: highlight then ray-traced shadow. The shadow is
            # cast from the footprint's own far edge at THIS ping, so
            # non-rectangular objects get correctly tapering shadows.
            for t in side_targets:
                half_len_pings = max(t.length / (2 * cfg.speed * cfg.ping_interval), 1.0)
                dp = (i - t.ping) / half_len_pings
                if abs(dp) > 1.0:
                    continue
                along_falloff = float(np.cos(dp * np.pi / 2) ** 0.5)
                half_w = t.half_width_at(dp)
                if half_w <= 0:
                    continue
                x0 = t.ground_range - half_w
                x1 = t.ground_range + half_w
                in_obj = (ground >= x0) & (ground <= x1)
                if in_obj.any():
                    # Leading (near-nadir) edge returns hardest.
                    edge = np.clip(1.4 - 0.8 * (ground - x0) / max(2 * half_w, 1e-3), 0.6, 1.4)
                    boost = 1.0 + (t.reflectivity - 1.0) * along_falloff * edge
                    if t.shape == "irregular":
                        # Clumpy texture (net piles, rock clusters): modulate the
                        # highlight with deterministic per-position noise.
                        texture = 0.55 + 0.9 * _hash_noise(i, ground, t.ping)
                        boost = 1.0 + (boost - 1.0) * texture
                    reflect = np.where(in_obj, reflect * boost, reflect)
                h = min(t.height, 0.95 * alt)
                x_end = x1 * alt / (alt - h)
                in_shadow = (ground > x1) & (ground <= x_end)
                if in_shadow.any():
                    shade = 1.0 - (1.0 - cfg.shadow_level) * along_falloff
                    reflect = np.where(in_shadow, reflect * shade, reflect)

            mean = cfg.mean_level * grazing * tvg_residual * reflect
            # First bottom return: bright, slightly range-smeared peak at r ~ alt.
            bottom = 2.6 * cfg.mean_level * np.exp(-(((slant - alt) / (2.2 * slant_res)) ** 2))
            mean = np.where(in_water, cfg.noise_floor, mean + bottom)

            # Rayleigh multiplicative speckle with unit mean.
            speckle = rng.rayleigh(scale=np.sqrt(2 / np.pi), size=cfg.n_samples)
            img[i] = mean * speckle
        sides[side] = np.clip(img, 0, 65535)

    pa = PingArray(
        port=sides["port"],
        starboard=sides["starboard"],
        nav=nav,
        source="synthetic",
        meta={
            "format": "synthetic",
            "sonar_name": "SYNTH-SSS",
            "scene_seed": cfg.seed,
            "slant_resolution": slant_res,
        },
    )
    return pa, targets
