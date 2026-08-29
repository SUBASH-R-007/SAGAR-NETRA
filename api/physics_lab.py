"""Interactive physics for the console's Physics Lab.

Three things a visitor can drive from the browser, all of them computed by the
*same functions that process real surveys* rather than a JavaScript
re-derivation. That distinction is the point: a slider here is not an
illustration of what the pipeline does, it is a call into it. Re-implementing
the formulas in the frontend would let the picture and the product drift apart,
and the picture would always look right.

* :func:`geometry_report` — resolution, multipath range and sound-speed error
  for a chosen altitude and range, straight out of :mod:`sonar_core.geometry`.
* :func:`shadow_round_trip` — the forward shadow model and its inversion. An
  object of height ``H`` casts a shadow ending at ``x_end``; inverting that with
  the deployed estimator must return ``H``. Showing both directions is what
  makes "height from shadow" believable rather than asserted.
* :func:`simulate_scene` — render a real synthetic scene from user-placed
  targets, preprocess it exactly as an uploaded survey is preprocessed, then
  measure each target's height from its shadow and report it against the truth
  the renderer used. This is the honest version of a demo: the measurement can
  disagree with the truth, and when it does the visitor sees that.
"""

from __future__ import annotations

import base64
import io
import math
from typing import Any

import numpy as np
from PIL import Image

from physicheck.shadow import analyze_shadow
from sonar_core.geometry import (
    DEFAULT_SOUND_VELOCITY,
    SonarGeometry,
    across_track_resolution_m,
    along_track_resolution_m,
    is_multipath_candidate,
    multipath_ground_range_m,
    sound_speed_range_error_m,
)
from sonar_core.preprocess.pipeline import preprocess
from sonar_core.synth.scene import SceneConfig, SynthTarget, make_scene
from tridentnet.data import CLASS_SPECS, _target_bbox

#: Guard rails for the scene simulator. It runs inside a request, so the work
#: has to stay bounded no matter what a caller asks for.
MAX_PINGS = 700
MAX_SAMPLES = 768
MAX_TARGETS = 8
MAX_HEIGHT_FRAC = 0.9  # an object taller than the towfish casts no usable shadow


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(min(max(float(value), lo), hi))


def _pick(mapping: dict[str, Any], key: str, default: float) -> float:
    """``mapping[key]`` unless it is missing *or explicitly None*.

    Pydantic's ``model_dump`` writes optional fields out as ``None`` rather
    than omitting them, so ``dict.get(key, default)`` returns ``None`` and the
    default never applies. Anything a caller left blank must fall back to the
    class's own typical size.
    """
    value = mapping.get(key)
    return float(default) if value is None else float(value)


def geometry_report(
    altitude_m: float,
    range_m: float,
    *,
    beam_deg: float | None = None,
    pulse_us: float | None = None,
    sound_velocity_mps: float | None = None,
) -> dict[str, Any]:
    """Every range-dependent limit for one towfish configuration.

    Overrides are accepted so the lab can show what a *different* sonar would
    buy — a narrower beam, a shorter pulse — against the shipped configuration
    in ``configs/sonar.yaml``.
    """
    sonar = SonarGeometry.load()
    beam = float(beam_deg) if beam_deg is not None else sonar.along_track_beam_deg
    pulse = (
        float(pulse_us) * 1e-6 if pulse_us is not None else sonar.pulse_length_s
    )
    sv = (
        float(sound_velocity_mps)
        if sound_velocity_mps is not None
        else DEFAULT_SOUND_VELOCITY
    )
    altitude = max(float(altitude_m), 0.0)
    slant = max(float(range_m), 0.0)

    # Ground range reachable at this slant range: the swath starts at the first
    # bottom return (nadir) and ends where the geometry runs out.
    max_ground = math.sqrt(max(slant**2 - altitude**2, 0.0))

    # Sample the along-track limit across the swath so the frontend can draw
    # the curve rather than two endpoints.
    steps = 24
    curve = []
    for i in range(steps + 1):
        ground = max_ground * i / steps
        slant_at = math.hypot(ground, altitude)
        curve.append(
            {
                "ground_range_m": round(ground, 2),
                "slant_range_m": round(slant_at, 2),
                "along_track_m": round(along_track_resolution_m(beam, slant_at), 4),
                "across_track_m": round(across_track_resolution_m(sv, pulse), 4),
            }
        )

    return {
        "altitude_m": round(altitude, 2),
        "slant_range_m": round(slant, 2),
        "max_ground_range_m": round(max_ground, 2),
        "beam_deg": beam,
        "pulse_us": round(pulse * 1e6, 3),
        "sound_velocity_mps": round(sv, 1),
        "across_track_resolution_m": round(across_track_resolution_m(sv, pulse), 4),
        "along_track_resolution_near_m": round(
            along_track_resolution_m(beam, max(altitude, 1e-6)), 4
        ),
        "along_track_resolution_far_m": round(along_track_resolution_m(beam, slant), 4),
        "multipath_ground_range_m": round(multipath_ground_range_m(altitude), 2),
        "sound_speed_error_far_m": round(
            sound_speed_range_error_m(slant, sonar.sound_velocity_uncertainty_frac), 3
        ),
        "sound_velocity_uncertainty_frac": sonar.sound_velocity_uncertainty_frac,
        "curve": curve,
        # Stated so the panel can say which numbers came from the shipped
        # config and which the visitor overrode.
        "defaults": {
            "beam_deg": sonar.along_track_beam_deg,
            "pulse_us": round(sonar.pulse_length_s * 1e6, 3),
        },
    }


def shadow_round_trip(
    altitude_m: float, height_m: float, ground_range_m: float
) -> dict[str, Any]:
    """Forward-model a shadow, then invert it with the deployed estimator.

    Forward: an object of height ``H`` whose far edge sits at ground range
    ``x_far`` blocks the beam out to ``x_end = x_far * A / (A - H)``.
    Inverse: ``H = A * (x_end - x_far) / x_end`` — the form
    :mod:`physicheck.shadow` uses on real imagery.

    The two must agree. Returning both, plus the geometry needed to draw the
    ray diagram, is what turns "we get height from shadow" into something a
    visitor can check rather than take on faith.
    """
    altitude = max(float(altitude_m), 0.1)
    x_far = max(float(ground_range_m), 0.1)
    # An object at or above the towfish has no shadow solution; clamp rather
    # than divide by zero, and say so.
    h_cap = MAX_HEIGHT_FRAC * altitude
    height = _clamp(height_m, 0.0, h_cap)
    clamped = abs(float(height_m) - height) > 1e-9

    if height <= 0.0:
        x_end = x_far
        shadow_len = 0.0
        recovered = 0.0
    else:
        x_end = x_far * altitude / (altitude - height)
        shadow_len = x_end - x_far
        recovered = altitude * (x_end - x_far) / x_end

    return {
        "altitude_m": round(altitude, 3),
        "height_m": round(height, 3),
        "height_clamped": clamped,
        "max_height_m": round(h_cap, 3),
        "ground_range_m": round(x_far, 3),
        "shadow_start_m": round(x_far, 3),
        "shadow_end_m": round(x_end, 3),
        "shadow_length_m": round(shadow_len, 3),
        "recovered_height_m": round(recovered, 3),
        # Should be ~0: a visible residual means the inversion is wrong, which
        # is exactly what this panel exists to make checkable.
        "round_trip_error_m": round(abs(recovered - height), 6),
        "grazing_angle_deg": round(math.degrees(math.atan2(altitude, x_far)), 2),
        "shadow_gain": round(shadow_len / height, 2) if height > 0 else None,
    }


def _to_png_b64(image: np.ndarray) -> str:
    """8-bit PNG data URI payload for a float image in [0, 1]."""
    arr = (np.clip(np.nan_to_num(image, nan=0.0), 0.0, 1.0) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def available_classes() -> list[dict[str, Any]]:
    """Placeable target classes with the size ranges the renderer accepts."""
    out = []
    for name, spec in CLASS_SPECS.items():
        out.append(
            {
                "cls": name,
                "natural": bool(spec.natural),
                "length_m": list(spec.length_m),
                "width_m": list(spec.width_m),
                "height_m": list(spec.height_m),
                "reflectivity": list(spec.reflectivity),
            }
        )
    return sorted(out, key=lambda d: (d["natural"], d["cls"]))


def simulate_scene(
    targets: list[dict[str, Any]],
    *,
    altitude_m: float = 8.0,
    slant_range_m: float = 50.0,
    n_pings: int = 400,
    n_samples: int = 512,
    seed: int = 26057,
) -> dict[str, Any]:
    """Render a scene from placed targets and measure each one from its shadow.

    The full L1 chain runs — bottom tracking, EGN, slant correction, despeckle,
    CLAHE — exactly as it does for an uploaded survey, so the returned waterfall
    is the real thing rather than a drawing. Heights are then measured from the
    shadows by :func:`physicheck.shadow.analyze_shadow` and reported next to the
    truth the renderer was given.

    No detector runs: this panel is about whether the *physics* recovers what
    was put there, and inserting a model between the two would make a
    disagreement ambiguous. Detection has its own tabs.
    """
    cfg = SceneConfig(
        n_pings=int(_clamp(n_pings, 120, MAX_PINGS)),
        n_samples=int(_clamp(n_samples, 256, MAX_SAMPLES)),
        slant_range=float(_clamp(slant_range_m, 20.0, 120.0)),
        altitude=float(_clamp(altitude_m, 3.0, 30.0)),
        seed=int(seed),
    )
    max_ground = math.sqrt(max(cfg.slant_range**2 - cfg.altitude**2, 0.0))

    placed: list[SynthTarget] = []
    for raw in list(targets)[:MAX_TARGETS]:
        cls = str(raw.get("cls", "cylinder_drum"))
        spec = CLASS_SPECS.get(cls)
        if spec is None:
            continue
        # Keep every target inside the usable swath and below the towfish, so
        # a placement that cannot physically be imaged is corrected rather than
        # rendered as a puzzle.
        ground = _clamp(
            _pick(raw, "ground_range_m", max_ground * 0.5),
            2.0, max(max_ground - 1.0, 3.0),
        )
        height = _clamp(
            _pick(raw, "height_m", sum(spec.height_m) / 2),
            0.05, MAX_HEIGHT_FRAC * cfg.altitude,
        )
        length = _clamp(_pick(raw, "length_m", sum(spec.length_m) / 2), 0.5, 40.0)
        width = _clamp(_pick(raw, "width_m", sum(spec.width_m) / 2), 0.3, 20.0)
        ping = int(_clamp(_pick(raw, "ping", cfg.n_pings // 2), 20, cfg.n_pings - 20))
        side = "port" if str(raw.get("side") or "starboard") == "port" else "starboard"
        placed.append(
            SynthTarget(
                cls=cls, side=side, ping=ping, ground_range=ground,
                length=length, width=width, height=height,
                reflectivity=float(
                    raw.get("reflectivity", sum(spec.reflectivity) / 2)
                ),
                natural=bool(spec.natural),
                shape=spec.shapes[0],
            )
        )

    pa, rendered = make_scene(cfg, placed)
    pre = preprocess(pa)
    gi = pre.ground

    measured: list[dict[str, Any]] = []
    for target in rendered:
        bbox = _target_bbox(gi, target, cfg, shadow_pad_cols=3)
        if bbox is None:
            continue
        r0, r1, c0, c1 = bbox
        analysis = analyze_shadow(
            pre.ground_raw, target.side,
            int(r0), max(int(r1) - 1, int(r0)),
            int(c0), max(int(c1) - 1, int(c0)),
        )
        est = None if not np.isfinite(analysis.height_m) else float(analysis.height_m)
        slant_at = math.hypot(target.ground_range, cfg.altitude)
        measured.append(
            {
                "cls": target.cls,
                "side": target.side,
                "natural": bool(target.natural),
                "ground_range_m": round(float(target.ground_range), 2),
                "truth_height_m": round(float(target.height), 2),
                "measured_height_m": None if est is None else round(est, 2),
                "height_error_m": None if est is None else round(est - target.height, 2),
                "shadow_len_m": round(float(analysis.shadow_len_m), 2),
                "has_highlight": bool(analysis.has_highlight),
                "has_shadow": bool(analysis.has_shadow),
                "along_track_resolution_m": round(
                    along_track_resolution_m(
                        SonarGeometry.load().along_track_beam_deg, slant_at
                    ),
                    3,
                ),
                "multipath_suspect": is_multipath_candidate(
                    float(target.ground_range), cfg.altitude,
                    SonarGeometry.load().multipath_tolerance_frac,
                ),
                "pixel": {
                    "ping0": int(r0), "ping1": max(int(r1) - 1, int(r0)),
                    "col0": int(c0), "col1": max(int(c1) - 1, int(c0)),
                },
            }
        )

    port = np.nan_to_num(gi.port, nan=0.0)[:, ::-1]
    stbd = np.nan_to_num(gi.starboard, nan=0.0)
    combined = np.hstack([port, stbd])

    return {
        "waterfall_png_b64": _to_png_b64(combined),
        "n_pings": int(gi.n_pings),
        "n_port_cols": int(gi.n_cols("port")),
        "n_stbd_cols": int(gi.n_cols("starboard")),
        "ground_res": round(float(gi.ground_res), 5),
        "altitude_m": cfg.altitude,
        "slant_range_m": cfg.slant_range,
        "max_ground_range_m": round(max_ground, 2),
        "seed": cfg.seed,
        "targets": measured,
        "geometry": geometry_report(cfg.altitude, cfg.slant_range),
    }
