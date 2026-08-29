"""Side-scan beam geometry: resolution, sound-speed error and multipath range.

The rest of the pipeline works in pixels and metres that the *data* already
carries — recorded slant range, tracked altitude, ground resolution. This
module holds the relationships that depend on the **sensor** rather than on
any one file: how wide a resolution cell is, how fast it degrades with range,
how much a sound-speed error moves a contact, and where a second bottom return
lands.

Why these matter operationally, rather than as physics trivia:

* **Across-track resolution is constant** at ``c*tau/2`` — set by pulse length
  alone, the same at 10 m and at 70 m. Nothing across-track gets blurrier with
  range.
* **Along-track resolution is not.** It is ``theta * R``, growing linearly, so
  the same object is smeared over three times as many pings at 75 m as at
  25 m. A reported ``length_m`` at far range is therefore a much softer number
  than the same figure near nadir, and a report that does not say so is
  overstating what the sonar measured.
* **Sound speed enters as a scale error.** Range is ``c*t/2``, so a 1% error in
  assumed ``c`` is a 1% error in every range — three quarters of a metre at
  75 m, which matters to a diver working from the coordinates.
* **Multipath repeats the seabed.** The second bottom return arrives at twice
  the altitude in slant range, which is ``A*sqrt(3)`` in ground range; a
  "target" sitting there is usually the seabed being heard twice.

Configuration lives in ``configs/sonar.yaml`` because these are properties of
the towfish, not of the survey, and a different sonar changes all of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "sonar.yaml"

#: Nominal seawater sound speed, used only when a file records none.
DEFAULT_SOUND_VELOCITY = 1500.0


@dataclass(frozen=True)
class SonarGeometry:
    """Towfish beam and pulse parameters.

    Defaults describe a typical 455 kHz survey sonar: a 0.5-degree along-track
    beam and a 100-microsecond pulse, which give 7.5 cm across-track resolution
    and 0.22 m along-track at 25 m range.
    """

    along_track_beam_deg: float = 0.5
    across_track_beam_deg: float = 50.0
    pulse_length_s: float = 1.0e-4
    #: Fractional uncertainty in assumed sound speed. 1% is the usual figure
    #: for an uncompensated survey in water with an unmeasured thermocline.
    sound_velocity_uncertainty_frac: float = 0.01
    #: Half-width of the band around the second bottom return treated as
    #: multipath-suspect, as a fraction of that range.
    multipath_tolerance_frac: float = 0.15

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> SonarGeometry:
        """Read ``configs/sonar.yaml``; missing file or keys fall back to defaults."""
        config = Path(path)
        if not config.exists():
            return cls()
        doc = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        section = doc.get("geometry") or {}
        known = {f: section[f] for f in cls.__dataclass_fields__ if f in section}
        return cls(**{k: float(v) for k, v in known.items()})


def slant_range_from_time(
    two_way_time_s: float, sound_velocity_mps: float = DEFAULT_SOUND_VELOCITY
) -> float:
    """Slant range from two-way travel time: ``R = c*t/2``.

    The halving is the whole point — the pulse travels to the seabed and back,
    so the echo's time of flight covers twice the distance being measured.
    Formats that record a sampling interval rather than a range (JSF) derive
    their range exactly this way.
    """
    return 0.5 * float(sound_velocity_mps) * float(two_way_time_s)


def across_track_resolution_m(
    sound_velocity_mps: float = DEFAULT_SOUND_VELOCITY,
    pulse_length_s: float = 1.0e-4,
) -> float:
    """Across-track (range) resolution ``c*tau/2`` — constant across the swath.

    Two targets closer together than half the pulse length in range return
    overlapping echoes and cannot be separated. Because it depends only on the
    transmitted pulse, this figure does *not* degrade with range: the limit at
    the far edge of the swath is the same as at nadir.
    """
    return 0.5 * float(sound_velocity_mps) * float(pulse_length_s)


def along_track_resolution_m(beam_width_deg: float, range_m: float) -> float:
    """Along-track resolution ``theta * R`` — degrades linearly with range.

    The beam is an angular wedge, so its footprint along the track widens the
    further it travels. This is the dominant reason far-range contacts are
    smeared along-track and why their reported lengths deserve a wider error
    bar than near-range ones.
    """
    return math.radians(float(beam_width_deg)) * max(float(range_m), 0.0)


def sound_speed_range_error_m(range_m: float, uncertainty_frac: float = 0.01) -> float:
    """Range error from an imperfectly known sound speed.

    Range is proportional to ``c``, so a fractional error in the assumed sound
    speed produces the same fractional error in range: at 75 m, 1% is 0.75 m.
    """
    return abs(float(range_m)) * abs(float(uncertainty_frac))


def multipath_ground_range_m(altitude_m: float, order: int = 2) -> float:
    """Ground range at which the *n*-th bottom return lands.

    The first bottom return arrives at slant range equal to the altitude ``A``.
    A pulse that bounces seabed-surface-seabed before returning arrives at
    ``order * A`` in slant range, which converts to ground range through the
    same right triangle the slant correction uses::

        g = sqrt((order*A)^2 - A^2) = A * sqrt(order^2 - 1)

    For the usual second return that is ``A*sqrt(3)`` — about 1.73 altitudes
    out. Returns 0.0 for a non-physical order or altitude.
    """
    a = float(altitude_m)
    if a <= 0 or order < 2:
        return 0.0
    return a * math.sqrt(float(order) ** 2 - 1.0)


def is_multipath_candidate(
    ground_range_m: float,
    altitude_m: float,
    tolerance_frac: float = 0.15,
    order: int = 2,
) -> bool:
    """Whether a contact sits in the band where a repeated seabed return lands.

    This is a *suspicion*, not a verdict: real debris does sometimes lie at
    1.73 altitudes, and flagging is left to inform an analyst rather than to
    delete anything. Returns False when the geometry is unknown, so missing
    altitude can never manufacture a flag.
    """
    expected = multipath_ground_range_m(altitude_m, order)
    if expected <= 0 or not np.isfinite(ground_range_m):
        return False
    return abs(float(ground_range_m) - expected) <= abs(tolerance_frac) * expected


def resolution_cell_m(
    range_m: float,
    beam_width_deg: float,
    sound_velocity_mps: float = DEFAULT_SOUND_VELOCITY,
    pulse_length_s: float = 1.0e-4,
) -> tuple[float, float]:
    """``(along_track_m, across_track_m)`` footprint of one resolution cell.

    The pair is the honest floor on what the sonar can resolve at that range:
    anything smaller than this cell is a single bright sample, not a measured
    shape, whatever its bounding box says.
    """
    return (
        along_track_resolution_m(beam_width_deg, range_m),
        across_track_resolution_m(sound_velocity_mps, pulse_length_s),
    )
