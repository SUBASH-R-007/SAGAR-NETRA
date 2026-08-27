"""L1 preprocessing: bottom tracking, slant-range correction, EGN, despeckle,
CLAHE, and SAHI-style tiling — pure NumPy/SciPy functions plus one pipeline
orchestrator with progress callbacks.

Pipeline order (slant-domain steps first, then ground-domain steps):

    PingArray -> track_bottom -> egn (slant, water-column aware)
             -> blank water column -> slant_to_ground -> despeckle -> clahe
             -> tile

Every stage is optional and config-driven; see ``configs/preprocess.yaml``.
"""

from sonar_core.preprocess.bottom_track import BottomTrack, blank_water_column, track_bottom
from sonar_core.preprocess.slant_range import GroundImage, slant_to_ground

__all__ = [
    "BottomTrack",
    "GroundImage",
    "blank_water_column",
    "slant_to_ground",
    "track_bottom",
]
