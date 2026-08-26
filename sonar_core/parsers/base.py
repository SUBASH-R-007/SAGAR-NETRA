"""Common ping-data model and the parser adapter registry.

Every supported input format (XTF, JSF, GeoTIFF, plain waterfall images) is
converted to a :class:`PingArray` before any processing happens, so layers
L2-L5 are format-agnostic. New formats plug in by subclassing
:class:`SonarParser` and decorating with :func:`register_parser`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

#: One navigation/attitude record per ping. Kept as a structured array so
#: corrections can be vectorized and never assume constancy along-track.
NAV_DTYPE = np.dtype(
    [
        ("time", np.float64),  # UTC epoch seconds
        ("lat", np.float64),  # WGS-84 degrees, +N
        ("lon", np.float64),  # WGS-84 degrees, +E
        ("heading", np.float32),  # degrees clockwise from true north
        ("altitude", np.float32),  # towfish height above seabed, metres
        ("sensor_depth", np.float32),  # towfish depth below surface, metres
        ("sound_velocity", np.float32),  # two-way-corrected water SV, m/s
        ("slant_range", np.float32),  # max slant range per side this ping, metres
        ("speed", np.float32),  # speed over ground, m/s
        ("layback", np.float32),  # towfish distance astern of nav antenna, metres
    ]
)

KNOTS_TO_MS = 0.514444


class ParserError(RuntimeError):
    """Raised when a file cannot be interpreted by any registered adapter."""


@dataclass
class PingArray:
    """Raw side-scan intensities plus per-ping navigation, in slant range.

    ``port``/``starboard`` are ``(n_pings, n_samples)`` float32 arrays of raw
    backscatter amplitude, sample 0 at nadir (zero slant range), increasing
    outward. ``nav`` is a ``(n_pings,)`` array with :data:`NAV_DTYPE` records.
    """

    port: np.ndarray
    starboard: np.ndarray
    nav: np.ndarray
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.port = np.ascontiguousarray(self.port, dtype=np.float32)
        self.starboard = np.ascontiguousarray(self.starboard, dtype=np.float32)
        if self.port.ndim != 2 or self.starboard.ndim != 2:
            raise ValueError("port/starboard must be 2-D (n_pings, n_samples)")
        if self.port.shape[0] != self.starboard.shape[0]:
            raise ValueError(
                f"ping-count mismatch: port {self.port.shape[0]} vs "
                f"starboard {self.starboard.shape[0]}"
            )
        self.nav = np.asarray(self.nav)
        if self.nav.dtype != NAV_DTYPE:
            raise ValueError(f"nav must have NAV_DTYPE, got {self.nav.dtype}")
        if self.nav.shape != (self.port.shape[0],):
            raise ValueError(
                f"nav must have one record per ping: {self.nav.shape} vs {self.port.shape[0]} pings"
            )

    # -- convenience ------------------------------------------------------

    @property
    def n_pings(self) -> int:
        return self.port.shape[0]

    def side(self, name: str) -> np.ndarray:
        if name not in ("port", "starboard"):
            raise KeyError(f"side must be 'port' or 'starboard', got {name!r}")
        return getattr(self, name)

    def n_samples(self, side: str) -> int:
        return self.side(side).shape[1]

    def slant_resolution(self, side: str) -> float:
        """Metres of slant range per sample (median over pings)."""
        n = self.n_samples(side)
        if n == 0:
            return 0.0
        return float(np.median(self.nav["slant_range"])) / n

    def sample_to_slant(self, side: str, sample_idx: np.ndarray | int) -> np.ndarray:
        """Slant range (m) at the *centre* of the given sample index."""
        res = self.slant_resolution(side)
        return (np.asarray(sample_idx, dtype=np.float64) + 0.5) * res

    def slice_pings(self, start: int, stop: int) -> PingArray:
        """Sub-survey view over a ping interval (data is not copied)."""
        return PingArray(
            port=self.port[start:stop],
            starboard=self.starboard[start:stop],
            nav=self.nav[start:stop],
            source=self.source,
            meta={**self.meta, "ping_offset": self.meta.get("ping_offset", 0) + start},
        )


class SonarParser(ABC):
    """Adapter base: turns one on-disk format into a :class:`PingArray`."""

    #: lowercase filename suffixes this adapter claims, e.g. ``(".xtf",)``
    suffixes: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return path.suffix.lower() in cls.suffixes

    @abstractmethod
    def parse(self, path: Path, **kwargs: Any) -> PingArray:
        """Read *path* and return its contents as a :class:`PingArray`."""


_REGISTRY: list[type[SonarParser]] = []


def register_parser(cls: type[SonarParser]) -> type[SonarParser]:
    _REGISTRY.append(cls)
    return cls


def _ensure_adapters_loaded() -> None:
    """Import adapter modules so their ``@register_parser`` decorators run."""
    import importlib

    for mod in ("xtf", "jsf", "image", "geotiff"):
        try:
            importlib.import_module(f"sonar_core.parsers.{mod}")
        except ImportError:  # optional deps (e.g. rasterio) may be absent
            continue


def load(path: str | Path, **kwargs: Any) -> PingArray:
    """Parse *path* with the first registered adapter that claims it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    _ensure_adapters_loaded()
    for parser_cls in _REGISTRY:
        if parser_cls.can_parse(path):
            return parser_cls().parse(path, **kwargs)
    raise ParserError(
        f"no parser for {path.name!r}; known suffixes: "
        f"{sorted(s for p in _REGISTRY for s in p.suffixes)}"
    )
