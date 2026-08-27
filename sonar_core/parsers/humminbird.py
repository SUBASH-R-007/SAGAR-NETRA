"""Humminbird .DAT + .SON/.IDX adapter: citizen-sonar recordings (N-10/L1).

A Humminbird recording is a ``RecNN.DAT`` metadata file plus a sibling
directory of the same stem holding one ``B00x.SON`` byte stream (and optional
``B00x.IDX`` index) per beam. Sidescan beams are ``B002`` (port) and ``B003``
(starboard); :meth:`HumminbirdParser.parse` takes the ``.DAT`` path and finds
the directory itself. Structures follow the reverse-engineered documentation
of the PING-Mapper project (and its ancestor PyHum); where references
disagree the PING-Mapper interpretation is used, as noted below.

.DAT header (9xx/11xx/Helix variant, big endian; offsets in bytes)::

    0   uint8   spacer
    1   uint8   water code (0 fresh, 1 deep salt, 2 shallow salt)
    4   uint32  sonar name / firmware code
    20  uint32  UTC epoch seconds at start of recording
    24  int32   easting  (mercator metres, see below)
    28  int32   northing
    32  10s     recording name
    44  uint32  number of records
    48  uint32  recording length, milliseconds
    52  uint32  line size (unknown/zero on some models)

.SON record: the 4-byte head marker ``C0 DE AB 21``, then tag-prefixed
attributes (one tag byte, big-endian payload) closed by the end tag ``0x21``,
then ``ping_cnt`` uint8 backscatter samples (sample 0 at the transducer, i.e.
nadir-first — never mirrored here). Tags decoded (payload size in bytes)::

    0x80 (4) record number         0x81 (4) time since start, ms
    0x82 (4) easting               0x83 (4) northing
    0x84 (4) GPS flag u16 + heading u16, tenths of a degree
    0x85 (4) GPS flag u16 + speed u16, cm/s
    0x87 (4) depth below transducer, cm
    0x50 (1) beam number           0x51 (1) volt scale, tenths of a volt
    0x92 (4) frequency, Hz         0xA0 (4) ping byte count
    0x53/0x54/0x56/0x57 (1) and 0x86/0x95 (4) undocumented

The 67-byte Helix header is written by :func:`write_humminbird`; the parser
walks tags generically so longer model variants (72/152-byte headers) decode
too. The ``.IDX`` sidecar is pairs of big-endian uint32 ``(time_ms,
byte_offset)`` per record; without it the parser scans for the head marker.

Ambiguities resolved per PING-Mapper's published file-structure notes:
depth and speed are centimetre-scaled (÷100 to metres and m/s; PyHum's docs
say the same but divide by 10), and positions are spherical-mercator metres
inverted with radius ``R = 6378388 m`` plus PyHum/PING-Mapper's ellipsoid
latitude correction ``lat = atan(1.0067642927 * tan(2*atan(exp(n/R)) -
pi/2))``, ``lon = deg(e/R)``.

The operator's range setting is *not* stored in the SON header: like
PING-Mapper, slant range is reconstructed as ``ping_cnt * pixel_size_m``
where ``pixel_size_m`` is the receiver's per-sample slant footprint
(default 0.02 m ~ c/(2*Fs) for a Helix sidescan receiver) — a tunable kwarg.
Boat-mounted transducer at the surface: height above seabed equals the
recorded depth (``altitude = depth``), ``sensor_depth ~ 0``.

Validation mirrors :mod:`sonar_core.parsers.jsf`: tests round-trip
``parse(write_humminbird(pa))`` against our own spec-following writer.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from sonar_core.parsers.base import (
    NAV_DTYPE,
    ParserError,
    PingArray,
    SonarParser,
    register_parser,
)

#: Sphere radius of Humminbird's mercator integers (PyHum/PING-Mapper).
HUMMINBIRD_RADIUS_M = 6378388.0
#: Sphere-to-WGS84 latitude stretch used by PyHum/PING-Mapper.
LAT_CORRECTION = 1.0067642927
HEAD_START = b"\xc0\xde\xab\x21"
HEAD_END_TAG = 0x21
DAT_MIN_SIZE = 56
#: Per-sample slant footprint of a Helix sidescan receiver, metres
#: (~ c / (2 * Fs) with c = 1500 m/s, Fs ~ 37.5 kHz effective sampling).
DEFAULT_PIXEL_SIZE_M = 0.02
#: Sidescan beam file stems per PING-Mapper (B000/B001 are down-looking).
SIDESCAN_BEAMS: dict[str, str] = {"port": "B002", "starboard": "B003"}

#: SON attribute tags -> payload size in bytes. Unknown tags abort parsing
#: (a wrong guess would silently misalign every following field).
_TAG_SIZES: dict[int, int] = {
    0x80: 4, 0x81: 4, 0x82: 4, 0x83: 4, 0x84: 4, 0x85: 4, 0x86: 4, 0x87: 4,
    0x50: 1, 0x51: 1, 0x92: 4, 0x53: 1, 0x54: 1, 0x95: 4, 0x56: 1, 0x57: 1,
    0xA0: 4,
}
_SIGNED_TAGS = frozenset({0x82, 0x83})  # mercator metres can be negative


def _decode_dat(buf: bytes, name: str) -> dict[str, Any]:
    """Decode the .DAT survey header (big endian, Helix layout)."""
    if len(buf) < DAT_MIN_SIZE:
        raise ParserError(f"{name}: .DAT too short ({len(buf)} bytes, need >= {DAT_MIN_SIZE})")
    return {
        "water_code": buf[1],
        "sonar_name": struct.unpack_from(">I", buf, 4)[0],
        "unix_time": struct.unpack_from(">I", buf, 20)[0],
        "utm_e": struct.unpack_from(">i", buf, 24)[0],
        "utm_n": struct.unpack_from(">i", buf, 28)[0],
        "recording_name": buf[32:42].split(b"\x00", 1)[0].decode("ascii", "replace"),
        "numrecords": struct.unpack_from(">I", buf, 44)[0],
        "recordlens_ms": struct.unpack_from(">I", buf, 48)[0],
        "linesize": struct.unpack_from(">I", buf, 52)[0],
    }


def _decode_record(buf: bytes, pos: int, name: str) -> tuple[dict[str, Any], int]:
    """Decode one SON record at *pos*; return ``(fields, next_offset)``."""
    if buf[pos : pos + 4] != HEAD_START:
        raise ParserError(f"{name}: no record head marker at byte {pos}")
    fields: dict[str, int] = {}
    p = pos + 4
    while True:
        if p >= len(buf):
            raise ParserError(f"{name}: truncated record header at byte {pos}")
        tag = buf[p]
        if tag == HEAD_END_TAG:
            p += 1
            break
        size = _TAG_SIZES.get(tag)
        if size is None or p + 1 + size > len(buf):
            raise ParserError(f"{name}: unknown/truncated SON tag 0x{tag:02x} at byte {p}")
        fields[tag] = int.from_bytes(buf[p + 1 : p + 1 + size], "big", signed=tag in _SIGNED_TAGS)
        p += 1 + size
    ping_cnt = fields.get(0xA0)
    if ping_cnt is None or p + ping_cnt > len(buf):
        raise ParserError(f"{name}: record at byte {pos} lacks a valid ping count")
    record = {
        "time_ms": fields.get(0x81, 0),
        "utm_e": fields.get(0x82, 0),
        "utm_n": fields.get(0x83, 0),
        "heading_ddeg": fields.get(0x84, 0) & 0xFFFF,  # low u16 of flag+heading word
        "speed_cms": fields.get(0x85, 0) & 0xFFFF,  # low u16 of flag+speed word
        "depth_cm": fields.get(0x87, 0),
        "beam": fields.get(0x50, 0),
        "frequency": fields.get(0x92, 0),
        "data": np.frombuffer(buf, dtype=np.uint8, count=ping_cnt, offset=p),
    }
    return record, p + ping_cnt


def _read_son(son_path: Path) -> list[dict[str, Any]]:
    """All records of one beam, via the .IDX index or head-marker scan."""
    buf = son_path.read_bytes()
    records: list[dict[str, Any]] = []
    idx_path = son_path.with_suffix(".IDX")
    if not idx_path.exists():
        idx_path = son_path.with_suffix(".idx")
    if idx_path.exists():
        pairs = np.frombuffer(idx_path.read_bytes(), dtype=">u4")
        for offset in pairs.reshape(-1, 2)[:, 1]:
            rec, _ = _decode_record(buf, int(offset), son_path.name)
            records.append(rec)
    else:
        pos = buf.find(HEAD_START)
        while pos >= 0:
            rec, end = _decode_record(buf, pos, son_path.name)
            records.append(rec)
            pos = buf.find(HEAD_START, end)
    if not records:
        raise ParserError(f"{son_path.name}: no SON records found (corrupt or empty)")
    return records


def _find_beam_file(rec_dir: Path, stem: str) -> Path | None:
    """Case-insensitive lookup of e.g. ``B002.SON`` inside the recording dir."""
    for candidate in rec_dir.iterdir():
        if candidate.stem.upper() == stem and candidate.suffix.upper() == ".SON":
            return candidate
    return None


def _mercator_to_lonlat(e: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PyHum/PING-Mapper inverse: sphere mercator -> WGS-84 degrees."""
    lon = np.degrees(e / HUMMINBIRD_RADIUS_M)
    gd = 2.0 * np.arctan(np.exp(n / HUMMINBIRD_RADIUS_M)) - np.pi / 2.0
    lat = np.degrees(np.arctan(LAT_CORRECTION * np.tan(gd)))
    return lon, lat


def _lonlat_to_mercator(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact inverse of :func:`_mercator_to_lonlat`, rounded to integer metres."""
    e = np.rint(HUMMINBIRD_RADIUS_M * np.radians(lon))
    gd = np.arctan(np.tan(np.radians(lat)) / LAT_CORRECTION)
    n = np.rint(HUMMINBIRD_RADIUS_M * np.log(np.tan(np.pi / 4.0 + gd / 2.0)))
    return e, n


@register_parser
class HumminbirdParser(SonarParser):
    """Adapter for Humminbird recordings, addressed by their ``.DAT`` file."""

    suffixes = (".dat",)

    def parse(
        self,
        path: Path,
        pixel_size_m: float = DEFAULT_PIXEL_SIZE_M,
        sound_velocity: float = 1500.0,
        **kwargs: Any,
    ) -> PingArray:
        """Read the ``.DAT`` at *path* plus its sibling recording directory.

        Parameters
        ----------
        pixel_size_m:
            Slant-range metres per SON sample; the range setting is not in
            the file, so per-ping slant range is ``ping_cnt * pixel_size_m``
            (PING-Mapper's reconstruction — see module docstring).
        sound_velocity:
            Assumed water sound velocity, m/s (not recorded by the unit).
        """
        dat = _decode_dat(path.read_bytes(), path.name)
        rec_dir = path.with_suffix("")
        if not rec_dir.is_dir():
            raise ParserError(f"{path.name}: recording directory {rec_dir.name!r} not found")

        beams: dict[str, list[dict[str, Any]]] = {}
        for side, stem in SIDESCAN_BEAMS.items():
            son = _find_beam_file(rec_dir, stem)
            if son is None:
                raise ParserError(
                    f"{path.name}: sidescan beam {stem}.SON missing in {rec_dir.name!r}"
                )
            beams[side] = _read_son(son)

        n_pings = min(len(beams["port"]), len(beams["starboard"]))
        sides: dict[str, np.ndarray] = {}
        for side in SIDESCAN_BEAMS:
            rows = [r["data"] for r in beams[side][:n_pings]]
            width = int(np.bincount([len(r) for r in rows]).argmax())
            img = np.zeros((n_pings, width), dtype=np.float32)
            for i, row in enumerate(rows):
                img[i, : min(len(row), width)] = row[:width]
            sides[side] = img

        # Nav from the port beam: both sides are pinged by the same head at
        # the same instant, so either is valid; port is the deterministic pick.
        ref = beams["port"][:n_pings]
        e = np.array([r["utm_e"] for r in ref], dtype=np.float64)
        n = np.array([r["utm_n"] for r in ref], dtype=np.float64)
        lon, lat = _mercator_to_lonlat(e, n)
        cnt = np.array([len(r["data"]) for r in ref], dtype=np.float64)

        nav = np.zeros(n_pings, dtype=NAV_DTYPE)
        nav["time"] = dat["unix_time"] + np.array([r["time_ms"] for r in ref]) / 1000.0
        nav["lat"] = lat
        nav["lon"] = lon
        nav["heading"] = np.array([r["heading_ddeg"] for r in ref]) / 10.0
        # Boat-mounted transducer at the surface: height above seabed is the
        # measured water depth; the sensor itself sits at ~0 m depth.
        nav["altitude"] = np.array([r["depth_cm"] for r in ref]) / 100.0
        nav["sensor_depth"] = 0.0
        nav["sound_velocity"] = float(sound_velocity)
        nav["slant_range"] = cnt * float(pixel_size_m)
        nav["speed"] = np.array([r["speed_cms"] for r in ref]) / 100.0

        meta = {
            "format": "humminbird",
            "sonar_name": f"Humminbird-{dat['sonar_name']}",
            "water_code": int(dat["water_code"]),
            "recording_name": dat["recording_name"],
            "numrecords": int(dat["numrecords"]),
            "recordlens_ms": int(dat["recordlens_ms"]),
            "beams": dict(SIDESCAN_BEAMS),
            "pixel_size_m": float(pixel_size_m),
        }
        return PingArray(
            port=sides["port"], starboard=sides["starboard"], nav=nav,
            source=str(path), meta=meta,
        )


def _pack_record(
    rec: np.void, i: int, e: int, n: int, beam: int, data: np.ndarray, t0: float
) -> bytes:
    """One 67-byte Helix SON record header + ping bytes (writer helper)."""
    head = bytearray()
    head += HEAD_START
    head += bytes([0x80]) + struct.pack(">I", i)
    head += bytes([0x81]) + struct.pack(">I", max(round((float(rec["time"]) - t0) * 1000.0), 0))
    head += bytes([0x82]) + struct.pack(">i", e)
    head += bytes([0x83]) + struct.pack(">i", n)
    head += bytes([0x84]) + struct.pack(">HH", 0, round(float(rec["heading"]) * 10.0) % 3600)
    head += bytes([0x85]) + struct.pack(">HH", 0, round(float(rec["speed"]) * 100.0))
    head += bytes([0x87]) + struct.pack(">I", round(float(rec["altitude"]) * 100.0))
    head += bytes([0x50, beam, 0x51, 0])
    head += bytes([0x92]) + struct.pack(">I", 455_000)
    head += bytes([0x53, 0, 0x54, 0])
    head += bytes([0x95]) + struct.pack(">I", 0)
    head += bytes([0x56, 0, 0x57, 0])
    head += bytes([0xA0]) + struct.pack(">I", len(data))
    head += bytes([HEAD_END_TAG])
    return bytes(head) + data.tobytes()


def write_humminbird(pa: PingArray, dat_path: str | Path) -> Path:
    """Serialize *pa* as a Humminbird recording (round-trip test writer).

    Mirrors the jsf.py round-trip-vs-own-writer approach: writes the 64-byte
    Helix ``.DAT``, the ``B002.SON``/``B003.SON`` sidescan beams (67-byte
    record headers, nadir-first uint8 pings — no mirroring), and their
    ``.IDX`` sidecars, into ``dat_path`` plus its sibling directory.
    """
    dat_path = Path(dat_path)
    rec_dir = dat_path.with_suffix("")
    rec_dir.mkdir(parents=True, exist_ok=True)

    t0 = float(np.floor(pa.nav["time"][0])) if pa.n_pings else 0.0
    e, n = _lonlat_to_mercator(
        pa.nav["lon"].astype(np.float64), pa.nav["lat"].astype(np.float64)
    )
    duration_ms = round((float(pa.nav["time"][-1]) - t0) * 1000.0) if pa.n_pings else 0

    dat = bytearray(64)
    dat[0] = 0xC1  # spacer (not validated by readers)
    dat[1] = 0  # water code: fresh
    struct.pack_into(">I", dat, 4, 1199)  # sonar name/firmware code
    struct.pack_into(">I", dat, 20, int(t0))
    struct.pack_into(">i", dat, 24, int(e[0]) if pa.n_pings else 0)
    struct.pack_into(">i", dat, 28, int(n[0]) if pa.n_pings else 0)
    dat[32:42] = dat_path.stem.encode("ascii", "replace")[:10].ljust(10, b"\x00")
    struct.pack_into(">I", dat, 44, pa.n_pings)
    struct.pack_into(">I", dat, 48, max(duration_ms, 0))
    struct.pack_into(">I", dat, 52, 0)  # line size: unknown on Helix
    dat_path.write_bytes(bytes(dat))

    beam_no = {"port": 2, "starboard": 3}
    for side, stem in SIDESCAN_BEAMS.items():
        son_chunks: list[bytes] = []
        idx = bytearray()
        offset = 0
        for i in range(pa.n_pings):
            rec = pa.nav[i]
            data = np.clip(np.rint(pa.side(side)[i]), 0, 255).astype(np.uint8)
            chunk = _pack_record(rec, i, int(e[i]), int(n[i]), beam_no[side], data, t0)
            son_chunks.append(chunk)
            time_ms = max(round((float(rec["time"]) - t0) * 1000.0), 0)
            idx += struct.pack(">II", time_ms, offset)
            offset += len(chunk)
        (rec_dir / f"{stem}.SON").write_bytes(b"".join(son_chunks))
        (rec_dir / f"{stem}.IDX").write_bytes(bytes(idx))
    return dat_path
