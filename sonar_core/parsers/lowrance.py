"""Lowrance .sl2/.sl3 adapter: citizen-sonar sidescan logs (blueprint N-10/L1).

Consumer Lowrance chartplotters log every enabled channel as a chain of
variable-size frames after an 8-byte file header. Field offsets follow the
community reverse-engineered specification maintained by the ``opensounder``
project (``sounder-log-formats``, formats 2 and 3) as implemented by the
``sonarlight`` and ``SL3Reader`` readers.

File header (8 bytes, little endian)::

    0  uint16  format   (1 = slg, 2 = sl2, 3 = sl3)
    2  uint16  version
    4  uint16  blocksize (nominal; frames carry their own size)
    6  uint16  reserved

SL2 frame header (144 bytes), offsets used here::

    28  uint16  frame size (header + sounding bytes -> next frame)
    32  uint16  channel (0 primary .. 5 sidescan composite)
    34  uint16  packet size (sounding bytes)
    36  uint32  frame index
    40  float32 upper range limit, feet
    44  float32 lower range limit, feet
    64  float32 water depth, feet
    100 float32 GPS speed, knots
    108 int32   x (Lowrance spherical-mercator easting, metres)
    112 int32   y (Lowrance spherical-mercator northing, metres)
    120 float32 course over ground, radians
    128 float32 heading, radians
    140 uint32  time, milliseconds (device-relative; no absolute epoch)

SL3 frame header (168 bytes for channels 0-5) — same physics, shuffled
offsets; the format word in the file header switches the offset table::

    8   uint16  frame size
    12  uint16  channel
    16  uint32  frame index
    20  float32 upper range limit, feet
    24  float32 lower range limit, feet
    40  uint32  UTC epoch seconds (valid in the first frame of the file)
    44  uint16  packet size
    48  float32 water depth, feet
    84  float32 GPS speed, knots
    92  int32   x
    96  int32   y
    104 float32 course over ground, radians
    112 float32 heading, radians
    124 uint32  time since file creation, milliseconds

Positions use Lowrance's spherical mercator on the WGS-84 *polar* radius
``R = 6356752.3142 m``: ``lon = deg(x / R)`` and
``lat = deg(2*atan(exp(y / R)) - pi/2)``.

The sidescan composite channel (5) stores each swath port-far -> nadir ->
starboard-far in one byte vector; the port half is reversed here so both
sides are nadir-first per the :class:`PingArray` convention. Lowrance units
are transom-mounted: the transducer rides at the surface, so its height
above the seabed *is* the water depth (``altitude = depth``) and
``sensor_depth ~ 0``. Slant range per side is the lower range limit.

Validation mirrors :mod:`sonar_core.parsers.jsf`: :func:`write_slx` emits
frames from the same offset tables so tests round-trip parse(write(pa))
against a known survey instead of depending on proprietary sample files.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from sonar_core.parsers.base import (
    KNOTS_TO_MS,
    NAV_DTYPE,
    ParserError,
    PingArray,
    SonarParser,
    register_parser,
)

FEET_TO_M = 0.3048
#: WGS-84 polar radius used by Lowrance's spherical mercator encoding.
LOWRANCE_RADIUS_M = 6356752.3142
FILE_HEADER_SIZE = 8
FORMAT_SL2 = 2
FORMAT_SL3 = 3
CHANNEL_PRIMARY = 0
CHANNEL_SIDESCAN = 5

#: Frame header size per format word (SL3 sizes apply to channels 0-5).
FRAME_HEADER_SIZE: dict[int, int] = {FORMAT_SL2: 144, FORMAT_SL3: 168}

#: Byte offset of each decoded field inside a frame, per format word.
_OFFSETS: dict[int, dict[str, int]] = {
    FORMAT_SL2: {
        "frame_size": 28, "channel": 32, "packet_size": 34, "frame_index": 36,
        "upper_limit": 40, "lower_limit": 44, "water_depth": 64, "gps_speed": 100,
        "x": 108, "y": 112, "course": 120, "heading": 128, "time_ms": 140,
    },
    FORMAT_SL3: {
        "frame_size": 8, "channel": 12, "packet_size": 44, "frame_index": 16,
        "upper_limit": 20, "lower_limit": 24, "water_depth": 48, "gps_speed": 84,
        "x": 92, "y": 96, "course": 104, "heading": 112, "time_ms": 124,
        "epoch": 40,
    },
}


def _parse_file_header(buf: bytes, name: str) -> tuple[int, int, int]:
    """Return ``(format, version, blocksize)`` or raise on a non-Lowrance file."""
    if len(buf) < FILE_HEADER_SIZE:
        raise ParserError(f"{name}: too short for a Lowrance header ({len(buf)} bytes)")
    fmt, version, blocksize, _reserved = struct.unpack_from("<HHHH", buf, 0)
    if fmt not in (FORMAT_SL2, FORMAT_SL3):
        raise ParserError(f"{name}: unsupported Lowrance format word {fmt} (expected 2 or 3)")
    return fmt, version, blocksize


def _iter_frames(buf: bytes, fmt: int, name: str):
    """Yield one decoded field-dict (plus raw sounding bytes) per frame."""
    off = _OFFSETS[fmt]
    hdr = FRAME_HEADER_SIZE[fmt]
    pos = FILE_HEADER_SIZE
    while pos + hdr <= len(buf):
        (frame_size,) = struct.unpack_from("<H", buf, pos + off["frame_size"])
        (packet,) = struct.unpack_from("<H", buf, pos + off["packet_size"])
        if frame_size < hdr or pos + frame_size > len(buf) or hdr + packet > frame_size:
            raise ParserError(
                f"{name}: corrupt frame at byte {pos}: "
                f"frame_size={frame_size}, packet={packet}"
            )
        (channel,) = struct.unpack_from("<H", buf, pos + off["channel"])
        frame: dict[str, Any] = {
            "channel": int(channel),
            "index": struct.unpack_from("<I", buf, pos + off["frame_index"])[0],
            "upper_m": struct.unpack_from("<f", buf, pos + off["upper_limit"])[0] * FEET_TO_M,
            "lower_m": struct.unpack_from("<f", buf, pos + off["lower_limit"])[0] * FEET_TO_M,
            "depth_m": struct.unpack_from("<f", buf, pos + off["water_depth"])[0] * FEET_TO_M,
            "speed_ms": struct.unpack_from("<f", buf, pos + off["gps_speed"])[0] * KNOTS_TO_MS,
            "x": struct.unpack_from("<i", buf, pos + off["x"])[0],
            "y": struct.unpack_from("<i", buf, pos + off["y"])[0],
            "course_rad": struct.unpack_from("<f", buf, pos + off["course"])[0],
            "heading_rad": struct.unpack_from("<f", buf, pos + off["heading"])[0],
            "time_ms": struct.unpack_from("<I", buf, pos + off["time_ms"])[0],
            "sounding": np.frombuffer(buf, dtype=np.uint8, count=packet, offset=pos + hdr),
        }
        if fmt == FORMAT_SL3:
            frame["epoch"] = struct.unpack_from("<I", buf, pos + off["epoch"])[0]
        yield frame
        pos += frame_size


def _modal_width(lengths: list[int]) -> int:
    """Most common sounding length — rare short/garbled frames are padded."""
    return int(np.bincount(lengths).argmax()) if lengths else 0


@register_parser
class LowranceParser(SonarParser):
    """Adapter for Lowrance .sl2/.sl3 sidescan-composite logs."""

    suffixes = (".sl2", ".sl3")

    def parse(self, path: Path, **kwargs: Any) -> PingArray:
        buf = path.read_bytes()
        fmt, version, blocksize = _parse_file_header(buf, path.name)
        frames = list(_iter_frames(buf, fmt, path.name))

        kept = [f for f in frames if f["channel"] == CHANNEL_SIDESCAN]
        channel_note = "sidescan"
        if not kept:
            # No sidescan composite recorded: fall back to the primary
            # down-looking echo so depth/nav QC can still run. The single
            # trace is duplicated to both sides (no split is meaningful for
            # a down-looking beam) and flagged in meta.
            kept = [f for f in frames if f["channel"] == CHANNEL_PRIMARY]
            channel_note = "primary"
        if not kept:
            raise ParserError(f"{path.name}: no sidescan (5) or primary (0) frames found")
        kept.sort(key=lambda f: f["index"])
        n_pings = len(kept)

        if channel_note == "sidescan":
            # Composite runs port-far -> nadir -> starboard-far: split at the
            # midpoint and reverse the port half to nadir-first order.
            port_rows = [f["sounding"][: len(f["sounding"]) // 2][::-1] for f in kept]
            stbd_rows = [f["sounding"][len(f["sounding"]) // 2 :] for f in kept]
        else:
            port_rows = [f["sounding"] for f in kept]
            stbd_rows = port_rows

        n_port = _modal_width([len(r) for r in port_rows])
        n_stbd = _modal_width([len(r) for r in stbd_rows])
        port = np.zeros((n_pings, n_port), dtype=np.float32)
        stbd = np.zeros((n_pings, n_stbd), dtype=np.float32)
        for i, (p_row, s_row) in enumerate(zip(port_rows, stbd_rows, strict=True)):
            port[i, : min(len(p_row), n_port)] = p_row[:n_port]
            stbd[i, : min(len(s_row), n_stbd)] = s_row[:n_stbd]

        # Vectorized nav assembly + unit conversions.
        y = np.array([f["y"] for f in kept], dtype=np.float64)
        x = np.array([f["x"] for f in kept], dtype=np.float64)
        hdg = np.array([f["heading_rad"] for f in kept], dtype=np.float64)
        cog = np.array([f["course_rad"] for f in kept], dtype=np.float64)
        time_ms = np.array([f["time_ms"] for f in kept], dtype=np.float64)
        # SL3 frame 0 carries the absolute epoch; SL2 time stays device-relative.
        base = float(frames[0].get("epoch", 0)) if fmt == FORMAT_SL3 else 0.0

        nav = np.zeros(n_pings, dtype=NAV_DTYPE)
        nav["time"] = base + time_ms / 1000.0
        nav["lat"] = np.degrees(2.0 * np.arctan(np.exp(y / LOWRANCE_RADIUS_M)) - np.pi / 2.0)
        nav["lon"] = np.degrees(x / LOWRANCE_RADIUS_M)
        # A magnetic heading of exactly 0.0 usually means "no compass fitted";
        # course over ground is the honest along-track direction then.
        nav["heading"] = np.degrees(np.where(hdg != 0.0, hdg, cog)) % 360.0
        # Transom mount: the transducer rides at the surface, so height above
        # the seabed equals the water depth and the sensor itself is at ~0 m.
        nav["altitude"] = [f["depth_m"] for f in kept]
        nav["sensor_depth"] = 0.0
        nav["sound_velocity"] = 1500.0  # not recorded in sl2/sl3
        nav["slant_range"] = [f["lower_m"] for f in kept]
        nav["speed"] = [f["speed_ms"] for f in kept]

        meta = {
            "format": "sl2" if fmt == FORMAT_SL2 else "sl3",
            "sonar_name": "Lowrance",
            "version": int(version),
            "blocksize": int(blocksize),
            "channel": channel_note,
            "time_reference": "epoch" if fmt == FORMAT_SL3 else "device-relative",
        }
        if channel_note == "primary":
            meta["note"] = (
                "no sidescan composite frames; primary down-looking channel "
                "duplicated to both sides"
            )
        return PingArray(port=port, starboard=stbd, nav=nav, source=str(path), meta=meta)


def write_slx(
    pa: PingArray,
    path: str | Path,
    fmt: int | None = None,
    channel: int = CHANNEL_SIDESCAN,
) -> Path:
    """Serialize *pa* as an .sl2/.sl3 log (round-trip validation writer).

    Mirrors the jsf.py approach: tests validate the parser against this
    spec-following writer rather than proprietary sample recordings. *fmt*
    (2 or 3) defaults to the path suffix. ``channel=CHANNEL_SIDESCAN`` writes
    one composite frame per ping (port reversed to far-first, then starboard),
    exercising the parser's split-and-reverse; ``channel=CHANNEL_PRIMARY``
    writes the starboard trace as a down-looking primary channel to exercise
    the fallback. Intensities are rounded into the uint8 sounding range.
    """
    path = Path(path)
    if fmt is None:
        fmt = FORMAT_SL3 if path.suffix.lower() == ".sl3" else FORMAT_SL2
    if fmt not in (FORMAT_SL2, FORMAT_SL3):
        raise ValueError(f"fmt must be 2 (sl2) or 3 (sl3), got {fmt}")
    off = _OFFSETS[fmt]
    hdr = FRAME_HEADER_SIZE[fmt]
    path.parent.mkdir(parents=True, exist_ok=True)

    lat_rad = np.radians(pa.nav["lat"].astype(np.float64))
    x_int = np.rint(LOWRANCE_RADIUS_M * np.radians(pa.nav["lon"].astype(np.float64)))
    y_int = np.rint(LOWRANCE_RADIUS_M * np.log(np.tan(np.pi / 4.0 + lat_rad / 2.0)))
    heading_rad = np.radians(pa.nav["heading"].astype(np.float64))
    base = float(np.floor(pa.nav["time"][0])) if pa.n_pings else 0.0
    # SL2 has no epoch field: hardware writes device-relative milliseconds.
    # Small (device-relative) times serialize verbatim; absolute epoch times
    # (~1.7e12 ms, overflowing the uint32 field) are rebased to the first
    # ping, matching what a real device would have recorded.
    sl2_base = 0.0
    if fmt == FORMAT_SL2 and pa.n_pings and float(pa.nav["time"].max()) * 1000.0 > 0xFFFFFFFF:
        sl2_base = base

    with path.open("wb") as fh:
        fh.write(struct.pack("<HHHH", fmt, 1, 3200, 0))
        pos = FILE_HEADER_SIZE
        for i in range(pa.n_pings):
            rec = pa.nav[i]
            if channel == CHANNEL_SIDESCAN:
                port = np.clip(np.rint(pa.port[i]), 0, 255).astype(np.uint8)
                stbd = np.clip(np.rint(pa.starboard[i]), 0, 255).astype(np.uint8)
                sounding = np.concatenate([port[::-1], stbd])  # port far -> nadir -> stbd far
            else:
                sounding = np.clip(np.rint(pa.starboard[i]), 0, 255).astype(np.uint8)
            frame = bytearray(hdr)
            struct.pack_into("<I", frame, 0, pos)  # frame offset (self)
            struct.pack_into("<H", frame, off["frame_size"], hdr + len(sounding))
            struct.pack_into("<H", frame, off["channel"], channel)
            struct.pack_into("<H", frame, off["packet_size"], len(sounding))
            struct.pack_into("<I", frame, off["frame_index"], i)
            struct.pack_into("<f", frame, off["upper_limit"], 0.0)
            struct.pack_into("<f", frame, off["lower_limit"], float(rec["slant_range"]) / FEET_TO_M)
            struct.pack_into("<f", frame, off["water_depth"], float(rec["altitude"]) / FEET_TO_M)
            struct.pack_into("<f", frame, off["gps_speed"], float(rec["speed"]) / KNOTS_TO_MS)
            struct.pack_into("<i", frame, off["x"], int(x_int[i]))
            struct.pack_into("<i", frame, off["y"], int(y_int[i]))
            struct.pack_into("<f", frame, off["course"], float(heading_rad[i]))
            struct.pack_into("<f", frame, off["heading"], float(heading_rad[i]))
            if fmt == FORMAT_SL3:
                struct.pack_into("<I", frame, off["epoch"], int(base))
                ms = round((float(rec["time"]) - base) * 1000.0)
            else:
                ms = round((float(rec["time"]) - sl2_base) * 1000.0)
            struct.pack_into("<I", frame, off["time_ms"], max(ms, 0))
            fh.write(bytes(frame))
            fh.write(sounding.tobytes())
            pos += hdr + len(sounding)
    return path
