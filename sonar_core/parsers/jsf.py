"""EdgeTech JSF adapter: message framing + message type 80 (sonar trace data).

Field offsets follow the EdgeTech JSF spec as implemented by MB-System's
``mbsys_jstar.h`` (the de-facto reference implementation):

Message header (16 bytes, little endian):
    0  uint16  start marker, 0x1601
    2  uint8   protocol version
    3  uint8   session id
    4  uint16  message type (80 = sonar trace)
    6  uint8   command type (2 = data)
    7  uint8   subsystem (0 = subbottom, 20 = low-freq SSS, 21 = high-freq SSS)
    8  uint8   channel (0 = port, 1 = starboard)
    9  uint8   sequence number
    10 uint16  reserved
    12 uint32  size of following payload, bytes

Message 80 trace header (240 bytes), offsets used here:
    0   int32   ping time, epoch seconds
    8   uint32  ping number
    34  int16   data format (0 = envelope: one uint16 per sample)
    80  int32   coordX  (longitude when coordUnits == 2)
    84  int32   coordY  (latitude  when coordUnits == 2)
    88  int16   coordUnits (1 = mm, 2 = arc-minutes * 1e4, 3 = decimetres)
    114 uint16  samples in this packet
    116 uint32  sampling interval, nanoseconds
    136 int32   sonar depth, millimetres
    144 int32   sonar altitude, millimetres
    148 float32 sound speed, m/s
    168 int16   weighting factor N (true amplitude = sample * 2^-N)
    172 int16   heading, 1/100 degree
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from sonar_core.parsers.base import (
    NAV_DTYPE,
    ParserError,
    PingArray,
    SonarParser,
    register_parser,
)

JSF_MARKER = 0x1601
MSG_SONAR_TRACE = 80
_MSG_HEADER = struct.Struct("<HBBHBBBBHI")
TRACE_HEADER_SIZE = 240
ARCMIN_1E4_PER_DEG = 60.0 * 10_000.0


def _read_trace(payload: bytes) -> dict[str, Any]:
    """Decode the fields we use from a message-80 payload."""
    if len(payload) < TRACE_HEADER_SIZE:
        raise ParserError(f"JSF trace payload too short: {len(payload)} bytes")
    h = payload[:TRACE_HEADER_SIZE]
    (n_samples,) = struct.unpack_from("<H", h, 114)
    (weight,) = struct.unpack_from("<h", h, 168)
    data = np.frombuffer(
        payload, dtype="<u2", count=n_samples, offset=TRACE_HEADER_SIZE
    ).astype(np.float32)
    if weight:
        data = data * float(2.0**-weight)
    (coord_units,) = struct.unpack_from("<h", h, 88)
    (coord_x,) = struct.unpack_from("<i", h, 80)
    (coord_y,) = struct.unpack_from("<i", h, 84)
    if coord_units == 2:
        lon = coord_x / ARCMIN_1E4_PER_DEG
        lat = coord_y / ARCMIN_1E4_PER_DEG
    else:  # projected units — surfaced in meta, nav left NaN for geotagging to reject
        lon = lat = float("nan")
    (sample_interval_ns,) = struct.unpack_from("<I", h, 116)
    (sound_speed,) = struct.unpack_from("<f", h, 148)
    sv = sound_speed if sound_speed > 0 else 1500.0
    return {
        "time": float(struct.unpack_from("<i", h, 0)[0]),
        "ping": int(struct.unpack_from("<I", h, 8)[0]),
        "lat": lat,
        "lon": lon,
        "coord_units": coord_units,
        "depth": struct.unpack_from("<i", h, 136)[0] / 1000.0,
        "altitude": struct.unpack_from("<i", h, 144)[0] / 1000.0,
        "sound_velocity": sv,
        "heading": struct.unpack_from("<h", h, 172)[0] / 100.0,
        "slant_range": n_samples * sample_interval_ns * 1e-9 * sv / 2.0,
        "data": data,
    }


def _iter_messages(fh: BinaryIO):
    """Yield (message_type, channel, subsystem, payload) for every JSF message."""
    while True:
        raw = fh.read(_MSG_HEADER.size)
        if len(raw) < _MSG_HEADER.size:
            return
        marker, _ver, _sess, msg_type, _cmd, subsystem, channel, _seq, _res, size = (
            _MSG_HEADER.unpack(raw)
        )
        if marker != JSF_MARKER:
            raise ParserError(f"bad JSF start marker 0x{marker:04x} (file corrupt?)")
        payload = fh.read(size)
        if len(payload) < size:
            return  # truncated final message
        yield msg_type, channel, subsystem, payload


@register_parser
class JSFParser(SonarParser):
    suffixes = (".jsf",)

    def parse(self, path: Path, **kwargs: Any) -> PingArray:
        # ping number -> {"port": trace, "starboard": trace}
        pings: dict[int, dict[str, dict[str, Any]]] = {}
        subsystems: set[int] = set()
        with path.open("rb") as fh:
            for msg_type, channel, subsystem, payload in _iter_messages(fh):
                if msg_type != MSG_SONAR_TRACE or subsystem == 0:
                    continue  # skip subbottom and non-trace messages
                subsystems.add(subsystem)
                trace = _read_trace(payload)
                side = "port" if channel == 0 else "starboard"
                pings.setdefault(trace["ping"], {})[side] = trace

        if not pings:
            raise ParserError(f"{path.name}: no sidescan message-80 records found")

        ping_numbers = sorted(pings)
        n_pings = len(ping_numbers)

        def width(side: str) -> int:
            counts = [len(p[side]["data"]) for p in pings.values() if side in p]
            return int(np.bincount(counts).argmax()) if counts else 0

        n_port, n_stbd = width("port"), width("starboard")
        port = np.zeros((n_pings, n_port), dtype=np.float32)
        stbd = np.zeros((n_pings, n_stbd), dtype=np.float32)
        nav = np.zeros(n_pings, dtype=NAV_DTYPE)

        for i, ping_no in enumerate(ping_numbers):
            record = pings[ping_no]
            ref = record.get("starboard") or record["port"]
            nav[i]["time"] = ref["time"]
            nav[i]["lat"] = ref["lat"]
            nav[i]["lon"] = ref["lon"]
            nav[i]["heading"] = ref["heading"]
            nav[i]["altitude"] = ref["altitude"]
            nav[i]["sensor_depth"] = ref["depth"]
            nav[i]["sound_velocity"] = ref["sound_velocity"]
            nav[i]["slant_range"] = ref["slant_range"]
            for side, dest, w in (("port", port, n_port), ("starboard", stbd, n_stbd)):
                if side in record and w:
                    d = record[side]["data"]
                    n = min(len(d), w)
                    dest[i, :n] = d[:n]

        meta = {
            "format": "jsf",
            "sonar_name": "EdgeTech-JSF",
            "subsystems": sorted(subsystems),
            "coord_units": int(next(iter(pings.values())).popitem()[1]["coord_units"]),
        }
        return PingArray(port=port, starboard=stbd, nav=nav, source=str(path), meta=meta)


def write_jsf(pa: PingArray, path: str | Path, subsystem: int = 20) -> Path:
    """Serialize *pa* as JSF message-80 records (used for round-trip tests)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        for i in range(pa.n_pings):
            rec = pa.nav[i]
            sv = max(float(rec["sound_velocity"]), 1.0)
            for channel, side in ((0, "port"), (1, "starboard")):
                data = np.clip(pa.side(side)[i], 0, 65535).astype("<u2")
                n = len(data)
                header = bytearray(TRACE_HEADER_SIZE)
                struct.pack_into("<i", header, 0, int(rec["time"]))
                struct.pack_into("<I", header, 8, i)
                struct.pack_into("<h", header, 34, 0)  # envelope format
                struct.pack_into("<i", header, 80, round(float(rec["lon"]) * ARCMIN_1E4_PER_DEG))
                struct.pack_into("<i", header, 84, round(float(rec["lat"]) * ARCMIN_1E4_PER_DEG))
                struct.pack_into("<h", header, 88, 2)
                struct.pack_into("<H", header, 114, n)
                interval_ns = round(2.0 * float(rec["slant_range"]) / max(n, 1) / sv * 1e9)
                struct.pack_into("<I", header, 116, interval_ns)
                struct.pack_into("<i", header, 136, round(float(rec["sensor_depth"]) * 1000))
                struct.pack_into("<i", header, 144, round(float(rec["altitude"]) * 1000))
                struct.pack_into("<f", header, 148, sv)
                struct.pack_into("<h", header, 168, 0)
                struct.pack_into("<h", header, 172, round(float(rec["heading"]) * 100))
                payload_size = TRACE_HEADER_SIZE + data.nbytes
                fh.write(
                    _MSG_HEADER.pack(
                        JSF_MARKER, 10, 0, MSG_SONAR_TRACE, 2, subsystem, channel, 0, 0,
                        payload_size,
                    )
                )
                fh.write(bytes(header))
                fh.write(data.tobytes())
    return path
