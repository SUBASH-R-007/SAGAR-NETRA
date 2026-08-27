"""XTF (Triton eXtended Triton Format) adapter, built on ``pyxtf``.

Reads per-ping navigation from ``XTFPingHeader`` (``SensorX/Ycoordinate``,
``SensorPrimaryAltitude``, ``SensorHeading``, ...) and slant range from the
per-channel ``XTFPingChanHeader``. Port/starboard channels are identified via
``TypeOfChannel`` in the file header's ``ChanInfo`` blocks.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyxtf
from pyxtf import XTFChannelType, XTFHeaderType

from sonar_core.parsers.base import (
    KNOTS_TO_MS,
    NAV_DTYPE,
    ParserError,
    PingArray,
    SonarParser,
    register_parser,
)


def _ping_epoch(ping: Any) -> float:
    """UTC epoch seconds from the XTF ping-header date/time fields."""
    try:
        dt = datetime(
            int(ping.Year),
            int(ping.Month),
            int(ping.Day),
            int(ping.Hour),
            int(ping.Minute),
            int(ping.Second),
            int(ping.HSeconds) * 10_000,  # hundredths -> microseconds
            tzinfo=UTC,
        )
    except ValueError:  # zeroed/garbage date fields in some loggers
        return float("nan")
    return dt.timestamp()


def _sound_velocity_two_way(raw: float) -> float:
    """XTF stores one-way sound velocity (~750 m/s) per spec; normalize to ~1500."""
    if 0.0 < raw < 1000.0:
        return raw * 2.0
    return raw if raw > 0.0 else 1500.0


@register_parser
class XTFParser(SonarParser):
    suffixes = (".xtf",)

    def parse(self, path: Path, **kwargs: Any) -> PingArray:
        file_header, packets = pyxtf.xtf_read(str(path), types=[XTFHeaderType.sonar])
        pings = packets.get(XTFHeaderType.sonar, [])
        if not pings:
            raise ParserError(f"{path.name}: no sonar pings found")

        # Map channel index -> side using the file header's ChanInfo blocks.
        side_of_channel: dict[int, str] = {}
        for idx, info in enumerate(file_header.ChanInfo):
            channel_type = int(info.TypeOfChannel)
            if channel_type == XTFChannelType.port.value:
                side_of_channel[idx] = "port"
            elif channel_type == XTFChannelType.stbd.value:
                side_of_channel[idx] = "starboard"
        if not side_of_channel:
            raise ParserError(f"{path.name}: no port/starboard channels in file header")

        # Sample counts can vary ping-to-ping; normalise to the modal count.
        def modal_samples(side: str) -> int:
            counts = [
                len(ping.data[ch])
                for ping in pings
                for ch, s in side_of_channel.items()
                if s == side and ping.data is not None and ch < len(ping.data)
            ]
            if not counts:
                raise ParserError(f"{path.name}: no data for {side} channel")
            return int(np.bincount(counts).argmax())

        n_port = modal_samples("port")
        n_stbd = modal_samples("starboard")
        n_pings = len(pings)

        port = np.zeros((n_pings, n_port), dtype=np.float32)
        stbd = np.zeros((n_pings, n_stbd), dtype=np.float32)
        nav = np.zeros(n_pings, dtype=NAV_DTYPE)

        for i, ping in enumerate(pings):
            nav[i]["time"] = _ping_epoch(ping)
            nav[i]["lat"] = float(ping.SensorYcoordinate)
            nav[i]["lon"] = float(ping.SensorXcoordinate)
            nav[i]["heading"] = float(ping.SensorHeading)
            nav[i]["altitude"] = float(ping.SensorPrimaryAltitude)
            nav[i]["sensor_depth"] = float(ping.SensorDepth)
            nav[i]["sound_velocity"] = _sound_velocity_two_way(float(ping.SoundVelocity))
            nav[i]["speed"] = float(ping.SensorSpeed) * KNOTS_TO_MS
            nav[i]["layback"] = float(ping.Layback)

            slant = 0.0
            for ch_header in ping.ping_chan_headers:
                slant = max(slant, float(ch_header.SlantRange))
            nav[i]["slant_range"] = slant

            if ping.data is None:
                continue
            for ch, side in side_of_channel.items():
                if ch >= len(ping.data):
                    continue
                samples = np.asarray(ping.data[ch], dtype=np.float32)
                dest, width = (port, n_port) if side == "port" else (stbd, n_stbd)
                n = min(len(samples), width)
                dest[i, :n] = samples[:n]

        meta = {
            "format": "xtf",
            "sonar_name": bytes(file_header.SonarName).split(b"\x00")[0].decode(errors="replace"),
            "nav_units": int(file_header.NavUnits),
            "n_dropped_channels": max(0, int(file_header.NumberOfSonarChannels) - 2),
        }
        return PingArray(port=port, starboard=stbd, nav=nav, source=str(path), meta=meta)
