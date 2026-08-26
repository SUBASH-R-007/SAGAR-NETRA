"""Write a :class:`PingArray` out as a spec-compliant XTF file.

Used by the synthetic-data factory to produce the bundled sample survey, and
by tests to round-trip the XTF adapter against real XTF bytes. Samples are
stored as unsigned 16-bit (``BytesPerSample=2`` / ``SampleFormat=3``), the
most common side-scan storage format.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pyxtf import XTFChannelType, XTFHeaderType
from pyxtf.xtf_ctypes import XTFFileHeader, XTFPingChanHeader, XTFPingHeader

from sonar_core.parsers.base import KNOTS_TO_MS, PingArray

XTF_MAGIC = 0xFACE
_UINT16_MAX = 65535


def _fill_channel_info(info, side_type: XTFChannelType, name: bytes) -> None:
    info.TypeOfChannel = side_type.value
    info.SubChannelNumber = 0
    info.CorrectionFlags = 1  # 1 = data stored in slant range (uncorrected)
    info.UniPolar = 1
    info.BytesPerSample = 2
    info.SampleFormat = 3  # 3 = unsigned 16-bit (X41 extension field)
    info.ChannelName = name


def write_xtf(pa: PingArray, path: str | Path) -> Path:
    """Serialize *pa* to *path*; intensities are clipped to uint16 range."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = XTFFileHeader()
    header.FileFormat = 0x7B  # 123, per XTF spec
    header.SystemType = 1
    header.RecordingProgramName = b"SAGARNET"  # field is 8 bytes
    header.RecordingProgramVersion = b"010"
    header.SonarName = pa.meta.get("sonar_name", "SYNTH-SSS").encode()[:15]
    header.SonarType = 0
    header.NavUnits = 3  # 3 = lat/lon in degrees
    header.NumberOfSonarChannels = 2
    _fill_channel_info(header.ChanInfo[0], XTFChannelType.port, b"Port")
    _fill_channel_info(header.ChanInfo[1], XTFChannelType.stbd, b"Starboard")

    n_port = pa.n_samples("port")
    n_stbd = pa.n_samples("starboard")

    with path.open("wb") as fh:
        fh.write(bytes(header))

        for i in range(pa.n_pings):
            rec = pa.nav[i]
            ping = XTFPingHeader()
            ping.MagicNumber = XTF_MAGIC
            ping.HeaderType = XTFHeaderType.sonar.value
            ping.SubChannelNumber = 0
            ping.NumChansToFollow = 2
            ping.NumBytesThisRecord = 256 + (64 + n_port * 2) + (64 + n_stbd * 2)

            t = float(rec["time"])
            if np.isfinite(t):
                dt = datetime.fromtimestamp(t, tz=UTC)
                ping.Year, ping.Month, ping.Day = dt.year, dt.month, dt.day
                ping.Hour, ping.Minute, ping.Second = dt.hour, dt.minute, dt.second
                ping.HSeconds = dt.microsecond // 10_000
                ping.JulianDay = dt.timetuple().tm_yday
            ping.PingNumber = i
            ping.SoundVelocity = float(rec["sound_velocity"]) / 2.0  # spec: one-way
            ping.SensorYcoordinate = float(rec["lat"])
            ping.SensorXcoordinate = float(rec["lon"])
            ping.SensorHeading = float(rec["heading"])
            ping.SensorPrimaryAltitude = float(rec["altitude"])
            ping.SensorDepth = float(rec["sensor_depth"])
            ping.SensorSpeed = float(rec["speed"]) / KNOTS_TO_MS  # spec: knots
            ping.Layback = float(rec["layback"])
            fh.write(bytes(ping))

            sv = max(float(rec["sound_velocity"]), 1.0)
            for channel_number, side, n_samples in (
                (0, "port", n_port),
                (1, "starboard", n_stbd),
            ):
                chan = XTFPingChanHeader()
                chan.ChannelNumber = channel_number
                chan.SlantRange = float(rec["slant_range"])
                chan.NumSamples = n_samples
                chan.TimeDuration = 2.0 * float(rec["slant_range"]) / sv
                chan.Weight = 0
                fh.write(bytes(chan))

                samples = np.clip(pa.side(side)[i], 0, _UINT16_MAX)
                fh.write(samples.astype("<u2").tobytes())

    return path
