"""Lowrance .sl2/.sl3 adapter: round-trip against our spec-following writer
(the jsf.py validation pattern), byte-level composite-order checks, load()
dispatch, corrupt-file handling, and a preprocess() flow check."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from sonar_core.parsers.base import NAV_DTYPE, ParserError, PingArray, load
from sonar_core.parsers.lowrance import (
    CHANNEL_PRIMARY,
    FILE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    LowranceParser,
    write_slx,
)

N_PINGS = 48
N_SAMPLES = 64


def _make_pa() -> PingArray:
    """Tiny survey with deliberately asymmetric sides (uint8-ranged values).

    Port row 0 is a strictly increasing ramp from nadir outward, so any
    accidental mirroring of the port half is caught by value comparison.
    """
    cols = np.arange(N_SAMPLES, dtype=np.float32)
    rows = np.arange(N_PINGS, dtype=np.float32)[:, None]
    port = (3.0 * cols[None, :] + rows) % 251.0
    starboard = (240.0 - 2.0 * cols[None, :] + 5.0 * rows) % 251.0
    nav = np.zeros(N_PINGS, dtype=NAV_DTYPE)
    nav["time"] = 1000.0 + 0.25 * np.arange(N_PINGS)
    nav["lat"] = 12.9 + 1e-5 * np.arange(N_PINGS)
    nav["lon"] = 80.2 + 2e-5 * np.arange(N_PINGS)
    nav["heading"] = (85.0 + 0.5 * np.arange(N_PINGS)) % 360.0
    nav["altitude"] = 2.0 + 0.01 * np.arange(N_PINGS)  # transom mount: = water depth
    nav["sound_velocity"] = 1500.0
    nav["slant_range"] = 30.0
    nav["speed"] = 2.5
    return PingArray(port=port, starboard=starboard, nav=nav, source="synthetic")


@pytest.fixture(scope="module")
def roundtrip_sl2(tmp_path_factory: pytest.TempPathFactory) -> tuple:
    pa = _make_pa()
    path = tmp_path_factory.mktemp("lowrance") / "survey.sl2"
    write_slx(pa, path)
    return pa, LowranceParser().parse(path), path


def test_intensities(roundtrip_sl2) -> None:
    original, parsed, _ = roundtrip_sl2
    np.testing.assert_allclose(parsed.port, original.port, atol=0.5)  # uint8 quantization
    np.testing.assert_allclose(parsed.starboard, original.starboard, atol=0.5)


def test_port_reversal_nadir_first(roundtrip_sl2) -> None:
    """On disk the composite must run port-far -> nadir -> starboard-far;
    parsed arrays must be nadir-first. Checked at the byte level so a
    matching writer/parser bug (neither reversing) cannot slip through."""
    original, parsed, path = roundtrip_sl2
    buf = path.read_bytes()
    hdr = FRAME_HEADER_SIZE[2]
    sounding0 = np.frombuffer(
        buf, dtype=np.uint8, count=2 * N_SAMPLES, offset=FILE_HEADER_SIZE + hdr
    )
    port0 = np.rint(original.port[0]).astype(np.uint8)
    stbd0 = np.rint(original.starboard[0]).astype(np.uint8)
    np.testing.assert_array_equal(sounding0[:N_SAMPLES], port0[::-1])  # far-first on disk
    np.testing.assert_array_equal(sounding0[N_SAMPLES:], stbd0)
    np.testing.assert_allclose(parsed.port[0], original.port[0], atol=0.5)  # nadir-first out


def test_navigation(roundtrip_sl2) -> None:
    original, parsed, _ = roundtrip_sl2
    # 1 m mercator integer grid at R = 6356752 m -> ~9e-6 degrees per count
    np.testing.assert_allclose(parsed.nav["lat"], original.nav["lat"], atol=1e-5)
    np.testing.assert_allclose(parsed.nav["lon"], original.nav["lon"], atol=1e-5)
    np.testing.assert_allclose(parsed.nav["heading"], original.nav["heading"], atol=0.01)
    np.testing.assert_allclose(parsed.nav["altitude"], original.nav["altitude"], atol=1e-3)
    np.testing.assert_allclose(parsed.nav["slant_range"], original.nav["slant_range"], rtol=1e-5)
    np.testing.assert_allclose(parsed.nav["time"], original.nav["time"], atol=2e-3)
    np.testing.assert_allclose(parsed.nav["speed"], original.nav["speed"], atol=1e-4)
    assert np.all(parsed.nav["sensor_depth"] == 0.0)  # transom mount: sensor at surface


def test_meta(roundtrip_sl2) -> None:
    _, parsed, _ = roundtrip_sl2
    assert parsed.meta["format"] == "sl2"
    assert parsed.meta["channel"] == "sidescan"


def test_sl3_roundtrip(tmp_path) -> None:
    pa = _make_pa()
    path = tmp_path / "survey.sl3"
    write_slx(pa, path)
    parsed = LowranceParser().parse(path)
    assert parsed.meta["format"] == "sl3"
    np.testing.assert_allclose(parsed.port, pa.port, atol=0.5)
    np.testing.assert_allclose(parsed.starboard, pa.starboard, atol=0.5)
    np.testing.assert_allclose(parsed.nav["lat"], pa.nav["lat"], atol=1e-5)
    np.testing.assert_allclose(parsed.nav["time"], pa.nav["time"], atol=2e-3)


def test_load_dispatch(roundtrip_sl2, tmp_path) -> None:
    _, _, sl2_path = roundtrip_sl2
    assert load(sl2_path).meta["format"] == "sl2"
    sl3_path = tmp_path / "dispatch.sl3"
    write_slx(_make_pa(), sl3_path)
    assert load(sl3_path).meta["format"] == "sl3"


def test_primary_fallback(tmp_path) -> None:
    """No sidescan frames -> primary channel duplicated to both sides + note."""
    pa = _make_pa()
    path = tmp_path / "primary.sl2"
    write_slx(pa, path, channel=CHANNEL_PRIMARY)
    parsed = LowranceParser().parse(path)
    assert parsed.meta["channel"] == "primary"
    assert "note" in parsed.meta
    np.testing.assert_array_equal(parsed.port, parsed.starboard)
    np.testing.assert_allclose(parsed.starboard, pa.starboard, atol=0.5)


def test_corrupt_files(tmp_path) -> None:
    bad_magic = tmp_path / "bad.sl2"
    bad_magic.write_bytes(b"garbage-not-a-lowrance-file")
    with pytest.raises(ParserError):
        LowranceParser().parse(bad_magic)

    truncated = tmp_path / "short.sl2"
    truncated.write_bytes(b"\x02\x00")
    with pytest.raises(ParserError):
        LowranceParser().parse(truncated)

    bad_frames = tmp_path / "frames.sl2"  # valid header, zero-size frame follows
    bad_frames.write_bytes(struct.pack("<HHHH", 2, 1, 3200, 0) + b"\x00" * 600)
    with pytest.raises(ParserError):
        LowranceParser().parse(bad_frames)

    empty = tmp_path / "empty.sl2"  # valid header, no frames at all
    empty.write_bytes(struct.pack("<HHHH", 2, 1, 3200, 0))
    with pytest.raises(ParserError):
        LowranceParser().parse(empty)


def test_preprocess_flow(roundtrip_sl2) -> None:
    """A parsed Lowrance survey must flow through the full M2 chain (bottom
    tracking may fall back to the header altitude — that is fine)."""
    from sonar_core.preprocess.pipeline import preprocess

    _, parsed, _ = roundtrip_sl2
    result = preprocess(parsed)
    assert result.ground.n_pings == N_PINGS
    assert len(result.tiles) >= 1
    for side in ("port", "starboard"):
        img = result.ground.side(side)
        assert img.shape[0] == N_PINGS
        assert np.isfinite(img).any()
