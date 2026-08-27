"""Humminbird .DAT + .SON/.IDX adapter: round-trip against our spec-following
writer (the jsf.py validation pattern), byte-level nadir-first check, the
IDX-less marker-scan path, load() dispatch, and corrupt-file handling."""

from __future__ import annotations

import numpy as np
import pytest

from sonar_core.parsers.base import NAV_DTYPE, ParserError, PingArray, load
from sonar_core.parsers.humminbird import (
    HumminbirdParser,
    write_humminbird,
)

N_PINGS = 24
N_SAMPLES = 40
PIXEL_SIZE_M = 0.02


def _make_pa() -> PingArray:
    """Tiny survey with asymmetric sides in the uint8 sounding range.

    Port row 0 increases strictly from nadir outward so a mirrored port
    array cannot match the original values.
    """
    cols = np.arange(N_SAMPLES, dtype=np.float32)
    rows = np.arange(N_PINGS, dtype=np.float32)[:, None]
    port = (5.0 * cols[None, :] + 2.0 * rows) % 251.0
    starboard = (230.0 - 4.0 * cols[None, :] + 3.0 * rows) % 251.0
    nav = np.zeros(N_PINGS, dtype=NAV_DTYPE)
    nav["time"] = 1.7e9 + 0.25 * np.arange(N_PINGS)
    nav["lat"] = 12.9 + 1e-5 * np.arange(N_PINGS)
    nav["lon"] = 80.2 + 2e-5 * np.arange(N_PINGS)
    nav["heading"] = (170.0 + 0.5 * np.arange(N_PINGS)) % 360.0
    nav["altitude"] = 3.0 + 0.02 * np.arange(N_PINGS)  # boat mount: = water depth
    nav["sound_velocity"] = 1500.0
    nav["slant_range"] = N_SAMPLES * PIXEL_SIZE_M  # cnt * pixel size reconstruction
    nav["speed"] = 1.75
    return PingArray(port=port, starboard=starboard, nav=nav, source="synthetic")


@pytest.fixture(scope="module")
def roundtrip(tmp_path_factory: pytest.TempPathFactory) -> tuple:
    pa = _make_pa()
    dat = tmp_path_factory.mktemp("humminbird") / "R00001.DAT"
    write_humminbird(pa, dat)
    return pa, HumminbirdParser().parse(dat, pixel_size_m=PIXEL_SIZE_M), dat


def test_intensities(roundtrip) -> None:
    original, parsed, _ = roundtrip
    np.testing.assert_allclose(parsed.port, original.port, atol=0.5)  # uint8 quantization
    np.testing.assert_allclose(parsed.starboard, original.starboard, atol=0.5)


def test_port_nadir_first_on_disk(roundtrip) -> None:
    """B002.SON ping bytes start at the transducer (nadir) and must never be
    mirrored: the first record's payload equals the port row verbatim."""
    original, parsed, dat = roundtrip
    son = (dat.with_suffix("") / "B002.SON").read_bytes()
    payload = np.frombuffer(son, dtype=np.uint8, count=N_SAMPLES, offset=67)
    np.testing.assert_array_equal(payload, np.rint(original.port[0]).astype(np.uint8))
    np.testing.assert_allclose(parsed.port[0], original.port[0], atol=0.5)


def test_navigation(roundtrip) -> None:
    original, parsed, _ = roundtrip
    # 1 m mercator integer grid at R = 6378388 m -> ~9e-6 degrees per count
    np.testing.assert_allclose(parsed.nav["lat"], original.nav["lat"], atol=2e-5)
    np.testing.assert_allclose(parsed.nav["lon"], original.nav["lon"], atol=2e-5)
    np.testing.assert_allclose(parsed.nav["heading"], original.nav["heading"], atol=0.06)
    np.testing.assert_allclose(parsed.nav["altitude"], original.nav["altitude"], atol=0.006)
    np.testing.assert_allclose(parsed.nav["speed"], original.nav["speed"], atol=0.006)
    np.testing.assert_allclose(parsed.nav["slant_range"], original.nav["slant_range"], rtol=1e-6)
    np.testing.assert_allclose(parsed.nav["time"], original.nav["time"], atol=2e-3)
    assert np.all(parsed.nav["sensor_depth"] == 0.0)  # boat mount: sensor at surface


def test_meta(roundtrip) -> None:
    _, parsed, _ = roundtrip
    assert parsed.meta["format"] == "humminbird"
    assert parsed.meta["beams"] == {"port": "B002", "starboard": "B003"}
    assert parsed.meta["numrecords"] == N_PINGS


def test_load_dispatch(roundtrip) -> None:
    _, _, dat = roundtrip
    parsed = load(dat)  # .DAT suffix routes to HumminbirdParser
    assert parsed.meta["format"] == "humminbird"
    assert parsed.n_pings == N_PINGS


def test_without_idx(tmp_path) -> None:
    """Deleting the .IDX sidecars must fall back to head-marker scanning."""
    pa = _make_pa()
    dat = tmp_path / "R00002.DAT"
    write_humminbird(pa, dat)
    for idx in dat.with_suffix("").glob("*.IDX"):
        idx.unlink()
    parsed = HumminbirdParser().parse(dat, pixel_size_m=PIXEL_SIZE_M)
    np.testing.assert_allclose(parsed.port, pa.port, atol=0.5)
    np.testing.assert_allclose(parsed.nav["lat"], pa.nav["lat"], atol=2e-5)


def test_corrupt_files(tmp_path) -> None:
    short = tmp_path / "short.DAT"
    short.write_bytes(b"\x00" * 10)  # far below the 56-byte header minimum
    with pytest.raises(ParserError):
        HumminbirdParser().parse(short)

    lonely = tmp_path / "lonely.DAT"  # valid size but no recording directory
    lonely.write_bytes(b"\x00" * 64)
    with pytest.raises(ParserError):
        HumminbirdParser().parse(lonely)

    pa = _make_pa()
    dat = tmp_path / "R00003.DAT"
    write_humminbird(pa, dat)
    son = dat.with_suffix("") / "B002.SON"
    son.write_bytes(b"\xff" * 128)  # garbage: no head markers, IDX now points wrong
    with pytest.raises(ParserError):
        HumminbirdParser().parse(dat)

    (dat.with_suffix("") / "B003.SON").unlink()  # missing sidescan beam
    with pytest.raises(ParserError):
        HumminbirdParser().parse(dat)
