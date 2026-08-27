"""XTF adapter: write a PingArray with the spec-compliant writer, read it back
with the parser, and verify intensities and navigation survive the trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sonar_core.parsers.base import load
from sonar_core.parsers.xtf import XTFParser
from sonar_core.parsers.xtf_writer import write_xtf


@pytest.fixture(scope="module")
def roundtrip(small_scene, tmp_path_factory) -> tuple:
    pa, _targets = small_scene
    path = tmp_path_factory.mktemp("xtf") / "roundtrip.xtf"
    write_xtf(pa, path)
    return pa, XTFParser().parse(path), path


def test_intensities_survive_uint16_quantisation(roundtrip) -> None:
    original, parsed, _ = roundtrip
    # Writer stores uint16, so agreement should be to rounding.
    np.testing.assert_allclose(parsed.port, np.floor(original.port), atol=1.0)
    np.testing.assert_allclose(parsed.starboard, np.floor(original.starboard), atol=1.0)
    assert parsed.port.shape == original.port.shape


def test_navigation_survives(roundtrip) -> None:
    original, parsed, _ = roundtrip
    np.testing.assert_allclose(parsed.nav["lat"], original.nav["lat"], atol=1e-7)
    np.testing.assert_allclose(parsed.nav["lon"], original.nav["lon"], atol=1e-7)
    np.testing.assert_allclose(parsed.nav["heading"], original.nav["heading"], atol=0.01)
    np.testing.assert_allclose(parsed.nav["altitude"], original.nav["altitude"], atol=1e-3)
    np.testing.assert_allclose(parsed.nav["slant_range"], original.nav["slant_range"], atol=1e-3)
    np.testing.assert_allclose(
        parsed.nav["sound_velocity"], original.nav["sound_velocity"], atol=0.1
    )
    np.testing.assert_allclose(parsed.nav["speed"], original.nav["speed"], atol=1e-3)
    np.testing.assert_allclose(parsed.nav["time"], original.nav["time"], atol=0.011)


def test_load_dispatches_by_suffix(roundtrip) -> None:
    _, _, path = roundtrip
    pa = load(path)
    assert pa.meta["format"] == "xtf"
    assert pa.meta["sonar_name"] == "SYNTH-SSS"


def test_load_unknown_suffix_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "file.xyz"
    bogus.write_bytes(b"nonsense")
    from sonar_core.parsers.base import ParserError

    with pytest.raises(ParserError, match="no parser"):
        load(bogus)
