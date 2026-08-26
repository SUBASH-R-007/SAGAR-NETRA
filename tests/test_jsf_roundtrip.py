"""JSF adapter: round-trip against our spec-following message-80 writer."""

from __future__ import annotations

import numpy as np
import pytest

from sonar_core.parsers.jsf import JSFParser, write_jsf


@pytest.fixture(scope="module")
def roundtrip(small_scene, tmp_path_factory) -> tuple:
    pa, _ = small_scene
    path = tmp_path_factory.mktemp("jsf") / "roundtrip.jsf"
    write_jsf(pa, path)
    return pa, JSFParser().parse(path)


def test_intensities(roundtrip) -> None:
    original, parsed = roundtrip
    np.testing.assert_allclose(parsed.port, np.floor(original.port), atol=1.0)
    np.testing.assert_allclose(parsed.starboard, np.floor(original.starboard), atol=1.0)


def test_navigation(roundtrip) -> None:
    original, parsed = roundtrip
    # 0.0001 arc-minute lat/lon grid -> ~0.19 m worst case
    np.testing.assert_allclose(parsed.nav["lat"], original.nav["lat"], atol=2e-6)
    np.testing.assert_allclose(parsed.nav["lon"], original.nav["lon"], atol=2e-6)
    np.testing.assert_allclose(parsed.nav["heading"], original.nav["heading"], atol=0.01)
    np.testing.assert_allclose(parsed.nav["altitude"], original.nav["altitude"], atol=1e-3)
    np.testing.assert_allclose(
        parsed.nav["slant_range"], original.nav["slant_range"], rtol=1e-4
    )
    np.testing.assert_allclose(parsed.nav["time"], original.nav["time"], atol=1.0)


def test_meta(roundtrip) -> None:
    _, parsed = roundtrip
    assert parsed.meta["format"] == "jsf"
    assert parsed.meta["subsystems"] == [20]
