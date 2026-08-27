"""Copilot extensions (SONAR-GPT blueprint): dimension/depth filter grammar,
combined queries, and rule-based survey summary drafting — all against an
in-memory repo with seeded contacts, fully offline (no LLM endpoint)."""

from __future__ import annotations

import pytest

from api.copilot import DimFilter, _parse_dim_filters, ask, draft_summary
from api.db import ContactRepo
from geoscribe.contact import (
    Contact,
    Dimensions,
    PhysicsEvidence,
    PixelRef,
    ReviewStatus,
    SeverityBreakdown,
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rule mode must be exercised, not a stray local LLM endpoint."""
    monkeypatch.delenv("SAGAR_LLM_ENDPOINT", raising=False)


def _contact(
    cid: str,
    cls: str,
    *,
    length: float = 1.0,
    height: float | None = None,
    depth: float | None = None,
    severity: float = 50.0,
    confidence: float = 80.0,
    review: ReviewStatus = ReviewStatus.pending,
    violation: bool = False,
    survey: str = "survey_a.xtf",
    nearest_layer: str | None = None,
    nearest_dist: float | None = None,
) -> Contact:
    return Contact(
        id=cid,
        cls=cls,
        confidence=confidence,
        lat=13.05,
        lon=80.35,
        dims=Dimensions(length_m=length, width_m=1.0, height_m=height),
        physics=PhysicsEvidence(
            highlight=True, shadow=height is not None, physics_violation=violation
        ),
        severity=severity,
        severity_breakdown=SeverityBreakdown(
            nearest_layer=nearest_layer, nearest_layer_distance_m=nearest_dist
        ),
        pixel=PixelRef(side="port", ping0=0, ping1=10, col0=5, col1=20),
        depth_m=depth,
        survey=survey,
        review=review,
    )


@pytest.fixture()
def repo() -> ContactRepo:
    r = ContactRepo(":memory:")
    r.add_contacts(
        [
            _contact("SN-1", "tire", length=2.0, height=0.4, depth=10.0, severity=30.0),
            _contact("SN-2", "container", length=6.0, height=2.5, depth=30.0, severity=70.0),
            _contact(
                "SN-3", "ghost_net", length=12.0, height=1.0, depth=50.0,
                severity=80.0, review=ReviewStatus.confirmed,
            ),
            _contact(
                "SN-4", "wreck", length=30.0, height=5.0, depth=35.0, severity=90.0,
                violation=True, nearest_layer="shipping_lane", nearest_dist=500.0,
            ),
            # Unmeasured: no shadow -> height None, no altitude -> depth None.
            _contact("SN-5", "container", length=4.0, severity=40.0, survey="survey_b.xtf"),
        ]
    )
    yield r
    r.close()


class TestDimFilterGrammar:
    def test_parses_length_height_depth(self) -> None:
        filters, scrubbed = _parse_dim_filters(
            "contacts longer than 5 m and over 2 m tall between 20 and 40 m depth"
        )
        assert {f.field for f in filters} == {"length", "height", "depth"}
        depth = next(f for f in filters if f.field == "depth")
        assert (depth.lo, depth.hi) == (20.0, 40.0)
        # Measurement words must not survive to the severity-band scan.
        assert "tall" not in scrubbed and "depth" not in scrubbed

    def test_unmeasured_never_matches(self) -> None:
        f = DimFilter("height", lo=2.0)
        assert not f.matches({"dims": {"height_m": None}, "depth_m": None})
        assert f.matches({"dims": {"height_m": 3.0}})


class TestDimensionQueries:
    def test_length_count(self, repo: ContactRepo) -> None:
        got = ask("how many contacts longer than 5 m?", repo)
        assert got["mode"] == "rules"
        assert got["rows"] == [{"n": 3}]  # SN-2, SN-3, SN-4
        assert "3 contacts match" in got["answer"]
        assert "length > 5 m" in got["answer"]

    def test_height_filter_excludes_unmeasured(self, repo: ContactRepo) -> None:
        got = ask("how many contacts over 2 m tall?", repo)
        assert got["rows"] == [{"n": 2}]  # SN-2 (2.5), SN-4 (5.0); SN-5 has no height
        assert "height > 2 m" in got["answer"]

    def test_depth_range_listing(self, repo: ContactRepo) -> None:
        got = ask("show contacts between 20 and 40 m depth", repo)
        assert {r["id"] for r in got["rows"]} == {"SN-2", "SN-4"}
        assert "depth 20-40 m" in got["answer"]

    def test_deeper_than_combined_with_class(self, repo: ContactRepo) -> None:
        got = ask("how many wrecks deeper than 30 m?", repo)
        assert got["rows"] == [{"n": 1}]  # SN-4 at 35 m
        assert "class=wreck" in got["answer"]
        assert "depth > 30 m" in got["answer"]

    def test_combined_with_review_status(self, repo: ContactRepo) -> None:
        got = ask("how many confirmed contacts longer than 5 m?", repo)
        assert got["rows"] == [{"n": 1}]  # SN-3
        assert "review=confirmed" in got["answer"]
        assert "length > 5 m" in got["answer"]

    def test_shallower_than(self, repo: ContactRepo) -> None:
        got = ask("top 5 contacts shallower than 15 m", repo)
        assert [r["id"] for r in got["rows"]] == ["SN-1"]  # SN-5 depth unknown: excluded
        assert "depth < 15 m" in got["answer"]

    def test_no_match_states_filters(self, repo: ContactRepo) -> None:
        got = ask("show tires taller than 4 m", repo)
        assert got["rows"] == []
        assert "No matching contacts" in got["answer"]
        assert "class=tire" in got["answer"] and "height > 4 m" in got["answer"]


class TestSummaryDrafting:
    def test_ask_routes_summary_intent(self, repo: ContactRepo) -> None:
        got = ask("give me a summary of the situation", repo)
        assert got["mode"] == "rules"
        assert "5 contact(s)" in got["answer"]
        assert "container x2" in got["answer"]
        assert "Physics violations:** 1 of 5" in got["answer"]
        assert "1 confirmed" in got["answer"] and "4 pending" in got["answer"]
        assert "shipping lane" in got["answer"] and "500" in got["answer"]

    def test_top3_have_positions(self, repo: ContactRepo) -> None:
        answer = draft_summary(repo)["answer"]
        for cid in ("SN-4", "SN-3", "SN-2"):  # severity 90, 80, 70
            assert cid in answer
        assert "SN-1" not in answer.split("Top contacts")[1].split("Physics")[0]
        assert "13.05000, 80.35000" in answer

    def test_survey_scoped(self, repo: ContactRepo) -> None:
        got = draft_summary(repo, survey="survey_b")
        assert got["mode"] == "rules"
        assert "1 contact(s)" in got["answer"]
        assert "SN-5" in got["answer"]

    def test_report_word_routes_and_scopes(self, repo: ContactRepo) -> None:
        got = ask("draft a report for survey survey_b", repo)
        assert "survey_b" in got["answer"]
        assert "SN-5" in got["answer"]

    def test_empty_repo(self) -> None:
        empty = ContactRepo(":memory:")
        try:
            got = draft_summary(empty)
            assert got["mode"] == "rules"
            assert "No contacts" in got["answer"]
        finally:
            empty.close()
