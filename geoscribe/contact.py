"""The Contact model: one verified detection, as it flows to reports, the API,
and the dashboard. Pydantic v2 so the JSON Schema is generated, published, and
validated in tests — this is the system's central data contract.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, computed_field


class ReviewStatus(StrEnum):
    pending = "pending"
    confirmed = "confirmed"
    rejected = "rejected"


class RecoveryStatus(StrEnum):
    """Physical recovery workflow: every contact starts ``flagged``; operations
    assign a retrieval asset (``assigned``) and close the loop once the object
    is on deck (``retrieved``). Orthogonal to :class:`ReviewStatus`, which
    judges whether the *detection* is real."""

    flagged = "flagged"
    assigned = "assigned"
    retrieved = "retrieved"


class PixelRef(BaseModel):
    """Where the contact lives in the processed ground-range imagery."""

    side: str = Field(pattern="^(port|starboard)$")
    ping0: int = Field(ge=0, description="first ping row of the box (inclusive)")
    ping1: int = Field(ge=0)
    col0: int = Field(ge=0, description="first ground-range column (inclusive)")
    col1: int = Field(ge=0)


class Dimensions(BaseModel):
    length_m: float = Field(ge=0, description="along-track extent")
    width_m: float = Field(ge=0, description="across-track extent")
    height_m: float | None = Field(
        default=None, ge=0, description="height from shadow; None when no usable shadow"
    )
    along_track_resolution_m: float | None = Field(
        default=None,
        ge=0,
        description=(
            "beam footprint along-track at this contact's range (theta * R): the "
            "resolution floor on length_m. It grows linearly with range, so a "
            "far-range length is a much softer number than a near-range one"
        ),
    )


class PhysicsEvidence(BaseModel):
    """Acoustic cues from PhysiCheck, rendered on the Evidence Card."""

    highlight: bool
    shadow: bool
    highlight_ratio: float | None = None
    shadow_ratio: float | None = None
    shadow_len_m: float | None = None
    height_m: float | None = None
    physics_violation: bool = False
    violation_reason: str | None = None
    multipath_suspect: bool = Field(
        default=False,
        description=(
            "sits where the second bottom return lands (ground range A*sqrt(3)), "
            "so may be the seabed heard twice rather than an object. Advisory: it "
            "never lowers confidence, because real debris does lie there sometimes"
        ),
    )


class SeverityBreakdown(BaseModel):
    hazard: float = 0.0
    size: float = 0.0
    height: float = 0.0
    depth: float = 0.0
    proximity: float = 0.0
    nearest_layer: str | None = None
    nearest_layer_distance_m: float | None = None


class Contact(BaseModel):
    """One reportable detection with full provenance."""

    id: str = Field(description="stable contact id, e.g. SN-20260826-0001")
    cls: str = Field(description="reportable class name")
    confidence: float = Field(ge=0, le=100, description="calibrated confidence, percent")
    brains: list[str] = Field(
        default_factory=list, description="which TridentNet brains fired: A/B/C"
    )
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    corners: list[tuple[float, float]] = Field(
        default_factory=list, description="footprint corners as (lat, lon)"
    )
    dims: Dimensions
    physics: PhysicsEvidence
    severity: float = Field(ge=0, le=100)
    severity_breakdown: SeverityBreakdown = Field(default_factory=SeverityBreakdown)
    pixel: PixelRef
    depth_m: float | None = Field(
        default=None, description="water depth at the contact (sensor depth + altitude)"
    )
    detected_at: str | None = Field(default=None, description="ping UTC time, ISO-8601")
    survey: str = Field(default="", description="source file name / survey id")
    evidence_png: str | None = Field(default=None, description="Evidence Card image path")
    thumbnail_png: str | None = Field(default=None)
    review: ReviewStatus = ReviewStatus.pending
    recovery: RecoveryStatus = Field(
        default=RecoveryStatus.flagged,
        description="recovery workflow state (flagged -> assigned -> retrieved); "
        "the default keeps contacts stored before this field existed valid",
    )
    notes: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    # -- strategy-PDF section 5.3 triage fields. Safe defaults throughout so
    # contact JSON stored before this schema still validates unchanged.
    priority: str = Field(
        default="LOW",
        pattern="^(HIGH|MEDIUM|LOW)$",
        description=(
            "operator triage priority derived from the severity bands "
            "(HIGH >= 75, MEDIUM >= 50, else LOW); see geoscribe.report.priority_for"
        ),
    )
    recommended_action: str | None = Field(
        default=None,
        description=(
            "operator instruction from the (class, severity band) rule table; "
            "see geoscribe.report.recommended_action_for"
        ),
    )
    position_accuracy_m: float = Field(
        default=0.0,
        ge=0,
        description=(
            "honest position error budget: 2*ground_res + layback term + nav fix "
            "term (see geoscribe.build.position_accuracy); 0.0 only in legacy "
            "records written before the estimate existed"
        ),
    )

    @computed_field(  # type: ignore[prop-decorator]
        description="[pixel.ping0, pixel.ping1] convenience alias in the JSON dump"
    )
    @property
    def ping_range(self) -> tuple[int, int]:
        return (self.pixel.ping0, self.pixel.ping1)


def contacts_json_schema() -> dict[str, Any]:
    """Published JSON Schema for a contacts.json document.

    Contact items derive from the model in *serialization* mode so computed
    conveniences (``ping_range``) are documented alongside stored fields. The
    ``summary`` block is optional: documents written before schema 5.3 (and
    fast preview passes without survey stats) must keep validating.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SAGAR-NETRA contacts report",
        "type": "object",
        "required": ["survey", "generated_at", "contacts"],
        "properties": {
            "survey": {"type": "string"},
            "generated_at": {"type": "string"},
            "pipeline_version": {"type": "string"},
            "summary": {
                "type": "object",
                "description": "survey-level roll-up (strategy-PDF section 5.3)",
                "properties": {
                    "total_detections": {"type": "integer"},
                    "high_confidence": {
                        "type": "integer",
                        "description": "contacts at or above 70% calibrated confidence",
                    },
                    "area_surveyed_sqkm": {"type": ["number", "null"]},
                    "debris_density_per_sqkm": {"type": ["number", "null"]},
                    "sonar_config": {
                        "type": ["object", "null"],
                        "properties": {
                            "range_m": {"type": ["number", "null"]},
                            "altitude_m": {"type": ["number", "null"]},
                            "n_pings": {"type": "integer"},
                            "sound_velocity_mps": {"type": ["number", "null"]},
                            "across_track_resolution_m": {"type": ["number", "null"]},
                            "along_track_resolution_max_m": {"type": ["number", "null"]},
                            "sound_speed_range_error_max_m": {"type": ["number", "null"]},
                        },
                    },
                },
            },
            "contacts": {
                "type": "array",
                "items": Contact.model_json_schema(mode="serialization"),
            },
        },
    }
