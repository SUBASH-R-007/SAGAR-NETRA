"""The Contact model: one verified detection, as it flows to reports, the API,
and the dashboard. Pydantic v2 so the JSON Schema is generated, published, and
validated in tests — this is the system's central data contract.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ReviewStatus(StrEnum):
    pending = "pending"
    confirmed = "confirmed"
    rejected = "rejected"


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
    notes: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


def contacts_json_schema() -> dict[str, Any]:
    """Published JSON Schema for a contacts.json document."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SAGAR-NETRA contacts report",
        "type": "object",
        "required": ["survey", "generated_at", "contacts"],
        "properties": {
            "survey": {"type": "string"},
            "generated_at": {"type": "string"},
            "pipeline_version": {"type": "string"},
            "contacts": {"type": "array", "items": Contact.model_json_schema()},
        },
    }
