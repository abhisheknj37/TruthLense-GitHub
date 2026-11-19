from typing import Any
from pydantic import BaseModel, Field


class Observation(BaseModel):
    code: str
    title: str
    detail: str
    category: str = "metadata"
    severity: str = "info"
    timestamp_seconds: float | None = None


class AssetResult(BaseModel):
    asset_id: str
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    analysis_status: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    observations: list[Observation] = Field(default_factory=list)
    model_assessment: dict[str, Any] = Field(default_factory=dict)
    video_analysis: dict[str, Any] = Field(default_factory=dict)
    sampled_frames: list[dict[str, Any]] = Field(default_factory=list)
    integrity_verification: dict[str, Any] = Field(default_factory=dict)
    forensic_parameters: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    text_extraction: dict[str, Any] = Field(default_factory=dict)


class CaseReport(BaseModel):
    report_id: str
    created_at: str
    overall_status: str
    summary: str
    assets: list[AssetResult]
    cross_modal_review: list[dict[str, Any]] = Field(default_factory=list)
    text_analysis: dict[str, Any] = Field(default_factory=dict)
    evidence_events: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str]
    human_review_required: bool = True
    case_title: str | None = None
