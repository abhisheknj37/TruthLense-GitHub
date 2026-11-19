from typing import Any
from pydantic import BaseModel, Field


class Observation(BaseModel):
    code: str
    title: str
    detail: str
    category: str = "metadata"
    severity: str = "info"
    timestamp_seconds: float | None = None


class ModelStatus(BaseModel):
    status: str = "not_configured"
    model_name: str | None = None
    note: str = "No validated authenticity model is configured."


class AssetResult(BaseModel):
    asset_id: str
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    analysis_status: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    observations: list[Observation] = Field(default_factory=list)
    model_assessment: ModelStatus = Field(default_factory=ModelStatus)
    sampled_frames: list[dict[str, Any]] = Field(default_factory=list)


class CaseReport(BaseModel):
    report_id: str
    created_at: str
    overall_status: str
    summary: str
    assets: list[AssetResult]
    cross_modal_review: list[dict[str, str]] = Field(default_factory=list)
    limitations: list[str]
    human_review_required: bool = True
