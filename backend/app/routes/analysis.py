from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.detectors.base import UnconfiguredDetector
from app.schemas import AssetResult, CaseReport, ModelStatus, Observation
from app.services.media import MediaError, inspect_media, media_type_for

router = APIRouter(tags=["analysis"])
MAX_FILE_BYTES = 40 * 1024 * 1024
MAX_FILES_PER_CASE = 8
detector = UnconfiguredDetector()


async def save_upload(upload: UploadFile, destination: Path) -> int:
    size = 0
    with destination.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_FILE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"{upload.filename}: file exceeds the 40 MB limit.",
                )
            output.write(chunk)
    return size


@router.post("/analyze", response_model=CaseReport)
async def analyze_case(
    files: Annotated[list[UploadFile], File(description="One or more image, video or audio files")],
    context_text: Annotated[str, Form()] = "",
):
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one media file.")
    if len(files) > MAX_FILES_PER_CASE:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES_PER_CASE} files per case.")

    assets: list[AssetResult] = []
    for upload in files:
        filename = Path(upload.filename or "upload").name
        suffix = Path(filename).suffix.lower()
        if not suffix:
            raise HTTPException(status_code=400, detail=f"{filename}: file extension is required.")

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                temporary_path = Path(temp.name)
            size = await save_upload(upload, temporary_path)
            result = inspect_media(temporary_path)
            model_result = detector.analyze(temporary_path)

            assets.append(AssetResult(
                asset_id=str(uuid4()),
                filename=filename,
                media_type=result["media_type"],
                size_bytes=size,
                analysis_status="metadata_only",
                metadata=result["metadata"],
                observations=[Observation(**item) for item in result["observations"]],
                model_assessment=ModelStatus(
                    status=model_result.status,
                    model_name=model_result.model_name,
                    note=model_result.note,
                ),
                sampled_frames=result["sampled_frames"],
            ))
        except MediaError as exc:
            raise HTTPException(status_code=422, detail=f"{filename}: {exc}") from exc
        finally:
            await upload.close()
            if temporary_path and temporary_path.exists():
                temporary_path.unlink(missing_ok=True)

    types = {asset.media_type for asset in assets}
    cross_modal: list[dict[str, str]] = []
    if "video" in types and "audio" in types:
        cross_modal.append({
            "type": "paired_media_review",
            "status": "human_review_required",
            "detail": "Video and separate audio assets were supplied. Automatic lip-sync or identity correlation is not configured in this MVP.",
        })
    if context_text.strip():
        cross_modal.append({
            "type": "user_context",
            "status": "provided",
            "detail": "User-supplied context was attached to this case; it has not been independently verified.",
        })

    return CaseReport(
        report_id=str(uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        overall_status="inconclusive_model_not_configured",
        summary=(
            "Technical media metadata was inspected. No validated authenticity model is connected, "
            "so TruthLens has not classified the supplied media as real or manipulated."
        ),
        assets=assets,
        cross_modal_review=cross_modal,
        limitations=[
            "No trained or validated deepfake detector is connected.",
            "Metadata presence or absence is not proof of manipulation or authenticity.",
            "Video frame sampling does not establish temporal authenticity.",
            "Audio signal statistics do not establish whether speech is synthetic.",
            "All findings require human review and original-source context.",
        ],
        human_review_required=True,
    )


@router.get("/capabilities")
def capabilities():
    return {
        "supported_inputs": {
            "image": [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"],
            "video": [".mp4", ".mov", ".avi", ".mkv", ".webm"],
            "audio": [".wav", ".flac", ".ogg"],
        },
        "max_file_bytes": MAX_FILE_BYTES,
        "max_files_per_case": MAX_FILES_PER_CASE,
        "authenticity_models": "not_configured",
        "available_analysis": [
            "image metadata inspection",
            "video structure and representative frame sampling",
            "audio signal metadata and level statistics",
            "case-level human-review notes",
        ],
    }
