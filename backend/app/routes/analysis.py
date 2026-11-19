from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from app.auth import get_current_user
from app.detectors.audio import audio_detector
from app.detectors.image import image_detector
from app.detectors.video import VideoDeepfakeAnalyzer
from app.schemas import AssetResult, CaseReport, Observation
from app.services.activity import (
    get_user_heatmap_path,
    delete_user_report,
    get_user_report,
    list_user_activity,
    save_report_and_activity,
)
from app.services.correlation import correlate_modalities
from app.services.integrity import compare_sha256
from app.services.media import MediaError, inspect_media

router = APIRouter(tags=["analysis"])
MAX_FILE_BYTES = 40 * 1024 * 1024
MAX_FILES_PER_CASE = 8
video_detector = VideoDeepfakeAnalyzer(image_detector)


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
    pair_media: Annotated[bool, Form()] = False,
    sha256_digests: Annotated[str, Form()] = "[]",
    case_title: Annotated[str, Form()] = "",
    current_user: dict = Depends(get_current_user),
):
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one media file.")
    if len(files) > MAX_FILES_PER_CASE:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES_PER_CASE} files per case.")
    try:
        client_digests = json.loads(sha256_digests or "[]")
        if not isinstance(client_digests, list) or len(client_digests) not in {0, len(files)}:
            raise ValueError
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=400,
            detail="sha256_digests must be a JSON array with one digest per uploaded file.",
        ) from exc

    assets: list[AssetResult] = []
    for index, upload in enumerate(files):
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

            digest = hashlib.sha256()
            with temporary_path.open("rb") as evidence_file:
                for chunk in iter(lambda: evidence_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            server_digest = digest.hexdigest()
            client_digest = client_digests[index] if client_digests else None
            integrity_verification = compare_sha256(client_digest, server_digest)
            integrity_verification["checked_at"] = datetime.now(timezone.utc).isoformat()

            video_analysis: dict = {}
            if result["media_type"] == "image":
                model_assessment = await run_in_threadpool(image_detector.analyze_image, temporary_path, True)
            elif result["media_type"] == "video":
                video_analysis = await run_in_threadpool(video_detector.analyze, temporary_path)
                model_assessment = {
                    "status": video_analysis.get("status", "inconclusive"),
                    "model_name": video_analysis.get("model_name"),
                    "prediction": video_analysis.get("prediction", "inconclusive"),
                    "scores": video_analysis.get("scores", {}),
                    "uncertainty_band": video_analysis.get("uncertainty_band", "unknown"),
                    "model_score_type": video_analysis.get("model_score_type"),
                    "note": video_analysis.get("note", ""),
                }
            else:
                model_assessment = await run_in_threadpool(audio_detector.analyze, temporary_path)

            observations = [Observation(**item) for item in result["observations"]]
            if integrity_verification["status"] == "mismatch":
                observations.append(Observation(
                    code="EVIDENCE_HASH_MISMATCH",
                    title="Client and server SHA-256 digests differ",
                    detail="The digest calculated before upload does not match the bytes received by the API. Preserve the original file and investigate the transfer.",
                    category="integrity",
                    severity="warning",
                ))
            elif integrity_verification["status"] == "match":
                observations.append(Observation(
                    code="EVIDENCE_HASH_MATCH",
                    title="Evidence transfer integrity verified",
                    detail="The client and API independently calculated the same SHA-256 digest for the received evidence bytes.",
                    category="integrity",
                    severity="info",
                ))

            if model_assessment.get("status") == "model_assessed":
                observations.append(Observation(
                    code="MODEL_SIGNAL_AVAILABLE",
                    title="Research model produced an indicator",
                    detail=(
                        f"Model: {model_assessment.get('model_name')}. "
                        f"Signal: {model_assessment.get('prediction')}. "
                        "Scores are not calibrated forensic probabilities; human review is required."
                    ),
                    category="model",
                    severity="warning" if model_assessment.get("prediction") in {"fake_signal", "inconclusive"} else "info",
                ))
            else:
                observations.append(Observation(
                    code="MODEL_INCONCLUSIVE",
                    title="Model assessment is inconclusive",
                    detail=model_assessment.get("note", "The model did not return an assessable result."),
                    category="model",
                    severity="warning",
                ))

            assets.append(AssetResult(
                asset_id=str(uuid4()),
                filename=filename,
                media_type=result["media_type"],
                size_bytes=size,
                sha256=server_digest,
                analysis_status=model_assessment.get("status", "inconclusive"),
                metadata=result["metadata"],
                observations=observations,
                model_assessment=model_assessment,
                video_analysis=video_analysis,
                sampled_frames=result["sampled_frames"],
                integrity_verification=integrity_verification,
            ))
        except MediaError as exc:
            raise HTTPException(status_code=422, detail=f"{filename}: {exc}") from exc
        finally:
            await upload.close()
            if temporary_path and temporary_path.exists():
                temporary_path.unlink(missing_ok=True)

    assets_for_correlation = [asset.model_dump() for asset in assets]
    cross_modal = correlate_modalities(assets_for_correlation, pair_media)
    if context_text.strip():
        cross_modal.append({
            "type": "user_context",
            "status": "provided",
            "detail": "User-supplied context was attached to this case; it has not been independently verified.",
        })

    assessed = [asset for asset in assets if asset.model_assessment.get("status") == "model_assessed"]
    if assessed:
        overall_status = "model_indicators_available_human_review_required"
        summary = (
            "One or more research models produced modality-specific indicators. "
            "These outputs are not proof of authenticity; inspect source evidence and limitations."
        )
    else:
        overall_status = "inconclusive"
        summary = (
            "No modality produced an assessable model result. Technical observations may still be available; "
            "the supplied evidence remains inconclusive."
        )

    title = case_title.strip() or (
        f"Review · {assets[0].filename}" if len(assets) == 1
        else f"Investigation · {len(assets)} evidence files"
    )
    report = CaseReport(
        report_id=str(uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        overall_status=overall_status,
        summary=summary,
        assets=assets,
        cross_modal_review=cross_modal,
        limitations=[
            "Model softmax scores are not calibrated probabilities or proof of authenticity.",
            "The image classifier is intended for face-cropped images and may not generalize to all image types or generators.",
            "The audio model was trained on a limited set of speech/TTS sources; performance on unseen systems, languages, codecs and noisy recordings is uncertain.",
            "Video analysis samples a limited number of face-bearing frames; it is not a validated temporal deepfake detector.",
            "Cross-modal comparison checks separate model signals only; it does not prove that audio and video are synchronized or belong to the same source.",
            "A matching SHA-256 digest verifies the bytes received by this API against the browser's pre-upload digest; it does not prove source identity or content authenticity.",
            "All results require human review and original-source context.",
        ],
        human_review_required=True,
        case_title=title,
    )

    try:
        save_report_and_activity(
            user_uid=current_user["uid"],
            report=report.model_dump(mode="json"),
            case_title=title,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The analysis finished, but TruthLens could not persist the case record. Check backend data-directory permissions.",
        ) from exc
    return report


@router.get("/activity")
def activity_feed(
    limit: int = Query(default=20, ge=1, le=50),
    current_user: dict = Depends(get_current_user),
):
    """Return persistent activity for the signed-in Firebase UID only."""
    return {"events": list_user_activity(current_user["uid"], limit=limit)}


@router.get("/reports/{report_id}")
def saved_report(
    report_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Load a saved report and its persisted attribution images for its owner."""
    report = get_user_report(current_user["uid"], report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    return report


@router.delete("/reports/{report_id}")
def delete_saved_report(
    report_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a saved case and its heatmap artifacts for its owner."""
    if not delete_user_report(current_user["uid"], report_id):
        raise HTTPException(status_code=404, detail="Investigation not found.")
    return {"deleted": True, "report_id": report_id}


@router.get("/reports/{report_id}/heatmap")
def saved_heatmap(
    report_id: str,
    asset_id: str = Query(...),
    face_index: int = Query(default=0, ge=0, le=2),
    current_user: dict = Depends(get_current_user),
):
    """Stream one persisted model-attribution PNG after checking report ownership."""
    path = get_user_heatmap_path(current_user["uid"], report_id, asset_id, face_index)
    if path is None:
        raise HTTPException(status_code=404, detail="No saved heatmap exists for this evidence item.")
    return FileResponse(path, media_type="image/png", filename=f"truthlens-heatmap-{asset_id}-{face_index}.png")


@router.get("/me", tags=["authentication"])
def current_user_profile(current_user: dict = Depends(get_current_user)):
    """Return the verified Firebase identity associated with the bearer token."""
    return current_user


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
        "models": {
            "image": image_detector.model_name,
            "audio": audio_detector.model_name,
            "video": "sampled face frames evaluated with the image model",
        },
        "available_analysis": [
            "image face-crop classification and optional transformer patch attribution",
            "video sampled-frame classification and timestamps",
            "audio speech-window classification",
            "paired audio/video model-signal comparison when user confirms related media",
            "client-to-server SHA-256 integrity comparison",
            "user-scoped persistent report and activity history",
            "persisted Grad-CAM-style attribution previews",
        ],
    }
