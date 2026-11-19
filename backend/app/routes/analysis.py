from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import tempfile
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.detectors.audio import audio_detector
from app.detectors.image import image_detector
from app.detectors.video import VideoDeepfakeAnalyzer
from app.schemas import AssetResult, CaseReport, Observation
from app.services.correlation import correlate_modalities
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

            digest = hashlib.sha256()
            with temporary_path.open("rb") as evidence_file:
                for chunk in iter(lambda: evidence_file.read(1024 * 1024), b""):
                    digest.update(chunk)

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
                sha256=digest.hexdigest(),
                analysis_status=model_assessment.get("status", "inconclusive"),
                metadata=result["metadata"],
                observations=observations,
                model_assessment=model_assessment,
                video_analysis=video_analysis,
                sampled_frames=result["sampled_frames"],
            ))
        except MediaError as exc:
            raise HTTPException(status_code=422, detail=f"{filename}: {exc}") from exc
        finally:
            await upload.close()
            if temporary_path and temporary_path.exists():
                temporary_path.unlink(missing_ok=True)

    assets_for_correlation = [
        asset.model_dump() for asset in assets
    ]
    cross_modal = correlate_modalities(assets_for_correlation, pair_media)
    if context_text.strip():
        cross_modal.append({
            "type": "user_context",
            "status": "provided",
            "detail": "User-supplied context was attached to this case; it has not been independently verified.",
        })

    assessed = [
        asset for asset in assets
        if asset.model_assessment.get("status") == "model_assessed"
    ]
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

    return CaseReport(
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
            "All results require human review and original-source context.",
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
            "SHA-256 evidence fingerprint",
        ],
    }
