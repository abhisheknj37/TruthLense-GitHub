from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import mimetypes
from pathlib import Path
import tempfile
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from app.auth import get_current_user
from app.detectors.audio import audio_detector
from app.detectors.image import image_detector
from app.detectors.video import VideoDeepfakeAnalyzer
from app.schemas import AssetResult, CaseReport, Observation
from app.services.activity import (
    get_user_activity_summary,
    get_user_heatmap_bytes,
    get_user_original_info,
    delete_user_report,
    get_user_report,
    list_user_activity,
    save_report_and_activity,
    save_user_report_review,
)
from app.services.correlation import correlate_modalities
from app.services.integrity import compare_sha256
from app.services.media import MediaError, inspect_media
from app.services.provenance import inspect_provenance
from app.services.ocr import extract_image_text, extract_video_frame_text
from app.services.transcription import transcribe_audio
from app.services.text_analysis import analyze_text_context, compare_text_sources
from app.services.claim_verification import verify_claims
from app.services.evidence_ledger import append_case_events, get_case_events, verify_case_chain
from app.services.advanced_forensics import temporal_video_forensics, face_identity_consistency, syncnet_lipsync, verify_c2pa
from app.services.sync_analysis import estimate_audio_video_activity_alignment
from app.services.optional_integrations import get_optional_integration_status
from app.services.database import is_postgres_enabled
from app.services.storage import upload_original_media, download_original_media, storage_is_configured

router = APIRouter(tags=["analysis"])
MAX_FILE_BYTES = 40 * 1024 * 1024
MAX_FILES_PER_CASE = 8
REVIEW_DECISIONS = {
    "supports_no_obvious_manipulation",
    "manipulation_suspected",
    "inconclusive",
    "escalate_for_more_evidence",
}
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
    retained_paths: dict[str, list[Path]] = {}
    extracted_text: list[str] = []
    text_sources: list[dict] = []
    if context_text.strip():
        text_sources.append({
            "source_id": "investigator-context",
            "source_type": "user_supplied_claim_or_caption",
            "label": "Investigator-supplied claim / caption / context",
            "filename": None,
            "text": context_text.strip()[:20000],
        })
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
            if result["media_type"] in {"video", "audio"}:
                retained_paths.setdefault(result["media_type"], []).append(temporary_path)

            digest = hashlib.sha256()
            with temporary_path.open("rb") as evidence_file:
                for chunk in iter(lambda: evidence_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            server_digest = digest.hexdigest()
            client_digest = client_digests[index] if client_digests else None
            integrity_verification = compare_sha256(client_digest, server_digest)
            integrity_verification["checked_at"] = datetime.now(timezone.utc).isoformat()

            video_analysis: dict = {}
            text_extraction: dict = {}
            provenance = inspect_provenance(temporary_path, server_digest, filename)
            provenance["content_credentials"] = await run_in_threadpool(verify_c2pa, temporary_path)
            if result["media_type"] == "image":
                text_extraction = await run_in_threadpool(extract_image_text, temporary_path)
                if text_extraction.get("text"):
                    extracted_text.append(text_extraction["text"])
                model_assessment = await run_in_threadpool(image_detector.analyze_image, temporary_path, True)
            elif result["media_type"] == "video":
                text_extraction = await run_in_threadpool(extract_video_frame_text, temporary_path)
                extracted_text.extend(frame.get("text", "") for frame in text_extraction.get("frames", []))
                video_analysis = await run_in_threadpool(video_detector.analyze, temporary_path)
                video_analysis["temporal_forensics"] = await run_in_threadpool(temporal_video_forensics, temporary_path)
                video_analysis["face_identity_consistency"] = await run_in_threadpool(face_identity_consistency, temporary_path)
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
                text_extraction = await run_in_threadpool(transcribe_audio, temporary_path)
                if text_extraction.get("text"):
                    extracted_text.append(text_extraction["text"])
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

            asset_id = str(uuid4())
            if result["media_type"] == "image":
                source_text = str(text_extraction.get("text") or "").strip()
                source_type = "image_ocr_caption"
                source_label = f"Visible text in {filename}"
            elif result["media_type"] == "video":
                frame_lines = [
                    f"[{frame.get('timestamp_seconds', 'unknown')}s] {frame.get('text', '').strip()}"
                    for frame in text_extraction.get("frames", [])
                    if str(frame.get("text") or "").strip()
                ]
                source_text = "\\n".join(frame_lines)
                source_type = "video_frame_ocr_caption"
                source_label = f"On-screen text in {filename}"
            else:
                source_text = str(text_extraction.get("text") or "").strip()
                source_type = "audio_transcript"
                source_label = f"Speech transcript from {filename}"
            if source_text:
                text_sources.append({
                    "source_id": asset_id,
                    "source_type": source_type,
                    "label": source_label,
                    "filename": filename,
                    "text": source_text[:20000],
                })
            storage_path = upload_original_media(
                temporary_path,
                owner_uid=str(current_user["uid"]),
                asset_id=asset_id,
                filename=filename,
                content_type=(upload.content_type if upload.content_type and upload.content_type != "application/octet-stream"
                              else mimetypes.guess_type(filename)[0] or "application/octet-stream"),
            )
            assets.append(AssetResult(
                asset_id=asset_id,
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
                forensic_parameters=result.get("forensic_parameters", {}),
                provenance=provenance,
                text_extraction=text_extraction,
                storage_path=storage_path,
            ))
        except MediaError as exc:
            for retained in retained_paths.values():
                for retained_path in retained:
                    retained_path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=f"{filename}: {exc}") from exc
        except Exception:
            for retained in retained_paths.values():
                for retained_path in retained:
                    retained_path.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()
            if temporary_path and temporary_path.exists() and temporary_path not in [p for paths in retained_paths.values() for p in paths]:
                temporary_path.unlink(missing_ok=True)

    assets_for_correlation = [asset.model_dump() for asset in assets]
    cross_modal = correlate_modalities(assets_for_correlation, pair_media)
    try:
        if pair_media and retained_paths.get("video") and retained_paths.get("audio"):
            cross_modal.append({
                "type": "experimental_audio_video_activity_alignment",
                **await run_in_threadpool(estimate_audio_video_activity_alignment, retained_paths["video"][0], retained_paths["audio"][0]),
            })
            cross_modal.append({
                "type": "ai_lip_sync",
                **await run_in_threadpool(syncnet_lipsync, retained_paths["video"][0], retained_paths["audio"][0]),
            })
        elif retained_paths.get("video"):
            cross_modal.append({
                "type": "embedded_video_lip_sync",
                **await run_in_threadpool(syncnet_lipsync, retained_paths["video"][0]),
            })
    finally:
        for paths in retained_paths.values():
            for retained_path in paths:
                retained_path.unlink(missing_ok=True)

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

    combined_text = "\n\n".join(source["text"] for source in text_sources).strip()
    text_analysis = analyze_text_context(combined_text)
    text_analysis["source_comparison"] = compare_text_sources(text_sources)
    if combined_text:
        text_analysis["claim_verification"] = await verify_claims(combined_text)
    report_id = str(uuid4())
    evidence_events = append_case_events(
        report_id,
        [asset.model_dump() for asset in assets],
        str(current_user.get("uid", "unknown")),
    )

    title = case_title.strip() or (
        f"Review · {assets[0].filename}" if len(assets) == 1
        else f"Investigation · {len(assets)} evidence files"
    )
    report = CaseReport(
        report_id=report_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        overall_status=overall_status,
        summary=summary,
        assets=assets,
        cross_modal_review=cross_modal,
        text_analysis=text_analysis,
        evidence_events=evidence_events,
        limitations=[
            "Model softmax scores are not calibrated probabilities or proof of authenticity.",
            "The image classifier is intended for face-cropped images and may not generalize to all image types or generators.",
            "The audio model was trained on a limited set of speech/TTS sources; performance on unseen systems, languages, codecs and noisy recordings is uncertain.",
            "Video analysis samples a limited number of face-bearing frames; it is not a validated temporal deepfake detector.",
            "Cross-modal comparison checks separate model signals only; it does not prove that audio and video are synchronized or belong to the same source.",
            "A matching SHA-256 digest verifies received bytes against the browser's pre-upload digest; it does not prove source identity or content authenticity.",
            "Image forensic parameters are descriptive lighting, texture, noise and compression measurements, not validated manipulation detectors.",
            "OCR requires optional pytesseract and the Tesseract system executable; text extraction may contain errors.",
            "Speech transcription requires TRUTHLENS_ASR_MODEL and a compatible ASR checkpoint; transcription may contain errors.",
            "Text analysis is rule-based triage, not fact-checking. OpenAlex candidate sources are not proof that a claim is supported.",
            "Optical-flow temporal indicators are heuristic review cues, not a trained temporal deepfake classifier.",
            "Face consistency requires optional InsightFace and model weights; embeddings compare faces within a clip and do not identify people.",
            "SyncNet requires its optional package, compatible checkpoint and FFmpeg; its output is a synchronization signal, not manipulation probability.",
            "C2PA validation requires the optional c2pa-python SDK and configured trust anchors; manifest presence is not proof of authenticity.",
            "The local SQLite evidence ledger is hash-linked and verifiable but lacks external anchoring, trusted timestamps and protection against full database replacement.",
            "Coarse audio/video activity alignment is experimental; it is not phoneme-to-viseme lip-sync verification.",
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


@router.get("/cases/{report_id}/chain-of-custody")
def case_chain_of_custody(report_id: str, current_user: dict = Depends(get_current_user)):
    uid = str(current_user.get("uid", ""))
    events = get_case_events(report_id, uid)
    if events and any(event.get("owner_uid") not in (None, uid) for event in events):
        raise HTTPException(status_code=403, detail="You do not have access to this evidence ledger.")
    return {"report_id": report_id, "events": events, "verification": verify_case_chain(report_id, uid)}


@router.get("/cases/{report_id}/chain-of-custody/verify")
def verify_case_chain_endpoint(report_id: str, current_user: dict = Depends(get_current_user)):
    uid = str(current_user.get("uid", ""))
    events = get_case_events(report_id, uid)
    if events and any(event.get("owner_uid") not in (None, uid) for event in events):
        raise HTTPException(status_code=403, detail="You do not have access to this evidence ledger.")
    return verify_case_chain(report_id, uid)


@router.post("/claims/verify")
async def verify_claims_endpoint(payload: dict, current_user: dict = Depends(get_current_user)):
    text = str(payload.get("text", "")).strip()
    if not text or len(text) > 20000:
        raise HTTPException(status_code=400, detail="Provide claim text between 1 and 20,000 characters.")
    return await verify_claims(text)


@router.get("/activity")
def activity_feed(
    limit: int = Query(default=20, ge=1, le=50),
    current_user: dict = Depends(get_current_user),
):
    """Return persistent activity for the signed-in Firebase UID only."""
    return {
        "events": list_user_activity(current_user["uid"], limit=limit),
        "summary": get_user_activity_summary(current_user["uid"]),
        "refresh_interval_seconds": 8,
    }


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


@router.post("/reports/{report_id}/review")
def submit_analyst_review(
    report_id: str,
    payload: dict,
    current_user: dict = Depends(get_current_user),
):
    """Persist a human assessment separately from automated model indicators."""
    decision = str(payload.get("decision") or "").strip()
    notes = str(payload.get("notes") or "").strip()
    reviewed_asset_ids = payload.get("reviewed_asset_ids", [])
    if decision not in REVIEW_DECISIONS:
        raise HTTPException(status_code=400, detail="Choose a supported analyst review decision.")
    if len(notes) < 5 or len(notes) > 5000:
        raise HTTPException(status_code=400, detail="Analyst notes must contain 5 to 5,000 characters.")
    if not isinstance(reviewed_asset_ids, list) or any(not isinstance(item, str) for item in reviewed_asset_ids):
        raise HTTPException(status_code=400, detail="reviewed_asset_ids must be an array of asset IDs.")

    report = get_user_report(current_user["uid"], report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    available_ids = {str(asset.get("asset_id")) for asset in report.get("assets", [])}
    if not set(reviewed_asset_ids).issubset(available_ids):
        raise HTTPException(status_code=400, detail="One or more reviewed asset IDs do not belong to this case.")

    review = {
        "status": "completed",
        "decision": decision,
        "notes": notes,
        "reviewed_asset_ids": sorted(set(reviewed_asset_ids)),
        "reviewed_by_uid": str(current_user.get("uid", "")),
        "reviewer": str(current_user.get("email") or current_user.get("name") or "Authenticated analyst"),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "scope_note": "This is a human assessment recorded for the case. It does not convert model indicators into proof or certify authenticity.",
    }
    if not save_user_report_review(current_user["uid"], report_id, review):
        raise HTTPException(status_code=404, detail="Investigation not found.")
    updated = get_user_report(current_user["uid"], report_id)
    return updated


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
    image_data = get_user_heatmap_bytes(current_user["uid"], report_id, asset_id, face_index)
    if image_data is None:
        raise HTTPException(status_code=404, detail="No saved heatmap exists for this evidence item.")
    return Response(content=image_data, media_type="image/png",
                    headers={"Content-Disposition": 'inline; filename="truthlens-heatmap.png"'})


@router.get("/reports/{report_id}/assets/{asset_id}/original")
def saved_original(report_id: str, asset_id: str, current_user: dict = Depends(get_current_user)):
    """Stream original evidence only after checking the Firebase UID owner."""
    info = get_user_original_info(current_user["uid"], report_id, asset_id)
    if info is None:
        raise HTTPException(status_code=404, detail="No retained original evidence exists for this item.")
    try:
        content = download_original_media(info["storage_path"])
    except Exception as exc:
        raise HTTPException(status_code=502, detail="The private evidence object could not be retrieved.") from exc
    safe_name = Path(info["filename"]).name.replace('"', "").replace("\\r", "").replace("\\n", "")
    media_type = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
    return Response(content=content, media_type=media_type,
                    headers={"Content-Disposition": f'attachment; filename="{safe_name}"'})


@router.get("/me", tags=["authentication"])
def current_user_profile(current_user: dict = Depends(get_current_user)):
    """Return the verified Firebase identity associated with the bearer token."""
    return current_user


@router.get("/capabilities")
def capabilities():
    return {
        "optional_integration_status": get_optional_integration_status(),
        "persistence": {
            "database": "supabase_postgresql" if is_postgres_enabled() else "local_sqlite",
            "original_evidence_storage": "supabase_private_bucket" if storage_is_configured() else "not_configured",
        },
        "advanced_forensics": {
            "temporal_video": "Optical-flow and scene-change review cues; heuristic only",
            "face_consistency": "Optional InsightFace; requires model weights",
            "lip_sync": "Optional SyncNet; requires compatible checkpoint, package and FFmpeg",
            "c2pa": "Optional c2pa-python validation; configure trust anchors",
            "claim_research": "OpenAlex scholarly candidate discovery; no automatic truth verdict",
            "evidence_ledger": "Supabase PostgreSQL hash-linked event history when configured; no external timestamp anchoring",
            "ocr": "Optional pytesseract and Tesseract executable",
            "transcription": "Optional ASR model configured through TRUTHLENS_ASR_MODEL",
        },
        "supported_inputs": {
            "image": [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"],
            "video": [".mp4", ".mov", ".avi", ".mkv", ".webm"],
            "audio": [".wav", ".flac", ".ogg"],
        },
        "max_file_bytes": MAX_FILE_BYTES,
        "max_files_per_case": MAX_FILES_PER_CASE,
        "models": {
            "image": image_detector.model_name,
            "audio": f"Wav2Vec2 ({audio_detector.model_name})",
            "video": "sampled face frames evaluated with the image model",
        },
        "model_checkpoints": {
            "image": image_detector.model_name,
            "audio": audio_detector.model_name,
        },
        "available_analysis": [
            "image face-crop classification and optional transformer patch attribution",
            "video sampled-frame classification and timestamps",
            "audio speech-window classification",
            "paired audio/video model-signal comparison when user confirms related media",
            "client-to-server SHA-256 integrity comparison",
            "user-scoped persistent report and activity history",
            "persisted Grad-CAM-style attribution previews",
            "descriptive image lighting, texture, noise and compression parameters",
            "heuristic temporal video optical-flow and scene-change timestamps",
            "optional within-video face-embedding consistency review",
            "optional C2PA Content Credentials manifest status",
            "optional OCR on images and sampled video frames",
            "optional ASR transcription when TRUTHLENS_ASR_MODEL is configured",
            "rule-based text triage and OpenAlex scholarly candidate discovery",
            "experimental audio/video activity-envelope alignment for confirmed pairs",
            "optional SyncNet synchronization review when model weights and FFmpeg are configured",
            "Supabase PostgreSQL hash-linked chain-of-custody events when configured, with local SQLite fallback",
            "private original-evidence retention in the truthlens-evidence Storage bucket when configured",
        ],
    }
