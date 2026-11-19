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
from app.detectors.whole_image import whole_image_detector
from app.detectors.video import VideoDeepfakeAnalyzer
from app.schemas import AssetResult, CaseReport, Observation
from app.services.activity import (
    get_user_activity_summary,
    get_user_heatmap_path,
    delete_user_report,
    get_user_report,
    list_user_activity,
    save_report_and_activity,
    save_ocr_correction,
)
from app.services.correlation import correlate_modalities, correlate_text_evidence, correlate_timed_model_signals
from app.services.semantic_review import compare_semantic_sources
from app.services.uncertainty_assessment import build_uncertainty_assessment
from app.services.integrity import compare_sha256
from app.services.media import MediaError, inspect_media
from app.services.provenance import inspect_provenance
from app.services.ocr import extract_image_text, extract_video_frame_text, compare_ocr_engines, _normalized_edit_distance
from app.services.transcription import transcribe_audio
from app.services.audio_review import audio_review_timeline
from app.services.image_comparison import compare_images
from app.services.text_analysis import analyze_text_context
from app.services.claim_verification import verify_claims
from app.services.evidence_ledger import append_case_events, append_case_event, get_case_events, verify_case_chain, delete_case_events
from app.services.evidence_storage import store_original, resolve_original, delete_case_originals, purge_expired_originals
from app.services.advanced_forensics import temporal_video_forensics, face_identity_consistency, syncnet_lipsync, verify_c2pa
from app.services.sync_analysis import estimate_audio_video_activity_alignment
from app.services.optional_integrations import get_optional_integration_status
from app.services.investigator_review import save_finding_review, list_finding_reviews

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
    retain_originals: Annotated[bool, Form()] = False,
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
    report_id = str(uuid4())
    retained_paths: dict[str, list[Path]] = {}
    extracted_text: list[str] = []
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
            if result["media_type"] in {"video", "audio", "image"}:
                retained_paths.setdefault(result["media_type"], []).append(temporary_path)

            digest = hashlib.sha256()
            with temporary_path.open("rb") as evidence_file:
                for chunk in iter(lambda: evidence_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            server_digest = digest.hexdigest()
            client_digest = client_digests[index] if client_digests else None
            integrity_verification = compare_sha256(client_digest, server_digest)
            integrity_verification["checked_at"] = datetime.now(timezone.utc).isoformat()
            asset_id = str(uuid4())
            evidence_storage = store_original(
                temporary_path,
                uid=str(current_user["uid"]),
                report_id=report_id,
                asset_id=asset_id,
                filename=filename,
                digest=server_digest,
                retain=retain_originals,
            )

            video_analysis: dict = {}
            text_extraction: dict = {}
            provenance = inspect_provenance(temporary_path, server_digest, filename)
            provenance["content_credentials"] = await run_in_threadpool(verify_c2pa, temporary_path)
            if result["media_type"] == "image":
                text_extraction = await run_in_threadpool(extract_image_text, temporary_path)
                text_extraction["engine_comparison"] = await run_in_threadpool(compare_ocr_engines, temporary_path)
                if text_extraction.get("analysis_eligible"):
                    extracted_text.append(text_extraction.get("analysis_text", ""))
                model_assessment = await run_in_threadpool(image_detector.analyze_image, temporary_path, True)
                if model_assessment.get("status") == "inconclusive_no_face_detected":
                    whole_image = await run_in_threadpool(whole_image_detector.analyze, temporary_path)
                    model_assessment["whole_image_assessment"] = whole_image
                    if whole_image.get("status") == "model_assessed":
                        model_assessment = {
                            **whole_image,
                            "detector_path": "whole_image_no_face_detected",
                            "face_model_status": "inconclusive_no_face_detected",
                            "whole_image_assessment": whole_image,
                            "note": "No face was detected. A separate whole-image model was used; its output is a research signal, not proof.",
                        }
            elif result["media_type"] == "text":
                text_extraction = result.get("text_extraction", {})
                extracted_text.append(str(text_extraction.get("text") or ""))
                model_assessment = {
                    "status": "not_applicable",
                    "prediction": "not_applicable",
                    "model_name": None,
                    "note": "Plain-text evidence is compared as a source; no image, video or audio classifier is applicable.",
                }
            elif result["media_type"] == "video":
                text_extraction = await run_in_threadpool(extract_video_frame_text, temporary_path)
                extracted_text.extend(
                    frame.get("analysis_text", "")
                    for frame in text_extraction.get("frames", [])
                    if frame.get("analysis_eligible")
                )
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
                model_assessment["audio_review_timeline"] = await run_in_threadpool(audio_review_timeline, temporary_path)

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

            uncertainty_assessment = build_uncertainty_assessment(
                result["media_type"], model_assessment, video_analysis
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
                uncertainty_assessment=uncertainty_assessment,
                evidence_storage=evidence_storage,
            ))
        except MediaError as exc:
            for retained in retained_paths.values():
                for retained_path in retained:
                    retained_path.unlink(missing_ok=True)
            delete_case_originals(str(current_user["uid"]), report_id)
            raise HTTPException(status_code=422, detail=f"{filename}: {exc}") from exc
        except Exception:
            for retained in retained_paths.values():
                for retained_path in retained:
                    retained_path.unlink(missing_ok=True)
            delete_case_originals(str(current_user["uid"]), report_id)
            raise
        finally:
            await upload.close()
            if temporary_path and temporary_path.exists() and temporary_path not in [p for paths in retained_paths.values() for p in paths]:
                temporary_path.unlink(missing_ok=True)

    assets_for_correlation = [asset.model_dump() for asset in assets]
    cross_modal = correlate_modalities(assets_for_correlation, pair_media)
    if pair_media:
        for video_asset in assets_for_correlation:
            if video_asset.get("media_type") != "video":
                continue
            for audio_asset in assets_for_correlation:
                if audio_asset.get("media_type") == "audio":
                    cross_modal.append(correlate_timed_model_signals(video_asset, audio_asset))
    try:
        if len(retained_paths.get("image", [])) >= 2:
            cross_modal.extend(await run_in_threadpool(compare_images, retained_paths["image"]))
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
    except Exception:
        delete_case_originals(str(current_user["uid"]), report_id)
        raise
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

    combined_text = "\n\n".join([context_text, *extracted_text]).strip()
    text_analysis = analyze_text_context(combined_text)
    low_confidence_ocr = [
        {
            "filename": asset.filename,
            "confidence_level": asset.text_extraction.get("confidence_level"),
            "mean_word_confidence": asset.text_extraction.get("mean_word_confidence"),
            "note": "Raw OCR is preserved in the evidence item, but excluded from automated claim analysis because it did not pass the OCR quality gate.",
        }
        for asset in assets
        if asset.text_extraction.get("engine") == "Tesseract OCR"
        and asset.text_extraction.get("text")
        and not asset.text_extraction.get("analysis_eligible", False)
    ]
    if low_confidence_ocr:
        text_analysis["ocr_review_required"] = low_confidence_ocr
    if combined_text:
        text_analysis["claim_verification"] = await verify_claims(combined_text)
    cross_modal.extend(correlate_text_evidence(
        [asset.model_dump() for asset in assets],
        context_text,
    ))
    semantic_sources = []
    if context_text.strip():
        semantic_sources.append({
            "source_id": "investigator-context",
            "label": "Investigator-supplied claim / context",
            "source_type": "claim",
            "text": context_text.strip(),
        })
    for asset in assets:
        extraction = asset.text_extraction or {}
        pieces = [str(extraction.get("text") or "")]
        pieces.extend(str(frame.get("text") or "") for frame in extraction.get("frames", []))
        extracted = "\\n".join(piece for piece in pieces if piece).strip()
        if extracted:
            semantic_sources.append({
                "source_id": asset.asset_id,
                "label": asset.filename,
                "source_type": asset.media_type,
                "text": extracted,
            })
    text_analysis["semantic_comparison"] = await run_in_threadpool(
        compare_semantic_sources, semantic_sources
    )
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
        delete_case_originals(str(current_user["uid"]), report_id)
        delete_case_events(report_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The analysis finished, but TruthLens could not persist the case record. Check backend data-directory permissions.",
        ) from exc
    return report


@router.get("/cases/{report_id}/chain-of-custody")
def case_chain_of_custody(report_id: str, current_user: dict = Depends(get_current_user)):
    uid = str(current_user.get("uid", ""))
    report = get_user_report(uid, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    events = get_case_events(report_id)
    return {"report_id": report_id, "events": events, "verification": verify_case_chain(report_id)}


@router.get("/cases/{report_id}/chain-of-custody/verify")
def verify_case_chain_endpoint(report_id: str, current_user: dict = Depends(get_current_user)):
    if get_user_report(str(current_user.get("uid", "")), report_id) is None:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    return verify_case_chain(report_id)


@router.get("/cases/{report_id}/chain-of-custody/manifest")
def export_chain_manifest(report_id: str, current_user: dict = Depends(get_current_user)):
    report = get_user_report(str(current_user["uid"]), report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    events = get_case_events(report_id)
    return {
        "manifest_version": "1.0",
        "report_id": report_id,
        "case_title": report.get("case_title"),
        "created_at": report.get("created_at"),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "assets": [{
            "asset_id": asset.get("asset_id"),
            "filename": asset.get("filename"),
            "media_type": asset.get("media_type"),
            "size_bytes": asset.get("size_bytes"),
            "sha256": asset.get("sha256"),
            "evidence_storage": asset.get("evidence_storage"),
        } for asset in report.get("assets", [])],
        "events": events,
        "verification": verify_case_chain(report_id),
        "limitation": "Local hash-linked manifest; not independently timestamped or externally anchored.",
    }


@router.post("/cases/retention/purge-expired")
def purge_expired_evidence(current_user: dict = Depends(get_current_user)):
    return purge_expired_originals(str(current_user["uid"]))


@router.post("/ocr/benchmark")
def benchmark_ocr(payload: dict, current_user: dict = Depends(get_current_user)):
    """Score OCR output against investigator-supplied ground truth; never treat CER as authenticity."""
    reference = str(payload.get("ground_truth", ""))
    hypothesis = str(payload.get("recognized_text", ""))
    if not reference.strip() or len(reference) > 20000 or len(hypothesis) > 20000:
        raise HTTPException(status_code=400, detail="Provide non-empty ground_truth and texts no longer than 20,000 characters.")
    result = _normalized_edit_distance(reference, hypothesis)
    return {
        **result,
        "metric": "character_error_rate",
        "interpretation": "Lower CER means closer transcription to the supplied reference. Results depend on reference quality and normalization; this is not an authenticity metric.",
        "reviewer_uid": str(current_user.get("uid", "")),
    }


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


@router.delete("/reports/{report_id}")
def delete_saved_report(
    report_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a saved case and its heatmap artifacts for its owner."""
    if not delete_user_report(current_user["uid"], report_id):
        raise HTTPException(status_code=404, detail="Investigation not found.")
    delete_case_originals(str(current_user["uid"]), report_id)
    delete_case_events(report_id)
    return {"deleted": True, "report_id": report_id}



@router.get("/reports/{report_id}/assets/{asset_id}/original")
def download_original_evidence(
    report_id: str,
    asset_id: str,
    current_user: dict = Depends(get_current_user),
):
    report = get_user_report(str(current_user["uid"]), report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    asset = next((item for item in report.get("assets", []) if item.get("asset_id") == asset_id), None)
    if asset is None:
        raise HTTPException(status_code=404, detail="Evidence item not found.")
    path, state = resolve_original(str(current_user["uid"]), report_id, asset)
    if path is None:
        code = 410 if state == "expired" else 404
        raise HTTPException(status_code=code, detail=f"Original evidence is unavailable: {state}.")
    storage = asset.get("evidence_storage") or {}
    return FileResponse(
        path,
        media_type=storage.get("media_type") or "application/octet-stream",
        filename=Path(asset.get("filename") or "evidence").name,
        headers={"X-Content-Type-Options": "nosniff"},
    )


@router.put("/reports/{report_id}/assets/{asset_id}/ocr-correction")
def put_ocr_correction(
    report_id: str,
    asset_id: str,
    payload: dict,
    current_user: dict = Depends(get_current_user),
):
    """Save a human correction separately; never overwrite machine OCR or auto-verify claims."""
    corrected_text = str(payload.get("corrected_text", ""))
    if len(corrected_text) > 12000:
        raise HTTPException(status_code=400, detail="Corrected text must be 12,000 characters or fewer.")
    try:
        correction = save_ocr_correction(
            user_uid=str(current_user["uid"]),
            report_id=report_id,
            asset_id=asset_id,
            corrected_text=corrected_text,
        )
        custody_event = append_case_event(
            report_id,
            {
                "event": "investigator_ocr_correction",
                "detail": "An investigator saved a correction to extracted text.",
                "asset_id": asset_id,
                "reviewer_uid": str(current_user["uid"]),
                "corrected_text_sha256": hashlib.sha256(corrected_text.strip().encode("utf-8")).hexdigest(),
                "corrected_character_count": len(corrected_text.strip()),
                "updated_at": correction["updated_at"],
            },
            str(current_user["uid"]),
        )
        return {**correction, "custody_event": custody_event}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/reports/{report_id}/reviews")
def get_finding_reviews(report_id: str, current_user: dict = Depends(get_current_user)):
    report = get_user_report(current_user["uid"], report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    return {"report_id": report_id, "reviews": list_finding_reviews(current_user["uid"], report_id)}


@router.put("/reports/{report_id}/reviews/{finding_id}")
def put_finding_review(
    report_id: str,
    finding_id: str,
    payload: dict,
    current_user: dict = Depends(get_current_user),
):
    report = get_user_report(current_user["uid"], report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    valid_finding_ids = {
        f"{asset.get('asset_id')}:{str(finding.get('code') or index)}"
        for asset in report.get("assets", [])
        for index, finding in enumerate(asset.get("observations") or [{"code": "MODEL_ASSESSMENT"}])
    }
    if finding_id not in valid_finding_ids:
        raise HTTPException(status_code=404, detail="Finding not found in this investigation.")
    status_value = str(payload.get("status", ""))
    comment = str(payload.get("comment", ""))
    try:
        saved_review = save_finding_review(
            str(current_user["uid"]), report_id, finding_id, status_value, comment
        )
        custody_event = append_case_event(
            report_id,
            {
                "event": "investigator_finding_review",
                "detail": "An investigator saved a decision for a technical finding.",
                "finding_id": finding_id,
                "review_status": status_value,
                "reviewer_uid": str(current_user["uid"]),
                "review_comment_sha256": hashlib.sha256(comment.strip().encode("utf-8")).hexdigest(),
                "updated_at": saved_review["updated_at"],
            },
            str(current_user["uid"]),
        )
        return {**saved_review, "custody_event": custody_event}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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
        "optional_integration_status": get_optional_integration_status(),
        "advanced_forensics": {
            "temporal_video": "Optical-flow and scene-change review cues; heuristic only",
            "face_consistency": "Optional InsightFace; requires model weights",
            "lip_sync": "Optional SyncNet; requires compatible checkpoint, package and FFmpeg",
            "c2pa": "Optional c2pa-python validation; configure trust anchors",
            "claim_research": "OpenAlex scholarly candidate discovery; no automatic truth verdict",
            "evidence_ledger": "Local SQLite hash-linked event history with optional HMAC signatures; not externally anchored",
            "original_evidence": "Explicit per-case opt-in private retention with digest-checked authenticated downloads and expiry purge",
            "semantic_review": "Optional local bidirectional NLI candidate comparison; uncalibrated and not a fact verdict",
            "ocr": "Optional pytesseract and Tesseract executable",
            "transcription": "Optional ASR model configured through TRUTHLENS_ASR_MODEL",
        },
        "supported_inputs": {
            "image": [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"],
            "video": [".mp4", ".mov", ".avi", ".mkv", ".webm"],
            "audio": [".wav", ".flac", ".ogg"],
            "text": [".txt", ".md"],
        },
        "max_file_bytes": MAX_FILE_BYTES,
        "max_files_per_case": MAX_FILES_PER_CASE,
        "max_text_file_bytes": 1 * 1024 * 1024,
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
            "video 16-point deterministic two-pass temporal sampling, timestamps, frame-level score dispersion and direction counts",
            "audio speech-window classification with segment-level scores and score-dispersion review",
            "paired audio/video model-signal comparison when user confirms related media",
            "relative-time audio/video model-score profile association (not a synchronization or authenticity verdict)",
            "client-to-server SHA-256 integrity comparison",
            "user-scoped persistent report and activity history",
            "persisted Grad-CAM-style attribution previews",
            "descriptive image lighting, texture, noise and compression parameters",
            "heuristic temporal video optical-flow and scene-change timestamps",
            "optional within-video face-embedding consistency review",
            "optional C2PA Content Credentials manifest status",
            "optional OCR on images and sampled video frames",
            "optional ASR transcription when TRUTHLENS_ASR_MODEL is configured",
            "UTF-8 TXT/MD source evidence ingestion (1 MB limit)",
            "rule-based text triage and OpenAlex scholarly candidate discovery",
            "optional local bidirectional NLI sentence-pair review via TRUTHLENS_NLI_MODEL",
            "experimental audio/video activity-envelope alignment for confirmed pairs",
            "optional SyncNet synchronization review when model weights and FFmpeg are configured",
            "local hash-linked chain-of-custody event history with verification and JSON manifest endpoints",
            "optional server-secret HMAC signatures for custody events",
            "opt-in private original-evidence retention, authenticated download and expiry purge",
            "source-grouped calibration evaluation with manually approved model-specific runtime mappings",
        ],
    }
