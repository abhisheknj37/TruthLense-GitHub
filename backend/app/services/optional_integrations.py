from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path
from typing import Any


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _file_config(name: str) -> tuple[str, Path | None]:
    value = os.getenv(name, "").strip()
    if not value:
        return "not_configured", None
    path = Path(value).expanduser()
    return ("configured_file_found" if path.is_file() else "configured_file_missing"), path


def get_optional_integration_status() -> dict[str, dict[str, Any]]:
    """Report optional integrations without importing models or downloading weights."""
    result: dict[str, dict[str, Any]] = {}

    temporal_status, temporal_path = _file_config("TRUTHLENS_TEMPORAL_MODEL")
    if temporal_status == "not_configured":
        temporal_note = "No trained temporal checkpoint configured; optical-flow and scene-change cues remain heuristic."
    elif temporal_status == "configured_file_missing":
        temporal_note = "Configured checkpoint path does not exist on this backend host."
    else:
        temporal_note = "Checkpoint file exists; architecture, preprocessing, labels and evaluation provenance are not validated here."
    result["temporal_video_model"] = {
        "status": temporal_status,
        "checkpoint": temporal_path.name if temporal_path else None,
        "note": temporal_note,
    }

    face_model = os.getenv("TRUTHLENS_FACE_MODEL", "buffalo_l").strip() or "buffalo_l"
    raw_threshold = os.getenv("TRUTHLENS_FACE_SIMILARITY_REVIEW_THRESHOLD", "0.35").strip() or "0.35"
    try:
        parsed_threshold = float(raw_threshold)
        threshold_valid = 0.0 <= parsed_threshold <= 1.0
    except ValueError:
        parsed_threshold = 0.35
        threshold_valid = False
    face_threshold = parsed_threshold if threshold_valid else 0.35
    face_missing = []
    if not _module_available("insightface"):
        face_missing.append("insightface")
    if not _module_available("onnxruntime"):
        face_missing.append("onnxruntime")
    face_cache = Path.home() / ".insightface" / "models" / face_model
    result["face_consistency"] = {
        "status": "dependencies_missing" if face_missing else ("model_cache_found" if face_cache.exists() else "model_download_on_first_use"),
        "missing": face_missing,
        "model": face_model,
        "model_cache_present": face_cache.exists(),
        "review_threshold": face_threshold,
        "threshold_config_valid": threshold_valid,
        "threshold_note": None if threshold_valid else "Invalid threshold; using the safe default 0.35. Configure a number from 0.0 to 1.0.",
        "note": "Within-video similarity is a heuristic review cue, not identity recognition or proof of a face swap.",
    }

    syncnet_path_status, syncnet_path = _file_config("TRUTHLENS_SYNCNET_WEIGHTS")
    syncnet_missing = []
    if not _module_available("syncnet_python"):
        syncnet_missing.append("syncnet-python package")
    if syncnet_path_status != "configured_file_found":
        syncnet_missing.append("compatible SyncNet checkpoint (TRUTHLENS_SYNCNET_WEIGHTS)")
    if not shutil.which("ffmpeg"):
        syncnet_missing.append("FFmpeg executable on PATH")
    result["syncnet"] = {
        "status": "ready_to_attempt" if not syncnet_missing else "needs_setup",
        "missing": syncnet_missing,
        "checkpoint": syncnet_path.name if syncnet_path else None,
        "note": "A ready status means prerequisites are detected, not that a successful score is guaranteed. SyncNet measures synchronization, not manipulation probability.",
    }

    c2pa_installed = _module_available("c2pa")
    result["c2pa"] = {
        "status": "sdk_installed_validation_review_required" if c2pa_installed else "dependency_missing",
        "missing": [] if c2pa_installed else ["c2pa-python package"],
        "note": "Review SDK validation output and trust configuration; a manifest alone does not establish trusted provenance.",
    }

    ocr_missing = []
    if not _module_available("pytesseract"):
        ocr_missing.append("pytesseract Python package")
    configured_tesseract = os.getenv("TRUTHLENS_TESSERACT_CMD", "").strip()
    if configured_tesseract:
        tesseract_path = Path(configured_tesseract).expanduser()
        tesseract_found = tesseract_path.is_file()
        tesseract_display = tesseract_path.name
    else:
        tesseract_found = shutil.which("tesseract") is not None
        tesseract_display = "tesseract on PATH"
    if not tesseract_found:
        ocr_missing.append("Tesseract OCR executable")
    ocr_languages = os.getenv("TRUTHLENS_OCR_LANGUAGES", "eng").strip() or "eng"
    language_status = "not_checked"
    missing_languages: list[str] = []
    if tesseract_found and _module_available("pytesseract"):
        try:
            import pytesseract
            if configured_tesseract:
                pytesseract.pytesseract.tesseract_cmd = str(Path(configured_tesseract).expanduser())
            pytesseract.get_tesseract_version()
            installed_languages = set(pytesseract.get_languages(config=""))
            missing_languages = sorted({
                language.strip() for language in ocr_languages.split("+") if language.strip()
            } - installed_languages)
            language_status = "missing" if missing_languages else "available"
        except Exception as exc:
            language_status = "engine_error"
            ocr_missing.append(f"Tesseract check failed: {type(exc).__name__}")
    rapidocr_installed = _module_available("rapidocr_onnxruntime")
    result["ocr_second_engine"] = {
        "status": "ready_to_attempt" if rapidocr_installed else "optional_dependency_missing",
        "engine": "RapidOCR",
        "missing": [] if rapidocr_installed else ["rapidocr-onnxruntime"],
        "note": "Independent OCR comparison is available when installed; agreement is not ground truth.",
    }

    result["ocr"] = {
        "status": "ready_to_attempt" if not ocr_missing and language_status == "available" else "needs_setup",
        "missing": ocr_missing,
        "executable": tesseract_display,
        "languages_requested": ocr_languages,
        "language_status": language_status,
        "missing_languages": missing_languages,
        "note": "OCR runs on images and sampled video frames; verify extracted text against the source. Requested language traineddata is checked when Tesseract is available.",
    }

    asr_model = os.getenv("TRUTHLENS_ASR_MODEL", "").strip()
    asr_model_display = asr_model or None
    if asr_model and (
        Path(asr_model).expanduser().is_absolute()
        or asr_model.startswith(("./", "../", "~", ".\\"))
        or (len(asr_model) > 2 and asr_model[1:3] in {":\\", ":/"})
    ):
        asr_model_display = asr_model.replace("\\", "/").rsplit("/", 1)[-1]
    result["speech_transcription"] = {
        "status": "configured_model_not_loaded" if asr_model else "not_configured",
        "model": asr_model_display,
        "note": "The model is loaded on first audio analysis; this status does not confirm that remote weights are reachable or inference succeeds.",
    }

    result["claim_research"] = {
        "status": "network_required_at_request_time",
        "provider": "OpenAlex",
        "note": "Related scholarly candidates are returned for human review; no true/false verdict is generated.",
    }

    calibration_artifact = os.getenv("TRUTHLENS_CALIBRATION_ARTIFACT", "").strip()
    calibration_status = "not_configured"
    if calibration_artifact:
        from app.services.calibration_registry import load_approved_calibration
        try:
            # Capability page is global; list each modality against its configured checkpoint.
            statuses = {
                pipeline: load_approved_calibration(
                    os.getenv("TRUTHLENS_" + pipeline.upper() + "_MODEL", "") or None,
                    pipeline,
                ).get("status")
                for pipeline in ("image", "video", "audio")
            }
            calibration_status = "configured"
        except Exception:
            statuses = {pipeline: "check_failed" for pipeline in ("image", "video", "audio")}
    else:
        statuses = {pipeline: "not_configured" for pipeline in ("image", "video", "audio")}
    result["calibration"] = {
        "status": calibration_status,
        "artifact": Path(calibration_artifact).name if calibration_artifact else None,
        "pipelines": statuses,
        "note": "Runtime calibration only applies to an explicitly approved artifact with an exact model ID and modality match. Candidate artifacts are not applied.",
    }

    result["score_evaluation"] = {
        "status": "ready" if _module_available("sklearn") else "dependency_missing",
        "missing": [] if _module_available("sklearn") else ["scikit-learn package"],
        "note": "Evaluation scripts use a labelled CSV supplied by the project team and do not calibrate live model outputs automatically.",
    }

    ledger_value = os.getenv("TRUTHLENS_LEDGER_DB", "").strip()
    ledger_path = Path(ledger_value).expanduser() if ledger_value else Path(__file__).resolve().parents[2] / "data" / "evidence_ledger.sqlite3"
    result["evidence_ledger"] = {
        "status": "local_hash_linked_ledger",
        "database_name": ledger_path.name,
        "exists": ledger_path.exists(),
        "note": "Local hash-chain verification only; not externally anchored or independently timestamped.",
    }
    return result
