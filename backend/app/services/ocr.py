from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


def _configure_tesseract(pytesseract) -> str:
    configured_path = os.getenv("TRUTHLENS_TESSERACT_CMD", "").strip()
    if configured_path:
        pytesseract.pytesseract.tesseract_cmd = str(Path(configured_path).expanduser())
    languages = os.getenv("TRUTHLENS_OCR_LANGUAGES", "eng").strip() or "eng"
    pytesseract.get_tesseract_version()
    installed = set(pytesseract.get_languages(config=""))
    requested = {language.strip() for language in languages.split("+") if language.strip()}
    missing = sorted(requested - installed)
    if missing:
        raise RuntimeError("Missing Tesseract language traineddata: " + ", ".join(missing))
    return languages


def _deskew_image(image: Image.Image) -> tuple[Image.Image, float]:
    """Correct small text-line rotations; avoid aggressive transforms on sparse images."""
    gray = np.asarray(image.convert("L"))
    if gray.size == 0:
        return image, 0.0
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    points = cv2.findNonZero(mask)
    if points is None or len(points) < 20:
        return image, 0.0
    angle = float(cv2.minAreaRect(points)[-1])
    angle = -(90.0 + angle) if angle < -45.0 else -angle
    if abs(angle) < 0.35 or abs(angle) > 12.0:
        return image, 0.0
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        gray, matrix, (width, height), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(rotated), round(angle, 2)


def _prepare_image(image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    image = ImageOps.exif_transpose(image).convert("L")
    original_size = image.size
    if max(image.size) < 1800:
        scale = min(3.0, 1800 / max(image.size))
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
    image, angle = _deskew_image(image)
    array = np.asarray(image)
    denoised = cv2.medianBlur(array, 3) if min(array.shape) >= 3 else array
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    prepared = Image.fromarray(enhanced)
    return prepared, {
        "original_dimensions": [int(original_size[0]), int(original_size[1])],
        "processed_dimensions": [int(prepared.width), int(prepared.height)],
        "deskew_correction_degrees": angle,
        "preprocessing": ["EXIF orientation", "grayscale", "upscale", "deskew", "median denoise", "CLAHE contrast"],
    }


def _extract_with_confidence(pytesseract, image: Image.Image, languages: str) -> dict[str, Any]:
    """Compare two layout modes and retain the text with the strongest word confidence."""
    candidates = []
    for psm in (6, 11):
        config = f"--oem 3 --psm {psm}"
        data = pytesseract.image_to_data(
            image, lang=languages, config=config,
            output_type=pytesseract.Output.DICT,
        )
        words = []
        confidences = []
        for token, raw_conf in zip(data.get("text", []), data.get("conf", [])):
            token = str(token).strip()
            try:
                confidence = float(raw_conf)
            except (TypeError, ValueError):
                continue
            if token and confidence >= 0:
                words.append(token)
                confidences.append(confidence)
        text = " ".join(words).strip()
        confidence = round(float(np.mean(confidences)), 1) if confidences else None
        candidates.append({
            "text": text,
            "mean_word_confidence": confidence,
            "word_count": len(words),
            "page_segmentation_mode": psm,
        })
    # Prefer confidence, then readable word count; empty candidates sort last.
    selected = max(
        candidates,
        key=lambda item: (
            item["mean_word_confidence"] if item["mean_word_confidence"] is not None else -1,
            item["word_count"],
        ),
    )
    return selected


def _ocr_note(confidence: float | None) -> tuple[str, str]:
    if confidence is None:
        return "unavailable", "No word-level confidence was returned by Tesseract; verify the image manually."
    if confidence < 45:
        return "low", "Low OCR confidence. Treat extracted text as unreliable and verify every character against the original."
    if confidence < 70:
        return "moderate", "Moderate OCR confidence. Review the extracted text against the original image."
    return "high", "OCR confidence is a Tesseract recognition estimate, not a probability that the text or media is authentic."


def _run_ocr(pytesseract, image: Image.Image, languages: str) -> dict[str, Any]:
    prepared, preprocessing = _prepare_image(image)
    result = _extract_with_confidence(pytesseract, prepared, languages)
    confidence = result["mean_word_confidence"]
    confidence_level, confidence_note = _ocr_note(confidence)
    text = result["text"][:12000]
    return {
        "status": "completed" if text else "completed_no_text",
        "text": text,
        "engine": "Tesseract OCR",
        "languages": languages,
        "character_count": len(text),
        "word_count": result["word_count"],
        "mean_word_confidence": confidence,
        "confidence_level": confidence_level,
        "page_segmentation_mode": result["page_segmentation_mode"],
        "preprocessing": preprocessing,
        "note": confidence_note,
    }


def extract_image_text(path: Path) -> dict[str, Any]:
    """OCR with conservative preprocessing and explicit word-confidence reporting."""
    try:
        import pytesseract
    except ImportError:
        return {"status": "unavailable", "text": "", "note": "Install pytesseract and the Tesseract OCR engine to enable OCR."}
    try:
        languages = _configure_tesseract(pytesseract)
        with Image.open(path) as source:
            return _run_ocr(pytesseract, source.copy(), languages)
    except Exception as exc:
        missing = "TesseractNotFoundError" in type(exc).__name__ or "TesseractNotFoundError" in str(exc)
        language_missing = "Missing Tesseract language traineddata:" in str(exc)
        return {
            "status": "engine_not_found" if missing else "language_data_missing" if language_missing else "error",
            "text": "",
            "note": "Tesseract executable not found. Install Tesseract and set TRUTHLENS_TESSERACT_CMD." if missing else str(exc) if language_missing else f"OCR could not process this image: {type(exc).__name__}. Check the image and OCR configuration.",
        }


def extract_video_frame_text(path: Path, max_frames: int = 8) -> dict[str, Any]:
    try:
        import pytesseract
    except ImportError:
        return {"status": "unavailable", "frames": [], "note": "Install pytesseract and the Tesseract OCR engine to enable video-frame OCR."}
    try:
        from app.services.media import sample_video_frames
        languages = _configure_tesseract(pytesseract)
        decoded = sample_video_frames(path, max_frames)
        frames = []
        for sample in decoded["sampled_frames"]:
            rgb = cv2.cvtColor(sample["frame"], cv2.COLOR_BGR2RGB)
            result = _run_ocr(pytesseract, Image.fromarray(rgb), languages)
            if result["text"]:
                frames.append({
                    "timestamp_seconds": sample["timestamp_seconds"],
                    "frame_index": sample["frame_index"],
                    **result,
                    "text": result["text"][:3000],
                })
        if decoded["frame_count"] == 0:
            return {
                "status": "insufficient_video_frames",
                "frames": [],
                "note": "The video opened, but sequential decoding returned no readable frames.",
            }
        return {
            "status": "completed" if frames else "completed_no_text",
            "frames": frames,
            "languages": languages,
            "decoded_frame_count": decoded["frame_count"],
            "sampled_frame_count": len(decoded["sampled_frames"]),
            "note": "Sequentially decoded sampled-frame OCR only; not full-video transcription or claim verification. Per-frame confidence is a Tesseract recognition estimate, not a media-authenticity score.",
        }
    except Exception as exc:
        missing = "TesseractNotFoundError" in type(exc).__name__ or "TesseractNotFoundError" in str(exc)
        language_missing = "Missing Tesseract language traineddata:" in str(exc)
        return {
            "status": "engine_not_found" if missing else "language_data_missing" if language_missing else "error",
            "frames": [],
            "note": "Tesseract executable not found; configure TRUTHLENS_TESSERACT_CMD." if missing else str(exc) if language_missing else f"Video OCR could not complete: {type(exc).__name__}. Check the video decoder and OCR configuration.",
        }
