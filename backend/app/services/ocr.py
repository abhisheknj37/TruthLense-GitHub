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
    # Fail early with a precise status instead of returning a generic OCR error.
    pytesseract.get_tesseract_version()
    installed = set(pytesseract.get_languages(config=""))
    requested = {language.strip() for language in languages.split("+") if language.strip()}
    missing = sorted(requested - installed)
    if missing:
        raise RuntimeError("Missing Tesseract language traineddata: " + ", ".join(missing))
    return languages


def _prepare_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("L")
    if max(image.size) < 1800:
        scale = min(3.0, 1800 / max(image.size))
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    image = ImageEnhance.Contrast(image).enhance(1.35)
    return image


def extract_image_text(path: Path) -> dict[str, Any]:
    """Optional multilingual OCR; languages must be installed in Tesseract."""
    try:
        import pytesseract
    except ImportError:
        return {"status": "unavailable", "text": "", "note": "Install pytesseract and the Tesseract OCR engine to enable OCR."}
    try:
        languages = _configure_tesseract(pytesseract)
        with Image.open(path) as source:
            image = _prepare_image(source)
            text = pytesseract.image_to_string(image, lang=languages, config="--oem 3 --psm 11").strip()
        return {
            "status": "completed" if text else "completed_no_text",
            "text": text[:12000],
            "engine": "Tesseract OCR",
            "languages": languages,
            "character_count": len(text),
            "note": "OCR is best-effort extraction. Non-English languages require matching Tesseract traineddata; verify extracted text against the original.",
        }
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
            image = _prepare_image(Image.fromarray(rgb))
            text = pytesseract.image_to_string(image, lang=languages, config="--oem 3 --psm 11").strip()
            if text:
                frames.append({
                    "timestamp_seconds": sample["timestamp_seconds"],
                    "frame_index": sample["frame_index"],
                    "text": text[:3000],
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
            "note": "Sequentially decoded sampled-frame OCR only; not full-video transcription or claim verification. Non-English languages require installed Tesseract traineddata.",
        }
    except Exception as exc:
        missing = "TesseractNotFoundError" in type(exc).__name__ or "TesseractNotFoundError" in str(exc)
        language_missing = "Missing Tesseract language traineddata:" in str(exc)
        return {
            "status": "engine_not_found" if missing else "language_data_missing" if language_missing else "error",
            "frames": [],
            "note": "Tesseract executable not found; configure TRUTHLENS_TESSERACT_CMD." if missing else str(exc) if language_missing else f"Video OCR could not complete: {type(exc).__name__}. Check the video decoder and OCR configuration.",
        }
