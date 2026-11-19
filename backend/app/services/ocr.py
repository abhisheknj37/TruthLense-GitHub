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
    return os.getenv("TRUTHLENS_OCR_LANGUAGES", "eng").strip() or "eng"


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
        missing = "TesseractNotFoundError" in type(exc).__name__
        return {
            "status": "engine_not_found" if missing else "error",
            "text": "",
            "note": "Tesseract executable not found. Install Tesseract and set TRUTHLENS_TESSERACT_CMD." if missing else f"OCR could not process this image: {type(exc).__name__}. Check that configured language traineddata is installed.",
        }


def extract_video_frame_text(path: Path, max_frames: int = 8) -> dict[str, Any]:
    try:
        import pytesseract
    except ImportError:
        return {"status": "unavailable", "frames": [], "note": "Install pytesseract and the Tesseract OCR engine to enable video-frame OCR."}
    languages = _configure_tesseract(pytesseract)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"status": "error", "frames": [], "note": "Video could not be opened for OCR."}
    frames = []
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        indexes = np.linspace(0, max(0, count - 1), min(max_frames, count), dtype=int) if count else []
        for index in indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = _prepare_image(Image.fromarray(rgb))
            text = pytesseract.image_to_string(image, lang=languages, config="--oem 3 --psm 11").strip()
            if text:
                frames.append({
                    "timestamp_seconds": round(float(index) / fps, 3) if fps else None,
                    "frame_index": int(index),
                    "text": text[:3000],
                })
    except Exception as exc:
        missing = "TesseractNotFoundError" in type(exc).__name__
        return {
            "status": "engine_not_found" if missing else "error",
            "frames": frames,
            "note": "Tesseract executable not found; configure TRUTHLENS_TESSERACT_CMD." if missing else f"Video OCR stopped: {type(exc).__name__}. Check installed language traineddata.",
        }
    finally:
        cap.release()
    return {
        "status": "completed",
        "frames": frames,
        "languages": languages,
        "note": "Sampled-frame OCR only; not full-video transcription or claim verification. Non-English languages require installed Tesseract traineddata.",
    }
