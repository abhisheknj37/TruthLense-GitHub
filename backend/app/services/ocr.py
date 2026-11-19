from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _configure_tesseract(pytesseract) -> None:
    configured_path = os.getenv("TRUTHLENS_TESSERACT_CMD", "").strip()
    if configured_path:
        pytesseract.pytesseract.tesseract_cmd = str(Path(configured_path).expanduser())


def extract_image_text(path: Path) -> dict[str, Any]:
    """Optional OCR using pytesseract and a local Tesseract executable."""
    try:
        import pytesseract
    except ImportError:
        return {"status": "unavailable", "text": "", "note": "Install pytesseract and the Tesseract OCR engine to enable OCR."}
    try:
        _configure_tesseract(pytesseract)
        with Image.open(path) as image:
            text = pytesseract.image_to_string(image.convert('RGB'), config='--psm 11').strip()
        return {"status": "completed" if text else "completed_no_text", "text": text[:12000], "engine": "Tesseract OCR", "character_count": len(text), "note": "OCR is best-effort extraction; verify against the original image."}
    except Exception as exc:
        return {"status": "engine_not_found" if "TesseractNotFoundError" in type(exc).__name__ else "error", "text": "", "note": "Tesseract executable not found. Install Tesseract and set TRUTHLENS_TESSERACT_CMD." if "TesseractNotFoundError" in type(exc).__name__ else f"OCR could not process this image: {type(exc).__name__}."}


def extract_video_frame_text(path: Path, max_frames: int = 8) -> dict[str, Any]:
    try:
        import pytesseract
    except ImportError:
        return {"status": "unavailable", "frames": [], "note": "Install pytesseract and the Tesseract OCR engine to enable video-frame OCR."}
    _configure_tesseract(pytesseract)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"status": "error", "frames": [], "note": "Video could not be opened for OCR."}
    frames = []
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        for index in np.linspace(0, max(0, count - 1), min(max_frames, count), dtype=int) if count else []:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                continue
            text = pytesseract.image_to_string(frame).strip()
            if text:
                frames.append({"timestamp_seconds": round(float(index) / fps, 3) if fps else None, "frame_index": int(index), "text": text[:3000]})
    except Exception as exc:
        return {"status": "error", "frames": frames, "note": f"Video OCR stopped: {type(exc).__name__}."}
    finally:
        cap.release()
    return {"status": "completed", "frames": frames, "note": "Sampled-frame OCR only; not full-video transcription or claim verification."}
