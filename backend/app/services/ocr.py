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


def _prepare_image_variants(image: Image.Image) -> tuple[list[tuple[str, Image.Image]], dict[str, Any]]:
    """Create conservative OCR variants; original uploaded media is never modified."""
    image = ImageOps.exif_transpose(image).convert("L")
    original_size = image.size
    if max(image.size) < 1800:
        scale = min(3.0, 1800 / max(image.size))
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    image, angle = _deskew_image(image)
    array = np.asarray(image)
    denoised = cv2.medianBlur(array, 3) if min(array.shape) >= 3 else array
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    otsu = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11,
    ) if min(contrast.shape) >= 31 else otsu
    variants = [
        ("contrast", Image.fromarray(contrast)),
        ("otsu", Image.fromarray(otsu)),
        ("adaptive", Image.fromarray(adaptive)),
    ]
    return variants, {
        "original_dimensions": [int(original_size[0]), int(original_size[1])],
        "processed_dimensions": [int(image.width), int(image.height)],
        "deskew_correction_degrees": angle,
        "preprocessing": [
            "EXIF orientation", "grayscale", "upscale", "deskew",
            "median denoise", "CLAHE contrast", "Otsu threshold",
            "adaptive threshold",
        ],
        "ocr_variants_tested": [name for name, _ in variants],
    }


def _candidate_quality(text: str, confidence: float | None, word_count: int) -> float:
    """Rank candidates using confidence and text legibility cues, not authenticity."""
    if confidence is None or not text.strip():
        return -1.0
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return -1.0
    alphanumeric_ratio = sum(char.isalnum() for char in visible) / len(visible)
    # Confidence remains dominant; the small legibility term discourages symbol noise.
    return confidence + (10.0 * alphanumeric_ratio) + min(word_count, 30) * 0.15


def _extract_with_confidence(pytesseract, variants: list[tuple[str, Image.Image]], languages: str) -> dict[str, Any]:
    """Compare preprocessing variants and layout modes; retain the most legible result."""
    candidates = []
    for variant_name, image in variants:
        for psm in (6, 11, 4):
            config = f"--oem 3 --psm {psm}"
            data = pytesseract.image_to_data(
                image, lang=languages, config=config,
                output_type=pytesseract.Output.DICT,
            )
            words, confidences = [], []
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
                "preprocessing_variant": variant_name,
                "selection_score": round(_candidate_quality(text, confidence, len(words)), 2),
            })
    selected = max(candidates, key=lambda item: item["selection_score"], default={
        "text": "", "mean_word_confidence": None, "word_count": 0,
        "page_segmentation_mode": None, "preprocessing_variant": None,
        "selection_score": -1.0,
    })
    # Do not expose the internal selection score as a confidence or authenticity score.
    return selected


def _detect_text_regions(image: Image.Image, max_regions: int = 12) -> list[tuple[int, int, int, int]]:
    """Propose broad text blocks using morphology; these are OCR crops, not semantic detection."""
    gray = np.asarray(image.convert("L"))
    if min(gray.shape) < 40:
        return []
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(15, gray.shape[1] // 35), 3)
    )
    joined = cv2.dilate(binary, kernel, iterations=1)
    joined = cv2.morphologyEx(
        joined, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)),
    )
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = (w * h) / float(width * height)
        if w < width * 0.08 or h < max(12, height * 0.012):
            continue
        if area_ratio < 0.002 or area_ratio > 0.85:
            continue
        pad_x, pad_y = max(8, int(w * 0.025)), max(6, int(h * 0.18))
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(width, x + w + pad_x), min(height, y + h + pad_y)
        boxes.append((x1, y1, x2, y2))
    boxes.sort(key=lambda box: (box[1], box[0]))
    # Merge overlapping/nearby boxes to avoid OCRing fragmented individual lines.
    merged = []
    for box in boxes:
        x1, y1, x2, y2 = box
        match = None
        for index, (a, b, d, e) in enumerate(merged):
            if x1 <= d + 12 and x2 >= a - 12 and y1 <= e + 12 and y2 >= b - 12:
                match = index
                break
        if match is None:
            merged.append(box)
        else:
            a, b, d, e = merged[match]
            merged[match] = (min(a, x1), min(b, y1), max(d, x2), max(e, y2))
    return sorted(merged, key=lambda box: (box[1], box[0]))[:max_regions]


def _extract_region_candidate(pytesseract, variants: list[tuple[str, Image.Image]], languages: str) -> dict[str, Any] | None:
    """OCR detected blocks and aggregate their word confidence in reading order."""
    base = variants[0][1]
    regions = _detect_text_regions(base)
    if not regions:
        return None
    words, confidences, selected_modes = [], [], []
    for x1, y1, x2, y2 in regions:
        crop_variants = []
        for name, image in variants:
            crop = image.crop((x1, y1, x2, y2))
            crop_variants.append((name, crop))
        best = _extract_with_confidence(pytesseract, crop_variants, languages)
        if best["text"]:
            words.append(best["text"])
            if best["mean_word_confidence"] is not None:
                confidences.append(best["mean_word_confidence"])
            selected_modes.append(best["page_segmentation_mode"])
    if not words:
        return None
    text = "\\n".join(words)
    confidence = round(float(np.mean(confidences)), 1) if confidences else None
    count = sum(len(re.findall(r"\\S+", part)) for part in words)
    return {
        "text": text,
        "mean_word_confidence": confidence,
        "word_count": count,
        "page_segmentation_mode": "region_crops",
        "preprocessing_variant": "per_region_best",
        "selection_score": round(_candidate_quality(text, confidence, count), 2),
        "detected_region_count": len(regions),
    }

def _ocr_note(confidence: float | None) -> tuple[str, str]:
    if confidence is None:
        return "unavailable", "No word-level confidence was returned by Tesseract; verify the image manually."
    if confidence < 45:
        return "low", "Low OCR confidence. Treat extracted text as unreliable and verify every character against the original."
    if confidence < 70:
        return "moderate", "Moderate OCR confidence. Review the extracted text against the original image."
    return "high", "OCR confidence is a Tesseract recognition estimate, not a probability that the text or media is authentic."


def _ocr_analysis_gate(text: str, confidence: float | None, word_count: int) -> dict[str, Any]:
    visible = [char for char in text if not char.isspace()]
    ratio = sum(char.isalnum() for char in visible) / len(visible) if visible else 0.0
    eligible = bool(text.strip() and confidence is not None and confidence >= 45 and word_count >= 3 and ratio >= 0.55)
    return {
        "analysis_text": text if eligible else "",
        "analysis_eligible": eligible,
        "analysis_gate": {
            "minimum_mean_word_confidence": 45,
            "minimum_word_count": 3,
            "minimum_alphanumeric_ratio": 0.55,
            "observed_alphanumeric_ratio": round(ratio, 3),
        },
    }


def _run_ocr(pytesseract, image: Image.Image, languages: str) -> dict[str, Any]:
    variants, preprocessing = _prepare_image_variants(image)
    full_page_result = _extract_with_confidence(pytesseract, variants, languages)
    region_result = _extract_region_candidate(pytesseract, variants, languages)
    candidates = [full_page_result] + ([region_result] if region_result else [])
    result = max(candidates, key=lambda item: item["selection_score"])
    confidence = result["mean_word_confidence"]
    confidence_level, confidence_note = _ocr_note(confidence)
    text = result["text"][:12000]
    gate = _ocr_analysis_gate(text, confidence, result["word_count"])
    return {
        "status": "completed" if text else "completed_no_text",
        "text": text,
        **gate,
        "engine": "Tesseract OCR",
        "languages": languages,
        "character_count": len(text),
        "word_count": result["word_count"],
        "mean_word_confidence": confidence,
        "confidence_level": confidence_level,
        "page_segmentation_mode": result["page_segmentation_mode"],
        "preprocessing_variant": result["preprocessing_variant"],
        "detected_text_region_count": result.get("detected_region_count", 0),
        "ocr_candidates_compared": ["full_page", "detected_regions"] if region_result else ["full_page"],
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
