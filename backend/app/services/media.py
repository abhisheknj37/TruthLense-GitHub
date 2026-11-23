from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import soundfile as sf
from PIL import Image, ExifTags, UnidentifiedImageError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".flac", ".ogg"}
TEXT_EXTS = {".txt", ".md"}
MAX_TEXT_BYTES = 1 * 1024 * 1024


class MediaError(ValueError):
    pass


def media_type_for(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in TEXT_EXTS:
        return "text"
    raise MediaError(f"Unsupported file extension: {ext or 'unknown'}")


def image_forensic_parameters(path: Path) -> dict[str, Any]:
    """Descriptive image-quality measurements; not validated manipulation detectors."""
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32)
    h, w = gray.shape
    cells = [float(gray_f[round(r*h/4):round((r+1)*h/4), round(c*w/4):round((c+1)*w/4)].mean())
             for r in range(4) for c in range(4)
             if round((r+1)*h/4) > round(r*h/4) and round((c+1)*w/4) > round(c*w/4)]
    illumination_std = float(np.std(cells)) if cells else 0.0
    texture_detail = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    residual = gray_f - cv2.GaussianBlur(gray_f, (3, 3), 0)
    noise_sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    gx, gy = np.abs(np.diff(gray_f, axis=1)), np.abs(np.diff(gray_f, axis=0))
    block_edges, other_edges = [], []
    if gx.size:
        mask = np.zeros(gx.shape[1], dtype=bool); mask[np.arange(7, gx.shape[1], 8)] = True
        block_edges.extend(gx[:, mask].ravel().tolist()); other_edges.extend(gx[:, ~mask].ravel().tolist())
    if gy.size:
        mask = np.zeros(gy.shape[0], dtype=bool); mask[np.arange(7, gy.shape[0], 8)] = True
        block_edges.extend(gy[mask, :].ravel().tolist()); other_edges.extend(gy[~mask, :].ravel().tolist())
    ratio = float(np.mean(block_edges)) / max(float(np.mean(other_edges)), 1e-6) if other_edges else 0.0
    return {
        "method": "classical_image_quality_indicators_v1",
        "interpretation": "Descriptive measurements only; none independently proves manipulation or authenticity.",
        "lighting": {"status": "measured", "grid_mean_intensity_std": round(illumination_std, 4), "note": "Brightness variation across a 4x4 grid; natural lighting also varies."},
        "texture": {"status": "measured", "laplacian_variance": round(texture_detail, 4), "note": "Focus/detail indicator, not a deepfake score."},
        "noise": {"status": "measured", "robust_high_frequency_sigma": round(noise_sigma, 4), "note": "High-frequency residual estimate; resizing and denoising affect it."},
        "compression": {"status": "heuristic", "eight_pixel_boundary_gradient_ratio": round(ratio, 4), "note": "Block-boundary heuristic, not a validated JPEG tampering detector."},
    }


def inspect_image(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            exif = image.getexif()
            exif_names = [
                ExifTags.TAGS.get(tag, str(tag))
                for tag in exif.keys()
            ]
            metadata = {
                "format": image.format,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "exif_fields": exif_names[:30],
                "exif_present": bool(exif),
            }
    except (UnidentifiedImageError, OSError) as exc:
        raise MediaError("The uploaded image could not be decoded.") from exc

    observations = [{
        "code": "IMAGE_METADATA_READ",
        "title": "Image metadata inspected",
        "detail": (
            "Dimensions, encoding and available EXIF fields were read. "
            "Missing or present metadata does not establish whether an image is authentic."
        ),
        "category": "metadata",
        "severity": "info",
    }]
    return metadata, observations


def sample_video_frames(path: Path, max_frames: int = 16) -> dict[str, Any]:
    """Count decoded frames, then decode evenly spaced frame indices in a second pass."""
    first = cv2.VideoCapture(str(path))
    if not first.isOpened():
        raise MediaError("The uploaded video could not be decoded by OpenCV.")
    fps_value = float(first.get(cv2.CAP_PROP_FPS) or 0)
    fps = fps_value if math.isfinite(fps_value) and 0 < fps_value < 1000 else None
    width_value = int(first.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height_value = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    width = width_value if width_value > 0 else None
    height = height_value if height_value > 0 else None
    decoded_count = 0
    try:
        while True:
            ok, _ = first.read()
            if not ok:
                break
            decoded_count += 1
    finally:
        first.release()

    if decoded_count == 0:
        return {
            "fps": fps, "frame_count": 0, "duration_seconds": None,
            "width": width, "height": height, "sampled_frames": [],
            "sampling_method": "two-pass exact temporal stratification",
        }

    limit = max(1, min(int(max_frames), decoded_count))
    target_indices = np.unique(np.linspace(0, decoded_count - 1, num=limit, dtype=int)).tolist()
    targets = set(target_indices)
    second = cv2.VideoCapture(str(path))
    if not second.isOpened():
        raise MediaError("The uploaded video could not be reopened for temporal sampling.")
    retained = []
    frame_index = 0
    try:
        while frame_index < decoded_count:
            ok, frame = second.read()
            if not ok:
                break
            if frame_index in targets:
                retained.append((frame_index, frame.copy()))
            frame_index += 1
    finally:
        second.release()

    samples = [{
        "frame_index": index,
        "timestamp_seconds": round(index / fps, 3) if fps else None,
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "frame": frame,
    } for index, frame in retained]
    return {
        "fps": fps,
        "frame_count": decoded_count,
        "duration_seconds": round(decoded_count / fps, 3) if fps else None,
        "width": width,
        "height": height,
        "sampled_frames": samples,
        "sampling_method": "two-pass exact temporal stratification",
        "target_frame_indices": target_indices,
        "sampled_frame_indices": [index for index, _ in retained],
    }


def inspect_video(path: Path, sample_limit: int = 16) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    decoded = sample_video_frames(path, sample_limit)
    samples = [
        {key: value for key, value in item.items() if key != "frame"}
        for item in decoded["sampled_frames"]
    ]
    frame_count = decoded["frame_count"]
    metadata = {
        "container_extension": path.suffix.lower(),
        "fps": round(decoded["fps"], 3) if decoded["fps"] else None,
        "frame_count": frame_count,
        "duration_seconds": decoded["duration_seconds"],
        "width": decoded["width"],
        "height": decoded["height"],
        "sampled_frame_count": len(samples),
        "sampling_method": decoded.get("sampling_method"),
        "sampled_frame_indices": decoded.get("sampled_frame_indices", []),
    }
    if frame_count == 0:
        detail = "The video container opened, but sequential decoding returned no readable frames."
        severity = "warning"
    else:
        detail = (
            f"Sequential decoding read {frame_count} frames and retained "
            f"{len(samples)} representative samples. Frame sampling is not, by itself, a deepfake detection result."
        )
        severity = "info"
    observations = [{
        "code": "VIDEO_FRAMES_SAMPLED",
        "title": "Video structure inspected",
        "detail": detail,
        "category": "video",
        "severity": severity,
    }]
    return metadata, observations, samples

def inspect_text(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Read a bounded plain-text evidence file without executing or interpreting markup."""
    size = path.stat().st_size
    if size > MAX_TEXT_BYTES:
        raise MediaError("Plain-text evidence exceeds the 1 MB text-file limit.")
    try:
        raw = path.read_bytes()
        content = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise MediaError("Text evidence must be UTF-8 encoded (UTF-8 with or without BOM).") from exc
    if not content.strip():
        raise MediaError("Text evidence is empty.")
    truncated = len(content) > 20000
    extraction = {
        "status": "completed",
        "engine": "UTF-8 plain-text reader",
        "text": content[:20000],
        "character_count": len(content),
        "word_count": len(content.split()),
        "analysis_eligible": True,
        "truncated_for_report": truncated,
        "note": "Plain-text source imported as evidence. Markup is treated as text; no links, scripts or embedded content are executed.",
    }
    metadata = {
        "encoding": "UTF-8",
        "byte_count": size,
        "character_count": len(content),
        "word_count": len(content.split()),
        "truncated_for_report": truncated,
    }
    observations = [{
        "code": "TEXT_EVIDENCE_IMPORTED",
        "title": "Plain-text evidence imported",
        "detail": "UTF-8 text was read and attached as a source for claim, transcript and caption comparison. It is not independently verified.",
        "category": "text",
        "severity": "info",
    }]
    return metadata, observations, extraction


def inspect_audio(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        info = sf.info(str(path))
        duration = float(info.duration)
        # Stream in small blocks to avoid loading a long recording into memory.
        peak = 0.0
        square_sum = 0.0
        count = 0
        clipped = 0
        for block in sf.blocks(str(path), blocksize=65536, dtype="float32", always_2d=True):
            if block.size == 0:
                continue
            abs_block = np.abs(block)
            peak = max(peak, float(abs_block.max()))
            square_sum += float(np.square(block, dtype=np.float64).sum())
            count += int(block.size)
            clipped += int(np.count_nonzero(abs_block >= 0.999))
        rms = math.sqrt(square_sum / count) if count else 0.0
    except Exception as exc:
        raise MediaError(
            "Audio decoding failed. For this MVP, use a WAV, FLAC or OGG file supported by libsndfile."
        ) from exc

    metadata = {
        "format": info.format,
        "subtype": info.subtype,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "duration_seconds": round(duration, 3),
        "peak_amplitude": round(peak, 5),
        "rms_amplitude": round(rms, 5),
        "near_clipped_sample_count": clipped,
    }
    observations = [{
        "code": "AUDIO_SIGNAL_INSPECTED",
        "title": "Audio signal inspected",
        "detail": (
            "Duration, sample rate, channels and signal-level statistics were measured. "
            "These measurements do not identify cloned or synthetic speech."
        ),
        "category": "audio",
        "severity": "info",
    }]
    return metadata, observations


def inspect_media(path: Path) -> dict[str, Any]:
    media_type = media_type_for(path)
    if media_type == "image":
        metadata, observations = inspect_image(path)
        samples = []
        forensic_parameters = image_forensic_parameters(path)
    elif media_type == "video":
        metadata, observations, samples = inspect_video(path)
        forensic_parameters = {}
    elif media_type == "audio":
        metadata, observations = inspect_audio(path)
        samples = []
        forensic_parameters = {}
    else:
        metadata, observations, text_extraction = inspect_text(path)
        samples = []
        forensic_parameters = {}
        return {
            "media_type": media_type, "metadata": metadata,
            "observations": observations, "sampled_frames": samples,
            "forensic_parameters": forensic_parameters,
            "text_extraction": text_extraction,
        }
    return {
        "media_type": media_type,
        "metadata": metadata,
        "observations": observations,
        "sampled_frames": samples,
        "forensic_parameters": forensic_parameters,
        "text_extraction": {},
    }
