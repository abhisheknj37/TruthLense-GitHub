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
    raise MediaError(f"Unsupported file extension: {ext or 'unknown'}")


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


def inspect_video(path: Path, sample_limit: int = 8) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise MediaError("The uploaded video could not be decoded by OpenCV.")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 else None
        samples: list[dict[str, Any]] = []

        if frame_count > 0:
            indexes = np.linspace(
                0, max(frame_count - 1, 0),
                num=min(sample_limit, frame_count),
                dtype=int,
            )
            for index in sorted(set(indexes.tolist())):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if ok:
                    samples.append({
                        "frame_index": int(index),
                        "timestamp_seconds": round(index / fps, 3) if fps > 0 else None,
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                    })
    finally:
        capture.release()

    metadata = {
        "container_extension": path.suffix.lower(),
        "fps": round(fps, 3) if fps > 0 else None,
        "frame_count": frame_count,
        "duration_seconds": round(duration, 3) if duration is not None else None,
        "width": width,
        "height": height,
        "sampled_frame_count": len(samples),
    }
    observations = [{
        "code": "VIDEO_FRAMES_SAMPLED",
        "title": "Video structure inspected",
        "detail": (
            f"Read {len(samples)} representative frames from {frame_count} reported frames. "
            "Frame sampling is not, by itself, a deepfake detection result."
        ),
        "category": "video",
        "severity": "info",
    }]
    return metadata, observations, samples


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
    elif media_type == "video":
        metadata, observations, samples = inspect_video(path)
    else:
        metadata, observations = inspect_audio(path)
        samples = []
    return {
        "media_type": media_type,
        "metadata": metadata,
        "observations": observations,
        "sampled_frames": samples,
    }
