from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import soundfile as sf
from scipy.signal import correlate


def estimate_audio_video_activity_alignment(video_path: Path, audio_path: Path) -> dict[str, Any]:
    """Compare coarse video lower-frame motion to audio RMS energy; not phoneme/viseme lip-sync."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "unavailable", "note": "Video could not be decoded."}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or count < 10:
        cap.release()
        return {"status": "insufficient_data", "note": "Valid FPS and at least ten frames are required."}
    step = max(1, int(round(fps / 8)))
    motion = []
    previous = None
    try:
        for index in range(0, count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            crop = cv2.resize(gray[int(h * .48):int(h * .92), int(w * .18):int(w * .82)], (96, 48))
            if previous is not None:
                motion.append(float(np.mean(cv2.absdiff(crop, previous))))
            previous = crop
    finally:
        cap.release()
    try:
        audio, sr = sf.read(str(audio_path), always_2d=True)
        audio = audio.mean(axis=1)
        window = max(1, int(sr * step / fps))
        energy = [float(np.sqrt(np.mean(audio[i:i + window] ** 2))) for i in range(0, len(audio) - window + 1, window)]
    except Exception:
        return {"status": "unavailable", "note": "Audio could not be decoded for activity alignment."}
    n = min(len(motion), len(energy))
    if n < 8:
        return {"status": "insufficient_data", "note": "Too few activity windows were available."}
    x, y = np.asarray(motion[:n]), np.asarray(energy[:n])
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    corr = correlate(x, y, mode="full", method="fft") / n
    lags = np.arange(-n + 1, n)
    valid = np.abs(lags) <= int(1 / max(step / fps, 1e-3))
    best = int(np.argmax(corr[valid]))
    lag = int(lags[valid][best])
    return {
        "status": "experimental_activity_alignment",
        "activity_correlation": round(float(corr[valid][best]), 4),
        "estimated_offset_seconds": round(lag * step / fps, 3),
        "windows_compared": int(n),
        "method": "lower-frame motion envelope versus audio RMS energy envelope",
        "interpretation": "Exploratory coarse alignment only; not phoneme-to-viseme lip-sync verification and cannot establish dubbing, manipulation, or authenticity.",
        "note": "Camera motion, cuts, multiple faces, off-screen speech, music and compression can affect this heuristic. Human review is required.",
    }
