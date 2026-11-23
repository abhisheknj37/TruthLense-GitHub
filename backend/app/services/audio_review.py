from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.ml_runtime import heavy_models_enabled

import numpy as np


def audio_review_timeline(path: Path, sample_rate: int = 16000, points: int = 160) -> dict[str, Any]:
    """Return compact waveform and acoustic-change review cues, not a spoof verdict."""
    if not heavy_models_enabled():
        return {"status": "skipped_low_memory", "waveform": [], "acoustic_review_points": [], "note": "Audio timeline extraction was skipped on this low-memory host to protect API availability."}
    try:
        import librosa
        audio, sr = librosa.load(str(path), sr=sample_rate, mono=True)
        if audio.size == 0:
            return {"status": "empty_audio", "waveform": [], "segments": []}
        duration = len(audio) / sr
        hop = max(1, int(np.ceil(len(audio) / points)))
        waveform = []
        for start in range(0, len(audio), hop):
            chunk = audio[start:start + hop]
            waveform.append({
                "timestamp_seconds": round(start / sr, 3),
                "amplitude_min": round(float(np.min(chunk)), 5),
                "amplitude_max": round(float(np.max(chunk)), 5),
                "rms": round(float(np.sqrt(np.mean(np.square(chunk)))), 5),
            })
        frame_length = 2048
        frame_hop = 512
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=frame_hop)[0]
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=frame_length, hop_length=frame_hop)[0]
        if len(rms) >= 4:
            median = float(np.median(rms))
            mad = float(np.median(np.abs(rms - median))) + 1e-6
            threshold = median + 4.0 * mad
            candidates = np.where(rms > threshold)[0]
        else:
            candidates = np.array([], dtype=int)
        segments = [{
            "timestamp_seconds": round(float(index * frame_hop / sr), 3),
            "rms": round(float(rms[index]), 5),
            "spectral_centroid_hz": round(float(centroid[index]), 2) if index < len(centroid) else None,
            "indicator": "unusually_high_energy"
        } for index in candidates[:100]]
        return {
            "status": "completed",
            "duration_seconds": round(duration, 3),
            "sample_rate_hz": sr,
            "waveform": waveform,
            "acoustic_review_points": segments,
            "method": "downsampled waveform, RMS energy and spectral centroid",
            "interpretation": "These are acoustic review cues only. Loud sounds, edits, music, silence and recording conditions can produce the same patterns; they do not identify synthetic speech.",
        }
    except Exception as exc:
        return {
            "status": "analysis_error",
            "waveform": [],
            "segments": [],
            "note": f"Audio timeline could not be generated ({type(exc).__name__}).",
        }
