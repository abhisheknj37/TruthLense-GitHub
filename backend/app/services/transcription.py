from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.services.ml_runtime import heavy_models_enabled

_pipeline = None
_pipeline_error = None


def transcribe_audio(path: Path) -> dict[str, Any]:
    """Optional local ASR; configure TRUTHLENS_ASR_MODEL to enable."""
    global _pipeline, _pipeline_error
    model_name = os.getenv("TRUTHLENS_ASR_MODEL", "").strip()
    if not heavy_models_enabled():
        return {"status": "skipped_low_memory", "text": "", "model": model_name or None, "note": "Local ASR was skipped because heavyweight ML inference is disabled on this low-memory host."}
    if not model_name:
        return {"status": "not_configured", "text": "", "note": "Set TRUTHLENS_ASR_MODEL to enable optional speech transcription."}
    if _pipeline is None and _pipeline_error is None:
        try:
            from transformers import pipeline
            _pipeline = pipeline("automatic-speech-recognition", model=model_name)
        except Exception as exc:
            _pipeline_error = type(exc).__name__
    if _pipeline is None:
        return {"status": "unavailable", "text": "", "model": model_name, "note": f"ASR model could not be loaded ({_pipeline_error})."}
    try:
        result = _pipeline(str(path), chunk_length_s=20, return_timestamps=True)
        return {"status": "completed", "text": str(result.get("text", ""))[:20000], "chunks": result.get("chunks", [])[:200], "model": model_name, "note": "Machine transcription may contain errors; verify against the original audio."}
    except Exception as exc:
        return {"status": "error", "text": "", "model": model_name, "note": f"Transcription failed ({type(exc).__name__})."}
