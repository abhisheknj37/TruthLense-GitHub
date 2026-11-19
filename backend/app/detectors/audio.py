from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


MODEL_ID = os.getenv("TRUTHLENS_AUDIO_MODEL", "Hemgg/Deepfake-audio-detection")
MODEL_LICENSE = "Apache-2.0 (Hugging Face model card)"
SAMPLE_RATE = 16000
WINDOW_SECONDS = 8.0
MAX_WINDOWS = 5


def score_band(fake_score: float) -> str:
    if 0.35 < fake_score < 0.65:
        return "ambiguous"
    return "model_score_separated"


class AudioDeepfakeDetector:
    """Lazy Wav2Vec2-based speech spoof indicator. Scores are not calibrated probabilities."""

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = threading.Lock()
        self.load_error: str | None = None
        self.fake_index: int | None = None
        self.real_index: int | None = None

    @property
    def model_name(self) -> str:
        return MODEL_ID

    def _load(self) -> None:
        if self.model is not None:
            return
        if self.load_error:
            raise RuntimeError(self.load_error)
        with self.lock:
            if self.model is not None:
                return
            try:
                self.processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
                self.model = AutoModelForAudioClassification.from_pretrained(
                    MODEL_ID, use_safetensors=True
                ).to(self.device).eval()
                labels = {
                    int(index): str(label).lower()
                    for index, label in self.model.config.id2label.items()
                }
                self.fake_index = next(
                    (i for i, label in labels.items()
                     if any(word in label for word in ("fake", "spoof", "synthetic", "ai"))),
                    None,
                )
                self.real_index = next(
                    (i for i, label in labels.items()
                     if any(word in label for word in ("real", "human", "bonafide", "bona fide"))),
                    None,
                )
                if self.fake_index is None or self.real_index is None:
                    raise RuntimeError(f"Model label mapping is not explicit: {labels}")
            except Exception as exc:
                self.load_error = f"Audio model could not be loaded: {exc}"
                self.model = None
                raise RuntimeError(self.load_error) from exc

    def analyze(self, path: Path) -> dict[str, Any]:
        try:
            self._load()
            audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
            duration = len(audio) / SAMPLE_RATE
            if duration < 2.0:
                return {
                    "status": "inconclusive_short_audio",
                    "model_name": MODEL_ID,
                    "prediction": "inconclusive",
                    "note": "The recording is shorter than two seconds; this model is intended for speech segments.",
                }

            starts = np.linspace(
                0.0,
                max(0.0, duration - WINDOW_SECONDS),
                num=min(MAX_WINDOWS, max(1, int(np.ceil(duration / WINDOW_SECONDS)))),
            )
            segments = []
            for start in sorted(set(float(value) for value in starts)):
                end = min(duration, start + WINDOW_SECONDS)
                clip = audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
                if len(clip) < int(2.0 * SAMPLE_RATE):
                    continue
                inputs = self.processor(
                    clip, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with torch.inference_mode():
                    logits = self.model(**inputs).logits[0]
                    probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                segments.append({
                    "start_seconds": round(start, 3),
                    "end_seconds": round(end, 3),
                    "fake_model_score": round(float(probs[self.fake_index]), 5),
                    "real_model_score": round(float(probs[self.real_index]), 5),
                })

            if not segments:
                return {
                    "status": "inconclusive_no_speech_windows",
                    "model_name": MODEL_ID,
                    "prediction": "inconclusive",
                    "note": "No valid audio windows could be analyzed.",
                }

            values = [item["fake_model_score"] for item in segments]
            median_fake = float(np.median(values))
            prediction = (
                "fake_signal" if median_fake >= 0.65
                else "real_signal" if median_fake <= 0.35
                else "inconclusive"
            )
            return {
                "status": "model_assessed",
                "model_name": MODEL_ID,
                "model_license": MODEL_LICENSE,
                "prediction": prediction,
                "scores": {
                    "fake_median_across_windows": round(median_fake, 5),
                    "real_median_across_windows": round(1 - median_fake, 5),
                },
                "model_score_type": "softmax_score_not_calibrated_probability",
                "uncertainty_band": score_band(median_fake),
                "segments": segments,
                "voice_synthesis_review": {
                    "status": "model_indicator_available" if prediction != "inconclusive" else "inconclusive",
                    "indicator": prediction,
                    "model": MODEL_ID,
                    "windows_assessed": len(segments),
                    "median_fake_model_score": round(median_fake, 5),
                    "note": (
                        "Synthetic/spoof-direction research-model indicator only. It does not identify a speaker, "
                        "establish who generated the voice, or prove voice cloning."
                    ),
                },
                "note": (
                    "Research speech model trained on a multi-ethnic English audio dataset at approximately 16 kHz. "
                    "The model card reports self-evaluated accuracy, but does not provide enough detail to establish "
                    "generalization to all languages, codecs, noise conditions or voice-generation systems."
                ),
            }
        except Exception as exc:
            return {
                "status": "model_error",
                "model_name": MODEL_ID,
                "prediction": "inconclusive",
                "uncertainty_band": "unknown",
                "note": str(exc),
            }


audio_detector = AudioDeepfakeDetector()
