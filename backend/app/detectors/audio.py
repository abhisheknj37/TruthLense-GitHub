from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


MODEL_ID = os.getenv(
    "TRUTHLENS_AUDIO_MODEL",
    "garystafford/wav2vec2-deepfake-voice-detector",
)
MODEL_LICENSE = "Apache-2.0 (model card)"
# The model card describes 2.5–13 second speech segments. Its reported training corpus
# contains 1,866 clips from a limited set of TTS systems, so this is a narrow research signal.
MIN_SECONDS = 2.5
MAX_SECONDS = 13.0
CHUNK_SECONDS = 8.0


def _band(score: float) -> str:
    if 0.35 < score < 0.65:
        return "ambiguous"
    if abs(score - 0.5) < 0.25:
        return "moderate_separation"
    return "stronger_model_separation"


class AudioDeepfakeDetector:
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = threading.Lock()
        self.load_error: str | None = None

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
            except Exception as exc:
                self.load_error = f"Audio model could not be loaded: {exc}"
                raise RuntimeError(self.load_error) from exc

    def analyze(self, path: Path) -> dict[str, Any]:
        try:
            self._load()
            audio, sample_rate = librosa.load(str(path), sr=16000, mono=True)
            duration = len(audio) / 16000
            if duration < MIN_SECONDS:
                return {
                    "status": "inconclusive_outside_model_duration",
                    "model_name": MODEL_ID,
                    "prediction": "inconclusive",
                    "note": f"Audio is {duration:.2f}s; this model card describes an optimal range of {MIN_SECONDS}-{MAX_SECONDS}s.",
                }
            # Evaluate up to five evenly spaced speech windows; do not assume music/noise is speech.
            starts = np.linspace(
                0, max(0.0, duration - CHUNK_SECONDS),
                num=min(5, max(1, int(np.ceil(duration / CHUNK_SECONDS)))),
            )
            results = []
            id2label = {
                int(index): str(label).lower()
                for index, label in self.model.config.id2label.items()
            }
            fake_index = next(
                (index for index, label in id2label.items() if "fake" in label or "synthetic" in label or "spoof" in label),
                1,  # The model card explicitly documents class 0=real, class 1=fake.
            )
            for start in starts:
                end = min(duration, float(start) + MAX_SECONDS)
                segment = audio[int(start * 16000):int(end * 16000)]
                if len(segment) < int(MIN_SECONDS * 16000):
                    continue
                inputs = self.processor(
                    segment, sampling_rate=16000, return_tensors="pt", padding=True
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with torch.inference_mode():
                    logits = self.model(**inputs).logits[0]
                    probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                fake_score = float(probabilities[fake_index])
                results.append({
                    "start_seconds": round(float(start), 3),
                    "end_seconds": round(float(end), 3),
                    "fake_model_score": round(fake_score, 5),
                })
            if not results:
                return {
                    "status": "inconclusive_no_valid_audio_windows",
                    "model_name": MODEL_ID,
                    "prediction": "inconclusive",
                    "note": "No valid speech analysis window could be formed.",
                }
            scores = [item["fake_model_score"] for item in results]
            median_fake = float(np.median(scores))
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
                "uncertainty_band": _band(median_fake),
                "segments": results,
                "note": (
                    "Research speech detector. Its model card describes 1,866 training clips from six TTS systems; "
                    "generalization to other voices, languages, generators, codecs, music and noisy recordings is uncertain. "
                    "This is not proof of a cloned voice."
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
