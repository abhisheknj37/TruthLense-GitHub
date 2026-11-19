from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Keep this detector separate from the face-crop classifier. The checkpoint is
# configurable because AI-image detector performance varies by generator/domain.
MODEL_ID = os.getenv("TRUTHLENS_WHOLE_IMAGE_MODEL", "umm-maybe/AI-image-detector")
FAKE_THRESHOLD = float(os.getenv("TRUTHLENS_WHOLE_IMAGE_FAKE_THRESHOLD", "0.65"))
REAL_THRESHOLD = float(os.getenv("TRUTHLENS_WHOLE_IMAGE_REAL_THRESHOLD", "0.35"))


class WholeImageAIDetector:
    def __init__(self) -> None:
        self.processor = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = threading.Lock()
        self.error: str | None = None
        self.ai_index: int | None = None
        self.real_index: int | None = None

    def _load(self) -> None:
        if self.model is not None:
            return
        if self.error:
            raise RuntimeError(self.error)
        with self.lock:
            if self.model is not None:
                return
            try:
                self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
                self.model = AutoModelForImageClassification.from_pretrained(
                    MODEL_ID, use_safetensors=True
                ).to(self.device).eval()
                labels = {int(i): str(v).lower() for i, v in self.model.config.id2label.items()}
                self.ai_index = next((i for i, v in labels.items() if any(k in v for k in ("artificial", "ai", "synthetic", "fake"))), None)
                self.real_index = next((i for i, v in labels.items() if any(k in v for k in ("real", "human", "authentic", "natural"))), None)
                if self.ai_index is None or self.real_index is None:
                    raise RuntimeError(f"Checkpoint labels are not explicitly AI/real: {labels}")
            except Exception as exc:
                self.error = f"Whole-image detector unavailable: {type(exc).__name__}: {str(exc)[:240]}"
                self.model = None
                raise RuntimeError(self.error) from exc

    def analyze(self, path: Path) -> dict[str, Any]:
        try:
            self._load()
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.lock, torch.no_grad():
                logits = self.model(**inputs).logits[0].float()
            probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
            ai_score = float(probabilities[self.ai_index])
            real_score = float(probabilities[self.real_index])
            prediction = "fake_signal" if ai_score >= FAKE_THRESHOLD else "real_signal" if ai_score <= REAL_THRESHOLD else "inconclusive"
            return {
                "status": "model_assessed",
                "model_name": MODEL_ID,
                "model_scope": "Whole-image AI-generated image research classifier; not validated for forensic decisions.",
                "prediction": prediction,
                "scores": {"ai_generated": round(ai_score, 5), "real": round(real_score, 5)},
                "model_score_type": "softmax_score_not_calibrated_probability",
                "note": "Whole-image model signal. Dataset coverage, image editing, compression and unseen generators can affect performance; human review is required.",
            }
        except Exception as exc:
            return {
                "status": "model_unavailable",
                "model_name": MODEL_ID,
                "prediction": "inconclusive",
                "scores": {},
                "note": str(exc),
            }


whole_image_detector = WholeImageAIDetector()
