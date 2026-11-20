from __future__ import annotations

import math
import os
import re
import threading
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

# This checkpoint is an older AI-art detector, not a general deepfake-photo
# or pixel-manipulation detector. Keep it configurable and disclose its scope.
MODEL_ID = os.getenv("TRUTHLENS_WHOLE_IMAGE_MODEL", "umm-maybe/AI-image-detector")
FAKE_THRESHOLD = float(os.getenv("TRUTHLENS_WHOLE_IMAGE_FAKE_THRESHOLD", "0.65"))
REAL_THRESHOLD = float(os.getenv("TRUTHLENS_WHOLE_IMAGE_REAL_THRESHOLD", "0.35"))
MODEL_SCOPE = (
    "Older AI-art classification research checkpoint (2022); not validated for "
    "deepfake photographs, image splicing, copy-move edits, screenshots or unseen generators."
)


def _normalise_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()


def resolve_binary_labels(id2label: dict[Any, Any]) -> tuple[int, int]:
    """Resolve explicit AI-art and human/real labels; reject ambiguous mappings."""
    labels = {int(index): _normalise_label(label) for index, label in id2label.items()}
    artificial_terms = {"artificial", "ai generated", "ai generated image", "synthetic", "fake"}
    human_terms = {"human", "real", "natural", "authentic", "photographic"}
    ai = [index for index, label in labels.items() if label in artificial_terms]
    real = [index for index, label in labels.items() if label in human_terms]
    if len(ai) != 1 or len(real) != 1 or ai[0] == real[0]:
        raise ValueError(f"Checkpoint must expose one explicit AI and one real/human label; found {labels}")
    return ai[0], real[0]


def _validate_thresholds() -> None:
    if not (math.isfinite(FAKE_THRESHOLD) and math.isfinite(REAL_THRESHOLD)):
        raise ValueError("Whole-image thresholds must be finite numbers.")
    if not (0.0 <= REAL_THRESHOLD < FAKE_THRESHOLD <= 1.0):
        raise ValueError("Require 0 <= REAL_THRESHOLD < FAKE_THRESHOLD <= 1.")


class WholeImageAIDetector:
    def __init__(self) -> None:
        self.processor = None
        self.model = None
        self.device = None
        self.lock = threading.Lock()
        self.error: str | None = None
        self.ai_index: int | None = None
        self.real_index: int | None = None
        self.labels: dict[str, str] = {}

    def _load(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model is not None:
            return
        if self.error:
            raise RuntimeError(self.error)
        with self.lock:
            if self.model is not None:
                return
            try:
                _validate_thresholds()
                self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
                self.model = AutoModelForImageClassification.from_pretrained(
                    MODEL_ID, use_safetensors=True
                ).to(self.device).eval()
                raw_labels = self.model.config.id2label
                self.ai_index, self.real_index = resolve_binary_labels(raw_labels)
                self.labels = {str(k): str(v) for k, v in raw_labels.items()}
                if int(self.model.config.num_labels) != 2:
                    raise ValueError("Whole-image detector requires a two-class checkpoint.")
            except Exception as exc:
                self.error = f"Whole-image detector unavailable: {type(exc).__name__}: {str(exc)[:240]}"
                self.model = None
                raise RuntimeError(self.error) from exc

    def analyze(self, path: Path) -> dict[str, Any]:
        import torch
        try:
            self._load()
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.lock, torch.inference_mode():
                logits = self.model(**inputs).logits[0].float()
            probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
            ai_score = float(probabilities[self.ai_index])
            real_score = float(probabilities[self.real_index])
            prediction = (
                "fake_signal" if ai_score >= FAKE_THRESHOLD
                else "real_signal" if ai_score <= REAL_THRESHOLD
                else "inconclusive"
            )
            return {
                "status": "model_assessed",
                "model_name": MODEL_ID,
                "model_scope": MODEL_SCOPE,
                "class_mapping": {"ai_generated_index": self.ai_index, "real_human_index": self.real_index, "labels": self.labels},
                "prediction": prediction,
                "scores": {"ai_generated": round(ai_score, 5), "real": round(real_score, 5)},
                "model_score_type": "softmax_score_not_calibrated_probability",
                "thresholds": {"fake": FAKE_THRESHOLD, "real": REAL_THRESHOLD},
                "note": (
                    "This is an AI-art research signal, not a general deepfake-photo or manipulation detector. "
                    "It cannot establish copy-move edits, splicing or authenticity. Dataset age, image domain, "
                    "compression and unseen generators can affect performance; human review is required."
                ),
            }
        except Exception as exc:
            return {
                "status": "model_unavailable",
                "model_name": MODEL_ID,
                "model_scope": MODEL_SCOPE,
                "prediction": "inconclusive",
                "scores": {},
                "note": str(exc),
            }


whole_image_detector = WholeImageAIDetector()
