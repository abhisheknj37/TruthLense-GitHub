from __future__ import annotations

import base64
import io
import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_ID = os.getenv("TRUTHLENS_IMAGE_MODEL", "sakshamkr1/deitfake-v2")
IMAGE_MODEL_LICENSE = "Apache-2.0 (model card)"
IMAGE_MODEL_SCOPE = "Research/education face-image classifier; not validated for forensic decisions."
FAKE_THRESHOLD = float(os.getenv("TRUTHLENS_FAKE_THRESHOLD", "0.65"))
REAL_THRESHOLD = float(os.getenv("TRUTHLENS_REAL_THRESHOLD", "0.35"))


def _uncertainty_band(fake_score: float) -> str:
    """A score-separation band, not calibrated statistical uncertainty."""
    if REAL_THRESHOLD < fake_score < FAKE_THRESHOLD:
        return "ambiguous"
    if abs(fake_score - 0.5) < 0.25:
        return "moderate_separation"
    return "stronger_model_separation"


def _find_encoder_layer(model: torch.nn.Module) -> torch.nn.Module:
    for root_name in ("deit", "vit", "beit", "dinov2"):
        root = getattr(model, root_name, None)
        encoder = getattr(root, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is not None and len(layers):
            return layers[-1]
    base = getattr(model, "base_model", None)
    encoder = getattr(base, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is not None and len(layers):
        return layers[-1]
    raise RuntimeError("Could not locate the final transformer encoder block for patch attribution.")


class ImageDeepfakeDetector:
    """Lazy-loaded face-image classifier with ViT token Grad-CAM-style attribution."""

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = threading.Lock()
        self.load_error: str | None = None
        self.fake_index: int | None = None
        self.real_index: int | None = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

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
                self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
                self.model = AutoModelForImageClassification.from_pretrained(
                    MODEL_ID, use_safetensors=True
                ).to(self.device).eval()
                labels = {
                    int(index): str(label).lower()
                    for index, label in self.model.config.id2label.items()
                }
                self.fake_index = next(
                    (i for i, label in labels.items() if "fake" in label or "synthetic" in label),
                    None,
                )
                self.real_index = next(
                    (i for i, label in labels.items() if "real" in label or "authentic" in label),
                    None,
                )
                if self.fake_index is None or self.real_index is None:
                    raise RuntimeError(
                        f"Model label mapping is not explicit: {labels}. "
                        "Set a checkpoint with documented real/fake class labels."
                    )
            except Exception as exc:
                self.load_error = f"Image model could not be loaded: {exc}"
                self.model = None
                raise RuntimeError(self.load_error) from exc

    def _face_crops(self, image: Image.Image) -> list[tuple[Image.Image, list[int]]]:
        # Respect EXIF orientation before detection (common in phone photos).
        image = ImageOps.exif_transpose(image).convert("RGB")
        rgb = np.asarray(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

        cascade_names = (
            "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt2.xml",
            "haarcascade_frontalface_alt.xml",
            "haarcascade_profileface.xml",
        )
        cascades = []
        for name in cascade_names:
            cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / name))
            if not cascade.empty():
                cascades.append(cascade)
        if self.face_cascade is not None and not self.face_cascade.empty():
            cascades.insert(0, self.face_cascade)

        # Run multiple cascades and contrast variants. A detector miss must remain
        # inconclusive; it must never be converted into a real/fake verdict.
        detections: list[tuple[int, int, int, int]] = []
        for cascade in cascades:
            for source in (gray, enhanced):
                found = cascade.detectMultiScale(
                    source,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(32, 32),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )
                detections.extend(tuple(int(v) for v in box) for box in found)

        # If no face is found at native scale, retry an upscaled image to help
        # with smaller faces. Convert the boxes back to original coordinates.
        if not detections:
            scale = 1.5
            enlarged = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            for cascade in cascades:
                found = cascade.detectMultiScale(
                    enlarged,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(48, 48),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )
                detections.extend(
                    (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
                    for x, y, w, h in found
                )

        if not detections:
            return []

        # Non-maximum suppression removes duplicate detections from the cascades.
        boxes = [[x, y, w, h] for x, y, w, h in detections]
        keep = cv2.dnn.NMSBoxes(boxes, [1.0] * len(boxes), 0.0, 0.35)
        selected = [detections[int(i)] for i in np.asarray(keep).reshape(-1)] if len(keep) else []

        crops: list[tuple[Image.Image, list[int]]] = []
        for x, y, w, h in sorted(selected, key=lambda f: f[2] * f[3], reverse=True)[:3]:
            pad = int(max(w, h) * 0.15)
            left, top = max(0, x - pad), max(0, y - pad)
            right = min(rgb.shape[1], x + w + pad)
            bottom = min(rgb.shape[0], y + h + pad)
            if right <= left or bottom <= top:
                continue
            crop = Image.fromarray(rgb[top:bottom, left:right]).convert("RGB")
            crops.append((crop, [left, top, right - left, bottom - top]))
        return crops

    @staticmethod
    def _overlay_base64(face: Image.Image, heat: np.ndarray) -> str:
        image = np.asarray(face.convert("RGB").resize((224, 224)))
        heat = cv2.resize(heat, (224, 224), interpolation=cv2.INTER_CUBIC)
        heat = np.uint8(np.clip(heat, 0, 1) * 255)
        color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(image, 0.62, color, 0.38, 0)
        output = io.BytesIO()
        Image.fromarray(overlay).save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")

    @staticmethod
    def _crop_preview_base64(face: Image.Image) -> str:
        output = io.BytesIO()
        face.convert("RGB").resize((256, 256)).save(output, format="JPEG", quality=82)
        return base64.b64encode(output.getvalue()).decode("ascii")

    def _predict_crop(self, face: Image.Image, make_cam: bool) -> dict[str, Any]:
        self._load()
        inputs = self.processor(images=face, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        cam_b64 = None
        cam_note = None
        with self.lock:
            if make_cam:
                try:
                    layer = _find_encoder_layer(self.model)
                    captured: dict[str, torch.Tensor] = {}

                    def capture(_module, _inputs, output):
                        hidden = output[0] if isinstance(output, (tuple, list)) else output
                        captured["tokens"] = hidden

                    handle = layer.register_forward_hook(capture)
                    try:
                        with torch.enable_grad():
                            outputs = self.model(**inputs)
                            tokens = captured.get("tokens")
                            if tokens is not None and tokens.ndim == 3 and tokens.requires_grad:
                                score = outputs.logits[0, self.fake_index]
                                gradients = torch.autograd.grad(score, tokens, retain_graph=False)[0]
                                weights = gradients.mean(dim=1, keepdim=True)
                                patch_map = torch.relu((tokens * weights).sum(dim=-1))[0]
                                token_count = patch_map.shape[0]
                                prefix = 1 if int(np.sqrt(max(token_count - 1, 0))) ** 2 == token_count - 1 else 2
                                patch_map = patch_map[prefix:]
                                grid = int(np.sqrt(patch_map.numel()))
                                if grid * grid == patch_map.numel():
                                    patch_map = patch_map.reshape(1, 1, grid, grid)
                                    patch_map = F.interpolate(
                                        patch_map, size=(224, 224), mode="bilinear", align_corners=False
                                    )[0, 0].detach().cpu().numpy()
                                    low, high = float(patch_map.min()), float(patch_map.max())
                                    if high > low:
                                        patch_map = (patch_map - low) / (high - low)
                                        cam_b64 = self._overlay_base64(face, patch_map)
                    finally:
                        handle.remove()
                except Exception as exc:
                    cam_note = f"Patch attribution unavailable: {exc}"
                    with torch.no_grad():
                        outputs = self.model(**inputs)
            else:
                with torch.no_grad():
                    outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits[0].float(), dim=-1).detach().cpu().numpy()
        fake_score = float(probs[self.fake_index])
        real_score = float(probs[self.real_index])
        if fake_score >= FAKE_THRESHOLD:
            prediction = "fake_signal"
        elif fake_score <= REAL_THRESHOLD:
            prediction = "real_signal"
        else:
            prediction = "inconclusive"
        return {
            "prediction": prediction,
            "scores": {"real": round(real_score, 5), "fake": round(fake_score, 5)},
            "model_score_type": "softmax_score_not_calibrated_probability",
            "uncertainty_band": _uncertainty_band(fake_score),
            "gradcam_overlay_base64": cam_b64,
            "gradcam_note": cam_note,
        }

    def analyze_pil_frame(self, image: Image.Image, include_cam: bool = False) -> dict[str, Any]:
        """Run the same face-crop detector on an in-memory video frame."""
        try:
            crops = self._face_crops(image)
            if not crops:
                return {"status": "inconclusive_no_face_detected", "prediction": "inconclusive", "faces": []}
            faces = []
            for index, (crop, box) in enumerate(crops):
                result = self._predict_crop(crop, include_cam and index == 0)
                result.update({"face_index": index, "bounding_box_xywh": box, "crop_preview_base64": self._crop_preview_base64(crop)})
                faces.append(result)
            values = [face["scores"]["fake"] for face in faces]
            median_fake = float(np.median(values))
            prediction = "fake_signal" if median_fake >= FAKE_THRESHOLD else "real_signal" if median_fake <= REAL_THRESHOLD else "inconclusive"
            return {
                "status": "model_assessed", "model_name": MODEL_ID,
                "prediction": prediction,
                "scores": {"fake_median_across_detected_faces": round(median_fake, 5), "real_median_across_detected_faces": round(1 - median_fake, 5)},
                "aggregation": {
                    "method": "median",
                    "face_count": len(faces),
                    "individual_fake_scores": [round(value, 5) for value in values],
                    "explanation": "The median is the middle face score (or mean of the two middle scores). It reduces the influence of a single unusually high or low face score; it is not a calibrated probability.",
                },
                "model_score_type": "median_softmax_score_not_calibrated_probability",
                "uncertainty_band": _uncertainty_band(median_fake),
                "faces": faces,
                "note": "Face-crop research model signal; no calibrated forensic probability is provided."
            }
        except Exception as exc:
            return {"status": "model_error", "model_name": MODEL_ID, "prediction": "inconclusive", "uncertainty_band": "unknown", "faces": [], "note": str(exc)}

    def analyze_image(self, path: Path, include_cam: bool = True) -> dict[str, Any]:
        try:
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
            crops = self._face_crops(image)
            if not crops:
                return {
                    "status": "inconclusive_no_face_detected",
                    "model_name": MODEL_ID,
                    "note": "The selected model is trained for face-cropped images; no suitable face crop was detected.",
                    "prediction": "inconclusive",
                    "faces": [],
                }
            faces = []
            for index, (crop, box) in enumerate(crops):
                result = self._predict_crop(crop, include_cam and index == 0)
                result.update({"face_index": index, "bounding_box_xywh": box, "crop_preview_base64": self._crop_preview_base64(crop)})
                faces.append(result)
            scores = [face["scores"]["fake"] for face in faces]
            median_fake = float(np.median(scores))
            if median_fake >= FAKE_THRESHOLD:
                prediction = "fake_signal"
            elif median_fake <= REAL_THRESHOLD:
                prediction = "real_signal"
            else:
                prediction = "inconclusive"
            return {
                "status": "model_assessed",
                "model_name": MODEL_ID,
                "model_license": IMAGE_MODEL_LICENSE,
                "model_scope": IMAGE_MODEL_SCOPE,
                "prediction": prediction,
                "scores": {
                    "fake_median_across_detected_faces": round(median_fake, 5),
                    "real_median_across_detected_faces": round(1 - median_fake, 5),
                },
                "aggregation": {
                    "method": "median",
                    "face_count": len(faces),
                    "individual_fake_scores": [round(value, 5) for value in scores],
                    "explanation": "The median is the middle face score (or mean of the two middle scores). It reduces the influence of one extreme face score; it is not a calibrated probability.",
                },
                "model_score_type": "median_softmax_score_not_calibrated_probability",
                "uncertainty_band": _uncertainty_band(median_fake),
                "note": (
                    "Face-crop model signal only. It was trained on a particular face-image dataset; "
                    "domain shift, compression, lighting, demographic variation and unseen generators can affect results."
                ),
                "faces": faces,
            }
        except Exception as exc:
            return {
                "status": "model_error",
                "model_name": MODEL_ID,
                "prediction": "inconclusive",
                "uncertainty_band": "unknown",
                "faces": [],
                "note": str(exc),
            }


image_detector = ImageDeepfakeDetector()
