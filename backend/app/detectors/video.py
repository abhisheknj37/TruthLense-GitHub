from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import ImageDeepfakeDetector
from app.services.media import sample_video_frames


class VideoDeepfakeAnalyzer:
    def __init__(self, image_detector: ImageDeepfakeDetector, max_frames: int = 8):
        self.image_detector = image_detector
        self.max_frames = max_frames

    def analyze(self, path: Path) -> dict[str, Any]:
        try:
            decoded = sample_video_frames(path, self.max_frames)
        except Exception as exc:
            return {
                "status": "video_decode_error",
                "prediction": "inconclusive",
                "note": f"Video could not be decoded for frame-level model analysis: {type(exc).__name__}.",
                "frames": [],
            }
        samples = decoded["sampled_frames"]
        if not samples:
            return {
                "status": "inconclusive_no_frames",
                "prediction": "inconclusive",
                "frames": [],
                "metadata": {key: value for key, value in decoded.items() if key != "sampled_frames"},
                "note": "The video opened, but sequential decoding returned no readable frames.",
            }

        frame_results = []
        for sample in samples:
            frame = sample["frame"]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.image_detector.analyze_pil_frame(Image.fromarray(rgb), include_cam=False)
            frame_results.append({
                "frame_index": sample["frame_index"],
                "timestamp_seconds": sample["timestamp_seconds"],
                "status": result.get("status"),
                "prediction": result.get("prediction", "inconclusive"),
                "scores": result.get("scores", {}),
                "uncertainty_band": result.get("uncertainty_band", "unknown"),
                "note": result.get("note"),
            })
        valid_scores = [
            item["scores"].get("fake_median_across_detected_faces")
            for item in frame_results
            if item.get("status") == "model_assessed"
            and isinstance(item.get("scores"), dict)
            and item["scores"].get("fake_median_across_detected_faces") is not None
        ]
        if valid_scores:
            median_fake = float(np.median(valid_scores))
            prediction = (
                "fake_signal" if median_fake >= 0.65
                else "real_signal" if median_fake <= 0.35
                else "inconclusive"
            )
            status = "model_assessed"
        else:
            median_fake = None
            prediction = "inconclusive"
            status = "inconclusive_no_face_frames"
        return {
            "status": status,
            "model_name": self.image_detector.model_name,
            "prediction": prediction,
            "scores": {
                "fake_median_across_face_frames": round(median_fake, 5) if median_fake is not None else None,
                "face_frames_analyzed": len(valid_scores),
                "sampled_frames": len(frame_results),
            },
            "model_score_type": "median_frame_softmax_score_not_calibrated_probability",
            "uncertainty_band": (
                "ambiguous" if median_fake is None or 0.35 < median_fake < 0.65
                else "model_signal_separation"
            ),
            "metadata": {key: value for key, value in decoded.items() if key != "sampled_frames"},
            "frames": frame_results,
            "note": (
                "Sequentially decoded representative frames were assessed independently. This is not temporal deepfake detection; "
                "frame aggregation cannot establish whether an entire video is authentic."
            ),
        }
