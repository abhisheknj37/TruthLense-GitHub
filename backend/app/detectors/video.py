from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import ImageDeepfakeDetector


class VideoDeepfakeAnalyzer:
    def __init__(self, image_detector: ImageDeepfakeDetector, max_frames: int = 8):
        self.image_detector = image_detector
        self.max_frames = max_frames

    def analyze(self, path: Path) -> dict[str, Any]:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return {
                "status": "video_decode_error",
                "prediction": "inconclusive",
                "note": "Video could not be decoded for frame-level model analysis.",
                "frames": [],
            }
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            if count <= 0:
                return {
                    "status": "inconclusive_no_frames",
                    "prediction": "inconclusive",
                    "frames": [],
                    "note": "Video contains no readable frames.",
                }
            indexes = np.linspace(0, count - 1, min(self.max_frames, count), dtype=int)
            frame_results = []
            for index in sorted(set(indexes.tolist())):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if not ok:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.image_detector.analyze_pil_frame(
                    Image.fromarray(rgb), include_cam=False
                )
                frame_results.append({
                    "frame_index": int(index),
                    "timestamp_seconds": round(index / fps, 3) if fps > 0 else None,
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
                    "fake_median_across_face_frames": round(median_fake, 5)
                    if median_fake is not None else None,
                    "face_frames_analyzed": len(valid_scores),
                    "sampled_frames": len(frame_results),
                },
                "model_score_type": "median_frame_softmax_score_not_calibrated_probability",
                "uncertainty_band": (
                    "ambiguous" if median_fake is None or 0.35 < median_fake < 0.65
                    else "model_signal_separation"
                ),
                "frames": frame_results,
                "note": (
                    "Evenly sampled face-bearing frames were assessed independently. This is not temporal deepfake detection; "
                    "frame aggregation cannot establish whether an entire video is authentic."
                ),
            }
        finally:
            capture.release()
