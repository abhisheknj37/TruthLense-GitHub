from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _dhash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def compare_images(paths: list[Path]) -> list[dict[str, Any]]:
    """Compare uploaded images for near-duplicates; not a manipulation detector."""
    results = []
    loaded = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            loaded.append((path.name, image))
    if len(loaded) < 2:
        return [{
            "type": "image_to_image_comparison",
            "status": "insufficient_images",
            "detail": "Upload at least two readable images to compare visual similarity.",
        }]

    hashes = [(name, image, _dhash(image)) for name, image in loaded]
    orb = cv2.ORB_create(nfeatures=800)
    for index, (name_a, image_a, hash_a) in enumerate(hashes):
        for name_b, image_b, hash_b in hashes[index + 1:]:
            distance = (hash_a ^ hash_b).bit_count()
            gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
            gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
            key_a, desc_a = orb.detectAndCompute(gray_a, None)
            key_b, desc_b = orb.detectAndCompute(gray_b, None)
            matches = []
            if desc_a is not None and desc_b is not None:
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
                pairs = matcher.knnMatch(desc_a, desc_b, k=2)
                matches = [first for first, second in pairs if first.distance < 0.72 * second.distance]
            denominator = max(1, min(len(key_a or []), len(key_b or [])))
            ratio = len(matches) / denominator
            status = "high_visual_similarity_review" if distance <= 8 or ratio >= 0.35 else "no_strong_similarity_signal"
            results.append({
                "type": "image_to_image_comparison",
                "status": status,
                "image_a": name_a,
                "image_b": name_b,
                "difference_hash_distance": distance,
                "orb_good_match_count": len(matches),
                "orb_match_ratio": round(ratio, 4),
                "method": "difference hash and ORB local-feature matching",
                "detail": (
                    "The images appear visually similar or may share visual content. This does not establish common origin, manipulation or authenticity."
                    if status == "high_visual_similarity_review" else
                    "No strong near-duplicate signal was found. Cropping, resizing, edits and compression can reduce similarity."
                ),
                "limitations": "Similarity screening only; not a validated copy-move, splicing or image-forgery detector.",
            })
    return results
