from pathlib import Path

import cv2
import numpy as np

from app.services.image_comparison import compare_images


def test_image_comparison_requires_two_readable_images(tmp_path: Path):
    result = compare_images([])
    assert result[0]["status"] == "insufficient_images"


def test_identical_images_are_flagged_for_similarity_review(tmp_path: Path):
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (140, 140), (255, 255, 255), 4)
    cv2.circle(image, (80, 80), 35, (255, 255, 255), 3)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    assert cv2.imwrite(str(first), image)
    assert cv2.imwrite(str(second), image)
    result = compare_images([first, second])
    assert len(result) == 1
    assert result[0]["status"] == "high_visual_similarity_review"
    assert "not a validated" in result[0]["limitations"]
