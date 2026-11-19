from pathlib import Path

import pytest

from evaluate_scores import aggregate_groups, calculate_metrics


def test_group_aggregation_counts_each_source_once_with_median():
    rows = [
        {"label": 1, "score": 0.9, "video_id": "clip-a"},
        {"label": 1, "score": 0.8, "video_id": "clip-a"},
        {"label": 0, "score": 0.1, "video_id": "clip-b"},
        {"label": 0, "score": 0.2, "video_id": "clip-b"},
    ]
    labels, scores, count = aggregate_groups(rows, "video_id")
    assert count == 2
    assert labels == [1, 0]
    assert scores == [0.85, 0.15]
    metrics = calculate_metrics(labels, scores, 0.65)
    assert metrics["sample_count"] == 2
    assert metrics["accuracy"] == 1.0


def test_group_aggregation_rejects_conflicting_source_labels():
    rows = [
        {"label": 0, "score": 0.2, "video_id": "same-clip"},
        {"label": 1, "score": 0.8, "video_id": "same-clip"},
    ]
    with pytest.raises(SystemExit, match="conflicting labels"):
        aggregate_groups(rows, "video_id")


def test_group_aggregation_rejects_missing_group_identifier():
    with pytest.raises(SystemExit, match="non-empty"):
        aggregate_groups([{"label": 0, "score": 0.2, "video_id": ""}], "video_id")
