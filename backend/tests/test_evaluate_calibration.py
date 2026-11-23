import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from evaluate_calibration import _candidate_group_split


def test_calibration_source_group_split_has_no_leakage_and_both_labels():
    labels = np.asarray([0] * 40 + [1] * 40)
    groups = np.asarray([f"source-{index}" for index in range(80)], dtype=object)
    train, held = _candidate_group_split(labels, groups, 0.25)
    assert set(groups[train]).isdisjoint(set(groups[held]))
    assert set(labels[train]) == {0, 1}
    assert set(labels[held]) == {0, 1}


def test_calibration_script_rejects_conflicting_labels_for_same_source(tmp_path):
    path = tmp_path / "labels.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["label", "score", "group_id"])
        for index in range(60):
            label = 0 if index % 2 == 0 else 1
            writer.writerow([label, 0.2 if label == 0 else 0.8, f"source-{index}"])
        writer.writerow([1, 0.9, "source-0"])
    result = subprocess.run(
        [sys.executable, "evaluate_calibration.py", str(path),
         "--model-id", "fixture", "--pipeline", "image"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Conflicting labels" in result.stderr + result.stdout
