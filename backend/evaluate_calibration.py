from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def expected_calibration_error(labels, probabilities, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = len(labels)
    value = 0.0
    details = []
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1] if index < bins - 1 else probabilities <= edges[index + 1])
        if mask.any():
            confidence = float(probabilities[mask].mean())
            accuracy = float(labels[mask].mean())
            value += int(mask.sum()) / total * abs(accuracy - confidence)
            details.append({"bin": index, "count": int(mask.sum()), "mean_confidence": confidence, "empirical_positive_rate": accuracy})
    return value, details


def main():
    parser = argparse.ArgumentParser(description="Fit and evaluate an isotonic calibration mapping.")
    parser.add_argument("csv_path")
    parser.add_argument("--output", default="calibration-results.json")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    with open(args.csv_path, newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    labels = np.asarray([int(float(row[args.label_column])) for row in rows])
    scores = np.asarray([float(row[args.score_column]) for row in rows])
    if len(labels) < 30 or len(np.unique(labels)) != 2 or np.any((scores < 0) | (scores > 1)):
        raise SystemExit("Need >=30 labelled examples, both classes, labels 0/1 and scores in [0,1].")
    split = int(len(labels) * .7)
    if len(np.unique(labels[:split])) != 2 or len(np.unique(labels[split:])) != 2:
        raise SystemExit("Both train and held-out partitions must contain both classes. Shuffle by independent source/case first.")
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(scores[:split], labels[:split])
    calibrated = calibrator.predict(scores[split:])
    held_out = labels[split:]
    ece, bins = expected_calibration_error(held_out, calibrated, args.bins)
    result = {
        "status": "evaluated",
        "n_total": int(len(labels)),
        "n_calibration_fit": int(split),
        "n_held_out": int(len(held_out)),
        "metrics_on_held_out": {
            "brier_score": float(brier_score_loss(held_out, calibrated)),
            "log_loss": float(log_loss(held_out, np.clip(calibrated, 1e-6, 1 - 1e-6))),
            "roc_auc": float(roc_auc_score(held_out, calibrated)),
            "expected_calibration_error": float(ece),
        },
        "reliability_bins": bins,
        "calibration_mapping": {"x_thresholds": calibrator.X_thresholds_.tolist(), "y_thresholds": calibrator.y_thresholds_.tolist()},
        "warning": "Metrics apply only to this dataset and split. Avoid leakage by splitting by source/video/person and validate on an independent external dataset before claiming generalization.",
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
