from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


def expected_calibration_error(labels, probabilities, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = len(labels)
    value = 0.0
    details = []
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (
            probabilities < edges[index + 1] if index < bins - 1 else probabilities <= edges[index + 1]
        )
        if mask.any():
            mean_score = float(probabilities[mask].mean())
            positive_rate = float(labels[mask].mean())
            value += int(mask.sum()) / total * abs(positive_rate - mean_score)
            details.append({
                "bin": index,
                "count": int(mask.sum()),
                "mean_score": mean_score,
                "empirical_positive_rate": positive_rate,
            })
    return float(value), details


def _candidate_group_split(labels: np.ndarray, groups: np.ndarray, test_size: float):
    """Select one deterministic group split with both labels on each side and no group leakage."""
    splitter = GroupShuffleSplit(n_splits=128, test_size=test_size, random_state=20261120)
    best = None
    best_score = float("inf")
    for train_index, test_index in splitter.split(np.zeros(len(labels)), labels, groups):
        train_labels, test_labels = labels[train_index], labels[test_index]
        if len(np.unique(train_labels)) != 2 or len(np.unique(test_labels)) != 2:
            continue
        train_groups, test_groups = set(groups[train_index]), set(groups[test_index])
        if train_groups & test_groups:
            continue
        # Prefer a held-out prevalence close to the complete dataset while preserving group isolation.
        prevalence_error = abs(float(test_labels.mean()) - float(labels.mean()))
        size_error = abs(len(test_index) / len(labels) - test_size)
        score = prevalence_error + size_error
        if score < best_score:
            best = (train_index, test_index)
            best_score = score
    if best is None:
        raise SystemExit(
            "Could not produce a leakage-free group split with both classes in train and hold-out. "
            "Add more independent source groups or revise the dataset."
        )
    return best


def _metrics(labels: np.ndarray, probabilities: np.ndarray, bins: int):
    ece, reliability = expected_calibration_error(labels, probabilities, bins)
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return {
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "expected_calibration_error": ece,
        "reliability_bins": reliability,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fit a candidate isotonic map with source/group-separated held-out evaluation."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("calibration-candidate.json"))
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--group-column", default="group_id")
    parser.add_argument("--model-id", required=True, help="Exact checkpoint/model identifier.")
    parser.add_argument("--pipeline", required=True, choices=["image", "video", "audio"])
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    if not 0.15 <= args.test_size <= 0.4:
        parser.error("--test-size must be between 0.15 and 0.40")
    if not 5 <= args.bins <= 20:
        parser.error("--bins must be between 5 and 20")

    raw_bytes = args.csv_path.read_bytes()
    dataset_hash = hashlib.sha256(raw_bytes).hexdigest()
    with args.csv_path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    required_columns = {args.score_column, args.label_column, args.group_column}
    if not rows or not required_columns.issubset(rows[0]):
        raise SystemExit(
            f"CSV must contain columns: {', '.join(sorted(required_columns))}. "
            "The group column is mandatory to prevent source leakage."
        )
    try:
        labels = np.asarray([int(float(row[args.label_column])) for row in rows], dtype=np.int64)
        scores = np.asarray([float(row[args.score_column]) for row in rows], dtype=np.float64)
        groups = np.asarray([str(row[args.group_column]).strip() for row in rows], dtype=object)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"CSV contains an invalid label, score or group: {exc}") from exc

    if len(labels) < 60:
        raise SystemExit("Need at least 60 independent labelled observations for a calibration candidate.")
    if set(np.unique(labels)) != {0, 1}:
        raise SystemExit("Labels must contain both classes encoded as 0 (real) and 1 (fake).")
    if np.any(~np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise SystemExit("All scores must be finite and lie in [0,1].")
    if any(not group for group in groups):
        raise SystemExit("Every row needs a non-empty group_id such as source video, speaker or original image.")
    unique_groups = set(groups)
    if len(unique_groups) < 10:
        raise SystemExit("Need at least 10 independent groups; rows from one source do not count as independent samples.")

    train_index, test_index = _candidate_group_split(labels, groups, args.test_size)
    if len(test_index) < 20 or len(set(groups[test_index])) < 5:
        raise SystemExit("Held-out set must contain at least 20 rows from at least 5 independent groups.")

    train_labels, train_scores = labels[train_index], scores[train_index]
    held_out_labels, held_out_scores = labels[test_index], scores[test_index]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(train_scores, train_labels)
    calibrated = calibrator.predict(held_out_scores)
    raw_metrics = _metrics(held_out_labels, held_out_scores, args.bins)
    calibrated_metrics = _metrics(held_out_labels, calibrated, args.bins)
    train_groups, held_groups = set(groups[train_index]), set(groups[test_index])
    if train_groups & held_groups:
        raise SystemExit("Internal error: source groups leaked between fit and held-out partitions.")

    result = {
        "artifact_version": "1.0",
        "status": "candidate_evaluated",
        "approval_status": "candidate",
        "deployment_approved": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "pipeline": args.pipeline,
        "score_column": args.score_column,
        "label_column": args.label_column,
        "group_column": args.group_column,
        "dataset": {
            "filename": args.csv_path.name,
            "sha256": dataset_hash,
            "n_total": int(len(labels)),
            "n_fit": int(len(train_index)),
            "n_held_out": int(len(test_index)),
            "n_groups_total": int(len(unique_groups)),
            "n_groups_fit": int(len(train_groups)),
            "n_groups_held_out": int(len(held_groups)),
            "fit_group_ids": sorted(str(value) for value in train_groups),
            "held_out_group_ids": sorted(str(value) for value in held_groups),
        },
        "metrics": {
            "raw_held_out": raw_metrics,
            "calibrated_held_out": calibrated_metrics,
            "delta_brier_score": raw_metrics["brier_score"] - calibrated_metrics["brier_score"],
            "delta_log_loss": raw_metrics["log_loss"] - calibrated_metrics["log_loss"],
            "delta_ece": raw_metrics["expected_calibration_error"] - calibrated_metrics["expected_calibration_error"],
        },
        "calibration_mapping": {
            "method": "isotonic_regression",
            "x_thresholds": calibrator.X_thresholds_.tolist(),
            "y_thresholds": calibrator.y_thresholds_.tolist(),
        },
        "approval_instructions": (
            "Review dataset provenance, representativeness, group split, reliability bins and metrics. "
            "Only after independent review may an authorized maintainer set approval_status='approved', "
            "deployment_approved=true, approved_by and approved_at. Keep a separate artifact per exact "
            "model checkpoint and modality pipeline. Candidate artifacts are not loaded by TruthLens."
        ),
        "warning": (
            "Held-out metrics apply only to this supplied dataset and split. Group splitting reduces source leakage "
            "but does not guarantee external generalization. Do not claim calibration without a representative "
            "independent validation corpus and documented approval."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "approval_status": result["approval_status"],
        "dataset": result["dataset"],
        "metrics": result["metrics"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
