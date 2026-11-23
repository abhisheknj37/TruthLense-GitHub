from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

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
    """Choose a deterministic source-disjoint split with both classes on both sides."""
    rng = np.random.default_rng(20261120)
    indices = np.arange(len(labels))
    test_count = max(2, min(len(labels) - 2, int(round(len(labels) * test_size))))
    best = None
    best_error = float("inf")
    for _ in range(512):
        shuffled = rng.permutation(indices)
        test_index = shuffled[:test_count]
        train_index = shuffled[test_count:]
        train_labels, test_labels = labels[train_index], labels[test_index]
        if len(np.unique(train_labels)) != 2 or len(np.unique(test_labels)) != 2:
            continue
        train_groups, test_groups = set(groups[train_index]), set(groups[test_index])
        if train_groups & test_groups:
            continue
        prevalence_error = abs(float(test_labels.mean()) - float(labels.mean()))
        size_error = abs(len(test_index) / len(labels) - test_size)
        error = prevalence_error + size_error
        if error < best_error:
            best = (train_index, test_index)
            best_error = error
    if best is None:
        raise SystemExit(
            "Could not produce a source-disjoint split with both classes in fit and held-out groups. "
            "Add more independent sources or revise the labelled dataset."
        )
    return best


def _auc_roc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if not positives or not negatives:
        raise ValueError("ROC AUC requires both classes.")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and sorted_scores[end + 1] == sorted_scores[index]:
            end += 1
        ranks[order[index:end + 1]] = (index + 1 + end + 1) / 2
        index = end + 1
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _metrics(labels: np.ndarray, probabilities: np.ndarray, bins: int):
    ece, reliability = expected_calibration_error(labels, probabilities, bins)
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    brier = float(np.mean((probabilities - labels) ** 2))
    logloss = float(np.mean(
        -(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
    ))
    return {
        "brier_score": brier,
        "log_loss": logloss,
        "roc_auc": _auc_roc(labels, probabilities),
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
    parsed_rows = []
    for row_number, row in enumerate(rows, start=2):
        try:
            label = int(float(row[args.label_column]))
            score = float(row[args.score_column])
            group = str(row[args.group_column]).strip()
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"Row {row_number}: invalid label, score or source group: {exc}") from exc
        if label not in {0, 1}:
            raise SystemExit(f"Row {row_number}: label must be 0 (real) or 1 (fake).")
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise SystemExit(f"Row {row_number}: score must be finite and in [0,1].")
        if not group:
            raise SystemExit(f"Row {row_number}: group_id cannot be empty.")
        parsed_rows.append((group, label, score))

    grouped: dict[str, dict[str, Any]] = {}
    for group, label, score in parsed_rows:
        entry = grouped.setdefault(group, {"labels": set(), "scores": []})
        entry["labels"].add(label)
        entry["scores"].append(score)
    conflicts = [group for group, value in grouped.items() if len(value["labels"]) != 1]
    if conflicts:
        raise SystemExit(
            "Each independent source group must have one ground-truth label. "
            f"Conflicting labels found for {conflicts[:5]}."
        )

    group_ids = sorted(grouped)
    if len(group_ids) < 60:
        raise SystemExit(
            "Need at least 60 independent source groups for a calibration candidate. "
            "Rows/frames/windows from one source count as one group."
        )
    labels = np.asarray([next(iter(grouped[group]["labels"])) for group in group_ids], dtype=np.int64)
    scores = np.asarray([float(np.median(grouped[group]["scores"])) for group in group_ids], dtype=np.float64)
    groups = np.asarray(group_ids, dtype=object)
    if set(np.unique(labels)) != {0, 1}:
        raise SystemExit("Source groups must contain both classes encoded as 0 (real) and 1 (fake).")

    train_index, test_index = _candidate_group_split(labels, groups, args.test_size)
    if len(test_index) < 20:
        raise SystemExit("Held-out set must contain at least 20 independent source groups.")
    train_labels, train_scores = labels[train_index], scores[train_index]
    held_out_labels, held_out_scores = labels[test_index], scores[test_index]
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError as exc:
        raise SystemExit(
            "Install optional calibration dependencies with: pip install -r requirements-forensics.txt"
        ) from exc
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
            "n_rows_total": int(len(parsed_rows)),
            "n_total": int(len(labels)),
            "n_fit": int(len(train_index)),
            "n_held_out": int(len(test_index)),
            "n_groups_total": int(len(group_ids)),
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
