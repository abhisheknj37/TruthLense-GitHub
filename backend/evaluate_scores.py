"""Evaluate saved TruthLens fake-direction scores against a labelled CSV.

CSV columns: label,score (label accepts real/0 or fake/1; score is fake-direction score).
This evaluates supplied predictions only; it does not train or validate a model checkpoint.
"""
import argparse
import csv
import json
from pathlib import Path


def auc_roc(labels, scores):
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranked = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][1] == ranked[i][1]:
            j += 1
        average_rank = (i + 1 + j + 1) / 2
        for k in range(i, j + 1):
            ranks[ranked[k][0]] = average_rank
        i = j + 1
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TruthLens fake-direction scores on labelled CSV rows.")
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--output", type=Path, default=Path("evaluation-results.json"))
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    labels, scores = [], []
    with args.csv_file.open(newline="", encoding="utf-8-sig") as source:
        for row_num, row in enumerate(csv.DictReader(source), start=2):
            label = row.get("label", "").strip().lower()
            try:
                score = float(row.get("score", "").strip())
            except ValueError:
                raise SystemExit(f"Row {row_num}: score must be numeric")
            if label not in {"real", "fake", "0", "1"} or not 0 <= score <= 1:
                raise SystemExit(f"Row {row_num}: label must be real/fake or 0/1 and score must be in [0,1]")
            labels.append(1 if label in {"fake", "1"} else 0)
            scores.append(score)
    if not labels:
        raise SystemExit("CSV contains no data rows")
    predictions = [int(score >= args.threshold) for score in scores]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result = {
        "sample_count": len(labels), "threshold": args.threshold,
        "accuracy": (tp + tn) / len(labels), "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "roc_auc": auc_roc(labels, scores),
        "brier_score": sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels),
        "confusion_matrix": {"true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp},
        "note": "Metrics apply only to this supplied labelled CSV and threshold. Brier score is a calibration metric only when scores are probability-like and representative.",
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\\nSaved: {args.output}")


if __name__ == "__main__":
    main()
