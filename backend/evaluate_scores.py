"""Evaluate saved TruthLens fake-direction scores against a labelled CSV.

Required columns: label,score. Optional --group-column computes source-level
metrics by taking the median score within each independent source group.
The script evaluates supplied predictions only; it does not validate a model.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


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


def calculate_metrics(labels, scores, threshold):
    predictions = [int(score >= threshold) for score in scores]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "sample_count": len(labels),
        "threshold": threshold,
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "roc_auc": auc_roc(labels, scores),
        "brier_score": sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels),
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
    }


def aggregate_groups(rows, group_column):
    grouped = defaultdict(list)
    for row in rows:
        group = row.get(group_column, "").strip()
        if not group:
            raise SystemExit(f"Every row must have a non-empty {group_column!r} value.")
        grouped[group].append(row)
    labels, scores = [], []
    for group, items in sorted(grouped.items()):
        group_labels = {item["label"] for item in items}
        if len(group_labels) != 1:
            raise SystemExit(
                f"Group {group!r} contains conflicting labels. "
                "Use a grouping key that represents one independently labelled source."
            )
        labels.append(next(iter(group_labels)))
        scores.append(float(median(item["score"] for item in items)))
    return labels, scores, len(grouped)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate labelled TruthLens scores; use source groups to avoid frame/window over-counting."
    )
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--output", type=Path, default=Path("evaluation-results.json"))
    parser.add_argument("--group-column", default=None, help="Optional independent source ID such as video_id, image_id or speaker_session.")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    rows = []
    with args.csv_file.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"label", "score"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit("CSV must include label,score columns.")
        if args.group_column and args.group_column not in reader.fieldnames:
            raise SystemExit(f"Group column {args.group_column!r} is missing from the CSV.")
        for row_num, row in enumerate(reader, start=2):
            label_text = row.get("label", "").strip().lower()
            try:
                score = float(row.get("score", "").strip())
            except (TypeError, ValueError):
                raise SystemExit(f"Row {row_num}: score must be numeric.")
            if label_text not in {"real", "fake", "0", "1"} or not math.isfinite(score) or not 0 <= score <= 1:
                raise SystemExit(f"Row {row_num}: label must be real/fake or 0/1 and score finite in [0,1].")
            rows.append({
                "label": 1 if label_text in {"fake", "1"} else 0,
                "score": score,
                args.group_column: row.get(args.group_column, "") if args.group_column else "",
            })
    if not rows:
        raise SystemExit("CSV contains no data rows.")

    row_labels = [row["label"] for row in rows]
    row_scores = [row["score"] for row in rows]
    result = {
        "row_level": calculate_metrics(row_labels, row_scores, args.threshold),
        "group_aggregation": None,
        "warning": (
            "Row-level metrics may over-count near-duplicate frames/windows from the same original source. "
            "Do not treat rows as independent samples unless the dataset construction supports that assumption."
        ),
    }
    if args.group_column:
        group_labels, group_scores, group_count = aggregate_groups(rows, args.group_column)
        result["group_aggregation"] = {
            "group_column": args.group_column,
            "method": "median fake-direction score within each source group",
            "group_count": group_count,
            "metrics": calculate_metrics(group_labels, group_scores, args.threshold),
        }
        result["warning"] = (
            "Group-level metrics give each independent source group one vote. The choice of group must prevent "
            "leakage (for example, all frames/windows from one video or all samples from one speaker/session). "
            "Results still apply only to this supplied dataset."
        )
    result["input"] = {
        "file": args.csv_file.name,
        "row_count": len(rows),
        "threshold": args.threshold,
        "group_column": args.group_column,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
