from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _read_artifact(path: Path, model_id: str | None, pipeline: str) -> dict[str, Any]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        mapping = artifact["calibration_mapping"]
        xs = [float(value) for value in mapping["x_thresholds"]]
        ys = [float(value) for value in mapping["y_thresholds"]]
        dataset = artifact["dataset"]
        metrics = artifact["metrics"]["calibrated_held_out"]
        if len(xs) < 2 or len(xs) != len(ys):
            raise ValueError("Invalid isotonic mapping arrays.")
        if any(xs[i] > xs[i + 1] or ys[i] > ys[i + 1] for i in range(len(xs) - 1)):
            raise ValueError("Calibration mapping is not monotonic.")
        if not all(0 <= value <= 1 for value in xs + ys):
            raise ValueError("Calibration mapping values must be in [0,1].")
        if artifact.get("model_id") != model_id or artifact.get("pipeline") != pipeline:
            return {"status": "model_or_pipeline_mismatch", "artifact": path.name}
        if artifact.get("approval_status") != "approved" or artifact.get("deployment_approved") is not True:
            return {"status": "candidate_not_approved", "artifact": path.name}
        if dataset.get("n_held_out", 0) < 20 or dataset.get("n_groups_held_out", 0) < 5:
            return {"status": "insufficient_held_out_validation", "artifact": path.name}
        if not dataset.get("sha256") or not dataset.get("group_column"):
            return {"status": "missing_dataset_provenance", "artifact": path.name}
        if not artifact.get("approved_by") or not artifact.get("approved_at"):
            return {"status": "approval_metadata_missing", "artifact": path.name}
        return {
            "status": "approved",
            "artifact": path.name,
            "model_id": model_id,
            "pipeline": pipeline,
            "x_thresholds": xs,
            "y_thresholds": ys,
            "dataset_sha256": dataset["sha256"],
            "held_out_count": int(dataset["n_held_out"]),
            "held_out_group_count": int(dataset["n_groups_held_out"]),
            "metrics": metrics,
            "approved_by": artifact["approved_by"],
            "approved_at": artifact["approved_at"],
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "invalid_artifact", "detail": str(exc)[:180], "artifact": path.name}


def load_approved_calibration(model_id: str | None, pipeline: str) -> dict[str, Any]:
    configured = os.getenv("TRUTHLENS_CALIBRATION_ARTIFACT", "").strip()
    if not configured:
        return {"status": "not_configured"}
    path = Path(configured).expanduser()
    if not path.exists():
        return {"status": "artifact_missing", "artifact": path.name}
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(path.glob("*.json"))
    else:
        return {"status": "artifact_missing", "artifact": path.name}
    if not candidates:
        return {"status": "no_artifacts_in_directory", "artifact": path.name}

    outcomes = [_read_artifact(item, model_id, pipeline) for item in candidates]
    approved = next((item for item in outcomes if item["status"] == "approved"), None)
    if approved:
        return approved
    priority = (
        "candidate_not_approved",
        "insufficient_held_out_validation",
        "missing_dataset_provenance",
        "approval_metadata_missing",
        "invalid_artifact",
        "model_or_pipeline_mismatch",
    )
    for status in priority:
        selected = next((item for item in outcomes if item["status"] == status), None)
        if selected:
            return selected
    return {"status": "no_matching_calibration", "artifact": path.name}


def apply_approved_calibration(score: float, calibration: dict[str, Any]) -> float | None:
    if calibration.get("status") != "approved":
        return None
    xs = calibration["x_thresholds"]
    ys = calibration["y_thresholds"]
    import numpy as np
    result = float(np.interp(float(score), xs, ys))
    return max(0.0, min(1.0, result))
