from __future__ import annotations

from typing import Any

from app.services.calibration_registry import apply_approved_calibration, load_approved_calibration

REAL_THRESHOLD = 0.35
FAKE_THRESHOLD = 0.65


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if 0.0 <= result <= 1.0 else None


def _extract_score(media_type: str, model: dict[str, Any], video: dict[str, Any]) -> float | None:
    scores = (video if media_type == "video" else model).get("scores") or {}
    keys = {
        "image": ("fake_median_across_detected_faces", "fake_median_across_faces", "fake", "fake_score", "ai_generated"),
        "video": ("fake_median_across_face_frames", "fake_median_across_frames", "fake"),
        "audio": ("fake_median_across_windows", "fake_median", "fake"),
    }.get(media_type, ("fake",))
    for key in keys:
        score = _number(scores.get(key))
        if score is not None:
            return score
    return None


def _extract_samples(media_type: str, model: dict[str, Any], video: dict[str, Any], score: float | None) -> list[float]:
    if media_type == "image":
        values = (model.get("aggregation") or {}).get("individual_fake_scores") or []
        if not values and isinstance(model.get("faces"), list):
            values = [(face.get("scores") or {}).get("fake") for face in model["faces"]]
    elif media_type == "video":
        values = [
            (frame.get("scores") or {}).get("fake_median_across_detected_faces")
            for frame in video.get("frames", [])
            if frame.get("status") == "model_assessed"
        ]
    elif media_type == "audio":
        values = [segment.get("fake_model_score") for segment in model.get("segments", [])]
    else:
        values = []

    samples = [parsed for value in values if (parsed := _number(value)) is not None]
    if samples:
        return samples

    if score is None:
        return []
    if media_type == "image":
        count = int((model.get("aggregation") or {}).get("face_count") or 1)
    elif media_type == "video":
        count = int((video.get("scores") or {}).get("face_frames_analyzed") or 1)
    elif media_type == "audio":
        count = int((model.get("voice_synthesis_review") or {}).get("windows_assessed") or 1)
    else:
        count = 1
    return [score] * max(1, min(count, 100))


def build_uncertainty_assessment(
    media_type: str,
    model_assessment: dict[str, Any] | None,
    video_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarise score separation and evidence coverage; never present it as calibrated confidence."""
    model = model_assessment or {}
    video = video_analysis or {}
    status = str((video if media_type == "video" else model).get("status") or model.get("status") or "not_assessed")
    prediction = str((video if media_type == "video" else model).get("prediction") or model.get("prediction") or "inconclusive")
    score = _extract_score(media_type, model, video)
    samples = _extract_samples(media_type, model, video, score)
    count = len(samples)

    if score is None or status != "model_assessed":
        return {
            "status": "not_assessable",
            "uncertainty_level": "high",
            "signal_separation": "unavailable",
            "raw_model_score": None,
            "score_direction": "not_available",
            "sample_count": count,
            "sample_support": "no_assessable_model_score",
            "sample_score_range": None,
            "sample_disagreement": False,
            "reasons": [
                "No usable model score was returned for this evidence item.",
                "Treat the result as inconclusive and review the original evidence.",
            ],
            "calibration_status": "not_calibrated",
            "method": "No statistical confidence interval or calibrated probability is available.",
            "interpretation": (
                "This assessment is unavailable because the modality model did not return an assessable score. "
                "An unavailable score must not be interpreted as evidence of authenticity."
            ),
        }

    if prediction == "fake_signal":
        zone = "fake_direction"
        boundary_margin = max(0.0, score - FAKE_THRESHOLD)
    elif prediction == "real_signal":
        zone = "real_direction"
        boundary_margin = max(0.0, REAL_THRESHOLD - score)
    else:
        zone = "ambiguous"
        boundary_margin = 0.0

    neutral_separation = abs(score - 0.5)
    if zone == "ambiguous" or neutral_separation < 0.10:
        separation = "weak"
    elif neutral_separation < 0.25:
        separation = "moderate"
    else:
        separation = "stronger"

    sample_min = min(samples) if samples else score
    sample_max = max(samples) if samples else score
    spread = round(sample_max - sample_min, 5)
    directions = {
        "fake" if value >= FAKE_THRESHOLD else
        "real" if value <= REAL_THRESHOLD else "ambiguous"
        for value in samples
    }
    disagreement = "fake" in directions and "real" in directions

    if count <= 1:
        support = "single_sample"
    elif count <= 3:
        support = "limited_samples"
    else:
        support = "multiple_samples"

    reasons: list[str] = []
    if zone == "ambiguous":
        reasons.append(
            f"The aggregate score ({score:.3f}) is in the ambiguous band between "
            f"{REAL_THRESHOLD:.2f} and {FAKE_THRESHOLD:.2f}; the model did not separate the classes."
        )
    else:
        reasons.append(
            f"The aggregate score is {boundary_margin:.3f} beyond the decision threshold for the {zone.replace('_', ' ')} zone."
        )

    if disagreement:
        reasons.append("Individual faces, frames or audio windows include both real-direction and fake-direction scores.")
    elif spread >= 0.25:
        reasons.append(f"Within-sample scores vary substantially (range {spread:.3f}).")
    elif spread >= 0.12:
        reasons.append(f"Some within-sample score variation is present (range {spread:.3f}).")

    if count == 1:
        reasons.append("Only one inference sample was available; within-file score stability cannot be estimated.")
    elif count <= 3:
        reasons.append(f"Only {count} inference samples were assessed; support is limited.")
    else:
        reasons.append(f"{count} inference samples were assessed; this describes only the sampled faces, frames or windows.")

    if zone == "ambiguous" or disagreement or spread >= 0.25 or boundary_margin < 0.05:
        uncertainty = "high"
    elif count <= 1 or spread >= 0.12 or boundary_margin < 0.15:
        uncertainty = "moderate"
    else:
        uncertainty = "lower"

    sample_label = {
        "single_sample": "Single inference sample",
        "limited_samples": "Limited sample coverage",
        "multiple_samples": "Multiple sampled inference units",
    }[support]

    model_id = str((video if media_type == "video" else model).get("model_name") or model.get("model_name") or "")
    calibration = load_approved_calibration(model_id or None, media_type)
    calibrated_probability = apply_approved_calibration(score, calibration)
    calibration_status = calibration.get("status", "not_configured")
    return {
        "status": "assessed",
        "uncertainty_level": uncertainty,
        "signal_separation": separation,
        "raw_model_score": round(score, 5),
        "score_direction": zone,
        "distance_beyond_decision_threshold": round(boundary_margin, 5),
        "decision_thresholds": {"real_at_or_below": REAL_THRESHOLD, "fake_at_or_above": FAKE_THRESHOLD},
        "sample_count": count,
        "sample_support": support,
        "sample_support_label": sample_label,
        "sample_score_range": {"minimum": round(sample_min, 5), "maximum": round(sample_max, 5), "range": spread},
        "sample_disagreement": disagreement,
        "reasons": reasons,
        "calibration_status": "approved_calibration_applied" if calibrated_probability is not None else calibration_status,
        "calibrated_fake_probability": round(calibrated_probability, 5) if calibrated_probability is not None else None,
        "calibration_artifact": calibration.get("artifact"),
        "calibration_model_id": calibration.get("model_id"),
        "calibration_held_out_count": calibration.get("held_out_count"),
        "calibration_held_out_group_count": calibration.get("held_out_group_count"),
        "calibration_approved_by": calibration.get("approved_by"),
        "calibration_approved_at": calibration.get("approved_at"),
        "method": (
            "Heuristic review of aggregate score location, distance beyond the decision threshold, "
            "within-sample score range and sample count. This is not a probability interval or confidence estimate. "
            + ("An approved, model-specific isotonic calibration mapping is applied to the aggregate fake-direction score." if calibrated_probability is not None else "No approved model-specific calibration mapping was available.")
        ),
        "interpretation": (
            "The model score is a relative classifier output. The uncertainty level describes score separation "
            "and the amount/consistency of sampled evidence only; it does not measure the probability that the "
            "prediction is correct. " + (f"The approved held-out calibration maps the aggregate score to {calibrated_probability:.3f}; this estimate is specific to the recorded model, pipeline and dataset." if calibrated_probability is not None else "The raw score is uncalibrated and must not be presented as a confidence percentage.")
        ),
    }
