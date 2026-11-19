from app.services.uncertainty_assessment import build_uncertainty_assessment


def test_ambiguous_model_score_is_high_uncertainty():
    result = build_uncertainty_assessment(
        "audio",
        {
            "status": "model_assessed",
            "prediction": "inconclusive",
            "scores": {"fake_median_across_windows": 0.51},
            "segments": [
                {"fake_model_score": 0.48},
                {"fake_model_score": 0.53},
            ],
        },
    )
    assert result["status"] == "assessed"
    assert result["uncertainty_level"] == "high"
    assert result["score_direction"] == "ambiguous"
    assert result["calibration_status"] == "not_calibrated"
    assert any("ambiguous band" in reason for reason in result["reasons"])


def test_disagreement_across_video_frames_increases_uncertainty():
    frames = [
        {"status": "model_assessed", "scores": {"fake_median_across_detected_faces": score}}
        for score in (0.18, 0.82, 0.86, 0.91, 0.84)
    ]
    result = build_uncertainty_assessment(
        "video",
        {"status": "model_assessed", "prediction": "fake_signal"},
        {
            "status": "model_assessed",
            "prediction": "fake_signal",
            "scores": {"fake_median_across_face_frames": 0.84, "face_frames_analyzed": 5},
            "frames": frames,
        },
    )
    assert result["sample_count"] == 5
    assert result["sample_disagreement"] is True
    assert result["uncertainty_level"] == "high"
    assert any("both real-direction and fake-direction" in reason for reason in result["reasons"])


def test_strong_separation_with_consistent_multiple_samples_has_lower_review_uncertainty():
    result = build_uncertainty_assessment(
        "audio",
        {
            "status": "model_assessed",
            "prediction": "fake_signal",
            "scores": {"fake_median_across_windows": 0.91},
            "segments": [{"fake_model_score": score} for score in (0.88, 0.91, 0.90, 0.92, 0.91)],
        },
    )
    assert result["uncertainty_level"] == "lower"
    assert result["signal_separation"] == "stronger"
    assert result["sample_support"] == "multiple_samples"
    assert result["sample_score_range"]["range"] == 0.04


def test_single_image_inference_is_marked_as_limited_support():
    result = build_uncertainty_assessment(
        "image",
        {
            "status": "model_assessed",
            "prediction": "real_signal",
            "scores": {"fake_median_across_detected_faces": 0.12},
            "aggregation": {"face_count": 1, "individual_fake_scores": [0.12]},
        },
    )
    assert result["uncertainty_level"] == "moderate"
    assert result["sample_count"] == 1
    assert result["sample_support"] == "single_sample"
    assert any("Only one inference sample" in reason for reason in result["reasons"])


def test_missing_model_score_is_not_reported_as_authenticity():
    result = build_uncertainty_assessment(
        "video",
        {"status": "inconclusive_no_face_frames", "prediction": "inconclusive"},
        {"status": "inconclusive_no_face_frames", "scores": {}, "frames": []},
    )
    assert result["status"] == "not_assessable"
    assert result["uncertainty_level"] == "high"
    assert result["raw_model_score"] is None
    assert "not be interpreted as evidence of authenticity" in result["interpretation"]
