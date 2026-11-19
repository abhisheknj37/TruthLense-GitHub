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
    assert result["calibration_status"] == "not_configured"
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


def test_runtime_calibration_rejects_candidate_artifact(tmp_path, monkeypatch):
    import json
    from app.services.calibration_registry import load_approved_calibration, apply_approved_calibration

    artifact = {
        "approval_status": "candidate",
        "deployment_approved": False,
        "model_id": "fixture-model",
        "pipeline": "audio",
        "dataset": {"n_held_out": 30, "n_groups_held_out": 8, "sha256": "a" * 64, "group_column": "group_id"},
        "metrics": {"calibrated_held_out": {"brier_score": 0.1}},
        "calibration_mapping": {"x_thresholds": [0.0, 1.0], "y_thresholds": [0.0, 1.0]},
    }
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    monkeypatch.setenv("TRUTHLENS_CALIBRATION_ARTIFACT", str(path))
    calibration = load_approved_calibration("fixture-model", "audio")
    assert calibration["status"] == "candidate_not_approved"
    assert apply_approved_calibration(0.5, calibration) is None


def test_runtime_calibration_requires_exact_model_and_pipeline_and_approval(tmp_path, monkeypatch):
    import json
    from app.services.calibration_registry import load_approved_calibration, apply_approved_calibration

    artifact = {
        "approval_status": "approved",
        "deployment_approved": True,
        "approved_by": "reviewer",
        "approved_at": "2026-11-20T00:00:00+00:00",
        "model_id": "fixture-model",
        "pipeline": "audio",
        "dataset": {"n_held_out": 30, "n_groups_held_out": 8, "sha256": "a" * 64, "group_column": "group_id"},
        "metrics": {"calibrated_held_out": {"brier_score": 0.1}},
        "calibration_mapping": {"x_thresholds": [0.0, 0.5, 1.0], "y_thresholds": [0.0, 0.2, 0.9]},
    }
    path = tmp_path / "approved.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    monkeypatch.setenv("TRUTHLENS_CALIBRATION_ARTIFACT", str(path))

    mismatch = load_approved_calibration("other-model", "audio")
    assert mismatch["status"] == "model_or_pipeline_mismatch"
    calibration = load_approved_calibration("fixture-model", "audio")
    assert calibration["status"] == "approved"
    assert apply_approved_calibration(0.5, calibration) == 0.2
    assert apply_approved_calibration(0.25, calibration) == 0.1


def test_calibration_directory_selects_exact_model_and_pipeline(tmp_path, monkeypatch):
    import json
    from app.services.calibration_registry import load_approved_calibration

    directory = tmp_path / "calibration"
    directory.mkdir()
    base = {
        "approval_status": "approved",
        "deployment_approved": True,
        "approved_by": "reviewer",
        "approved_at": "2026-11-20T00:00:00+00:00",
        "dataset": {"n_held_out": 30, "n_groups_held_out": 8, "sha256": "a" * 64, "group_column": "group_id"},
        "metrics": {"calibrated_held_out": {"brier_score": 0.1}},
        "calibration_mapping": {"x_thresholds": [0.0, 0.5, 1.0], "y_thresholds": [0.0, 0.2, 0.9]},
    }
    for name, model_id, pipeline in (
        ("audio.json", "audio-model", "audio"),
        ("image.json", "image-model", "image"),
        ("video.json", "image-model", "video"),
    ):
        directory.joinpath(name).write_text(json.dumps({
            **base, "model_id": model_id, "pipeline": pipeline
        }), encoding="utf-8")
    monkeypatch.setenv("TRUTHLENS_CALIBRATION_ARTIFACT", str(directory))
    assert load_approved_calibration("audio-model", "audio")["artifact"] == "audio.json"
    assert load_approved_calibration("image-model", "video")["artifact"] == "video.json"
    assert load_approved_calibration("image-model", "audio")["status"] == "model_or_pipeline_mismatch"
