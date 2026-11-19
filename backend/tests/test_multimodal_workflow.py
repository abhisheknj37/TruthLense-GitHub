import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.auth import get_current_user
from app.main import app
from app.routes import analysis as analysis_module


def test_full_case_upload_runs_image_video_audio_and_text_workflow(monkeypatch):
    """Exercise the complete API orchestration with deterministic fake model outputs."""
    monkeypatch.setattr(
        analysis_module,
        "inspect_media",
        lambda path: {
            "media_type": {".jpg": "image", ".mp4": "video", ".wav": "audio"}[Path(path).suffix.lower()],
            "metadata": {"fixture": True},
            "observations": [],
            "sampled_frames": [{"frame_index": 0, "timestamp_seconds": 0.0, "prediction": "fake_signal"}]
            if Path(path).suffix.lower() == ".mp4" else [],
            "forensic_parameters": {},
        },
    )
    monkeypatch.setattr(
        analysis_module.image_detector,
        "analyze_image",
        lambda *args: {
            "status": "model_assessed",
            "model_name": "fixture-image-model",
            "prediction": "real_signal",
            "scores": {"fake_score": 0.12, "real_score": 0.88},
            "model_score_type": "fixture_softmax",
        },
    )
    monkeypatch.setattr(
        analysis_module.video_detector,
        "analyze",
        lambda path: {
            "status": "model_assessed",
            "model_name": "fixture-video-frame-model",
            "prediction": "fake_signal",
            "scores": {"fake_median_across_face_frames": 0.81},
            "uncertainty_band": "model_score_separated",
            "model_score_type": "fixture_softmax",
            "note": "Fixture video model output.",
        },
    )
    monkeypatch.setattr(
        analysis_module.audio_detector,
        "analyze",
        lambda path: {
            "status": "model_assessed",
            "model_name": "fixture-audio-model",
            "prediction": "fake_signal",
            "scores": {"fake_median_across_windows": 0.78},
            "model_score_type": "fixture_softmax",
            "voice_synthesis_review": {
                "status": "model_indicator_available",
                "indicator": "fake_signal",
                "model": "fixture-audio-model",
                "windows_assessed": 2,
                "median_fake_model_score": 0.78,
                "note": "Fixture synthetic/spoof-direction indicator only.",
            },
        },
    )
    monkeypatch.setattr(
        analysis_module, "extract_image_text",
        lambda path: {"status": "completed", "text": "The city bridge reopened after inspection."},
    )
    monkeypatch.setattr(
        analysis_module, "extract_video_frame_text",
        lambda path: {"status": "completed", "frames": [
            {"timestamp_seconds": 1.2, "text": "The city bridge reopened after inspection."}
        ]},
    )
    monkeypatch.setattr(
        analysis_module, "transcribe_audio",
        lambda path: {"status": "completed", "text": "The city bridge reopened after inspection."},
    )
    monkeypatch.setattr(
        analysis_module, "inspect_provenance",
        lambda path, digest, filename: {"sha256": digest, "status": "fixture"},
    )
    monkeypatch.setattr(analysis_module, "verify_c2pa", lambda path: {"status": "not_configured"})
    monkeypatch.setattr(analysis_module, "temporal_video_forensics", lambda path: {"status": "completed"})
    monkeypatch.setattr(analysis_module, "face_identity_consistency", lambda path: {"status": "dependency_missing"})
    monkeypatch.setattr(
        analysis_module, "estimate_audio_video_activity_alignment",
        lambda video, audio: {"status": "completed", "method": "fixture alignment"},
    )
    monkeypatch.setattr(
        analysis_module, "syncnet_lipsync",
        lambda video, audio=None: {"status": "model_weights_missing", "note": "Fixture optional dependency status."},
    )
    monkeypatch.setattr(
        analysis_module, "upload_original_media",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(analysis_module, "save_report_and_activity", lambda **kwargs: {})
    monkeypatch.setattr(analysis_module, "append_case_events", lambda *args, **kwargs: [])
    
    async def fake_claims(text):
        return {
            "status": "completed",
            "claims_reviewed": 0,
            "results": [],
            "important_limit": "Fixture lookup; no truth verdict.",
        }

    monkeypatch.setattr(analysis_module, "verify_claims", fake_claims)

    files = [
        ("files", ("evidence.jpg", b"image-fixture", "image/jpeg")),
        ("files", ("recording.mp4", b"video-fixture", "video/mp4")),
        ("files", ("speech.wav", b"audio-fixture", "audio/wav")),
    ]
    file_bytes = [item[1][1] for item in files]
    digests = [hashlib.sha256(content).hexdigest() for content in file_bytes]

    app.dependency_overrides[get_current_user] = lambda: {
        "uid": "multimodal-demo-user",
        "email": "analyst@example.test",
    }
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/analyze",
                data={
                    "context_text": "The city bridge reopened after inspection.",
                    "case_title": "Multimodal workflow fixture",
                    "pair_media": "true",
                    "sha256_digests": json.dumps(digests),
                },
                files=files,
            )
        assert response.status_code == 200, response.text
        report = response.json()
        assert {asset["media_type"] for asset in report["assets"]} == {"image", "video", "audio"}
        assert all(asset["integrity_verification"]["status"] == "match" for asset in report["assets"])
        assert report["human_review_required"] is True

        comparison = report["text_analysis"]["source_comparison"]
        assert comparison["status"] == "completed"
        assert comparison["source_count"] == 4
        assert comparison["comparison_count"] == 6
        source_types = {source["source_type"] for source in comparison["sources"]}
        assert source_types == {
            "user_supplied_claim_or_caption",
            "image_ocr_caption",
            "video_frame_ocr_caption",
            "audio_transcript",
        }

        review_types = {item["type"] for item in report["cross_modal_review"]}
        assert "paired_audio_video_model_signal_comparison" in review_types
        assert "experimental_audio_video_activity_alignment" in review_types
        assert "ai_lip_sync" in review_types
        assert report["overall_status"] == "model_indicators_available_human_review_required"
    finally:
        app.dependency_overrides.pop(get_current_user, None)
