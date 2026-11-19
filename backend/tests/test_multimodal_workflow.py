import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.auth import get_current_user
from app.main import app
from app.routes import analysis as analysis_module


def test_full_case_upload_compares_image_video_audio_and_claim_sources(monkeypatch):
    """Exercise the real upload/report route with deterministic model and extractor fixtures."""
    media_types = {".jpg": "image", ".mp4": "video", ".wav": "audio"}
    monkeypatch.setattr(
        analysis_module,
        "inspect_media",
        lambda path: {
            "media_type": media_types[Path(path).suffix.lower()],
            "metadata": {"fixture": True},
            "observations": [],
            "sampled_frames": [{"frame_index": 0, "timestamp_seconds": 0.0}]
            if Path(path).suffix.lower() == ".mp4" else [],
            "forensic_parameters": {},
        },
    )
    monkeypatch.setattr(
        analysis_module, "inspect_provenance",
        lambda path, digest, filename: {"sha256": digest, "status": "fixture"},
    )
    monkeypatch.setattr(analysis_module, "verify_c2pa", lambda path: {"status": "not_configured"})
    monkeypatch.setattr(
        analysis_module, "extract_image_text",
        lambda path: {
            "status": "completed",
            "text": "The city bridge reopened after inspection.",
            "analysis_text": "The city bridge reopened after inspection.",
            "analysis_eligible": True,
            "engine": "fixture OCR",
        },
    )
    monkeypatch.setattr(analysis_module, "compare_ocr_engines", lambda path: {"status": "completed"})
    monkeypatch.setattr(
        analysis_module, "extract_video_frame_text",
        lambda path: {
            "status": "completed",
            "frames": [{
                "frame_index": 0,
                "timestamp_seconds": 1.2,
                "text": "The city bridge reopened after inspection.",
                "analysis_text": "The city bridge reopened after inspection.",
                "analysis_eligible": True,
            }],
        },
    )
    monkeypatch.setattr(
        analysis_module, "transcribe_audio",
        lambda path: {"status": "completed", "text": "The city bridge reopened after inspection."},
    )
    monkeypatch.setattr(
        analysis_module.image_detector, "analyze_image",
        lambda *args: {
            "status": "model_assessed",
            "model_name": "fixture-image",
            "prediction": "real_signal",
            "scores": {"fake_score": 0.1, "real_score": 0.9},
            "model_score_type": "fixture_softmax",
        },
    )
    monkeypatch.setattr(
        analysis_module.video_detector, "analyze",
        lambda path: {
            "status": "model_assessed",
            "model_name": "fixture-video",
            "prediction": "fake_signal",
            "scores": {"fake_median_across_face_frames": 0.82},
            "uncertainty_band": "model_score_separated",
            "model_score_type": "fixture_softmax",
            "note": "Fixture video model output.",
        },
    )
    monkeypatch.setattr(
        analysis_module.audio_detector, "analyze",
        lambda path: {
            "status": "model_assessed",
            "model_name": "fixture-audio",
            "prediction": "fake_signal",
            "scores": {"fake_median_across_windows": 0.78},
            "model_score_type": "fixture_softmax",
            "segments": [{"start_seconds": 0, "end_seconds": 2, "fake_model_score": 0.78}],
            "voice_synthesis_review": {
                "status": "model_indicator_available",
                "indicator": "fake_signal",
                "model": "fixture-audio",
                "windows_assessed": 1,
                "median_fake_model_score": 0.78,
                "note": "Fixture synthetic/spoof-direction indicator only.",
            },
        },
    )
    monkeypatch.setattr(analysis_module, "audio_review_timeline", lambda path: {"status": "completed"})
    monkeypatch.setattr(analysis_module, "temporal_video_forensics", lambda path: {"status": "completed"})
    monkeypatch.setattr(analysis_module, "face_identity_consistency", lambda path: {"status": "dependency_missing"})
    monkeypatch.setattr(
        analysis_module, "estimate_audio_video_activity_alignment",
        lambda video, audio: {"status": "completed", "method": "fixture activity alignment"},
    )
    monkeypatch.setattr(
        analysis_module, "syncnet_lipsync",
        lambda video, audio=None: {"status": "model_weights_missing"},
    )
    monkeypatch.setattr(analysis_module, "append_case_events", lambda *args, **kwargs: [])
    monkeypatch.setattr(analysis_module, "save_report_and_activity", lambda **kwargs: None)

    async def fake_claim_verification(text):
        return {"status": "completed", "claims_reviewed": 0, "results": []}

    monkeypatch.setattr(analysis_module, "verify_claims", fake_claim_verification)

    upload_bytes = [b"fixture-image", b"fixture-video", b"fixture-audio"]
    files = [
        ("files", ("evidence.jpg", upload_bytes[0], "image/jpeg")),
        ("files", ("clip.mp4", upload_bytes[1], "video/mp4")),
        ("files", ("speech.wav", upload_bytes[2], "audio/wav")),
    ]
    digests = [hashlib.sha256(value).hexdigest() for value in upload_bytes]
    app.dependency_overrides[get_current_user] = lambda: {
        "uid": "multimodal-user", "email": "analyst@example.test"
    }
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/analyze",
                data={
                    "context_text": "The city bridge reopened after inspection.",
                    "case_title": "Multimodal source comparison fixture",
                    "pair_media": "true",
                    "sha256_digests": json.dumps(digests),
                },
                files=files,
            )
        assert response.status_code == 200, response.text
        report = response.json()
        assert {item["media_type"] for item in report["assets"]} == {"image", "video", "audio"}
        assert all(item["integrity_verification"]["status"] == "match" for item in report["assets"])
        assert report["human_review_required"] is True

        text_pairs = [
            item for item in report["cross_modal_review"]
            if item["type"] == "text_evidence_overlap"
        ]
        assert len(text_pairs) == 6
        source_kinds = {item["source_a_kind"] for item in text_pairs} | {
            item["source_b_kind"] for item in text_pairs
        }
        assert source_kinds == {"claim", "image_ocr", "caption_or_video_ocr", "transcript"}
        assert all("does not establish" in item["detail"] for item in text_pairs)
        assert any(
            item["type"] == "paired_audio_video_model_signal_comparison"
            for item in report["cross_modal_review"]
        )
        assert any(
            item["type"] == "experimental_audio_video_activity_alignment"
            for item in report["cross_modal_review"]
        )
        audio_asset = next(item for item in report["assets"] if item["media_type"] == "audio")
        assert audio_asset["model_assessment"]["voice_synthesis_review"]["indicator"] == "fake_signal"
        assert report["overall_status"] == "model_indicators_available_human_review_required"
    finally:
        app.dependency_overrides.pop(get_current_user, None)
