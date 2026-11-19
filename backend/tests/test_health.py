from fastapi.testclient import TestClient
from app.main import app
from app.services.correlation import correlate_modalities

client = TestClient(app)


def test_cors_allows_vite_fallback_ports():
    response = client.options(
        "/api/activity",
        headers={
            "Origin": "http://localhost:5174",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5174"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_capabilities_advertise_model_pipelines():
    response = client.get("/api/capabilities")
    assert response.status_code == 200
    models = response.json()["models"]
    assert "deitfake" in models["image"]
    assert "wav2vec2" in models["audio"].lower()
    assert "hemgg/deepfake-audio-detection" in response.json()["model_checkpoints"]["audio"].lower()


def test_cross_modal_requires_user_pair_confirmation():
    assets = [
        {"media_type": "video", "video_analysis": {"scores": {"fake_median_across_face_frames": 0.8}}},
        {"media_type": "audio", "model_assessment": {"scores": {"fake_median_across_windows": 0.8}}},
    ]
    unconfirmed = correlate_modalities(assets, paired_media=False)
    assert unconfirmed[0]["status"] == "not_assessed"


def test_cross_modal_disagreement_is_inconclusive_review():
    assets = [
        {"media_type": "video", "video_analysis": {"scores": {"fake_median_across_face_frames": 0.8}}},
        {"media_type": "audio", "model_assessment": {"scores": {"fake_median_across_windows": 0.2}}},
    ]
    result = correlate_modalities(assets, paired_media=True)
    assert result[0]["status"] == "modality_disagreement"


def test_me_requires_firebase_bearer_token():
    response = client.get("/api/me")
    assert response.status_code == 401


def test_activity_requires_firebase_bearer_token():
    response = client.get("/api/activity")
    assert response.status_code == 401


def test_saved_report_requires_firebase_bearer_token():
    response = client.get("/api/reports/report-123")
    assert response.status_code == 401


def test_saved_heatmap_requires_firebase_bearer_token():
    response = client.get(
        "/api/reports/report-123/heatmap",
        params={"asset_id": "asset-123", "face_index": 0},
    )
    assert response.status_code == 401



def test_analyst_review_endpoint_requires_authentication():
    response = client.post(
        "/api/reports/report-123/review",
        json={"decision": "inconclusive", "notes": "Insufficient evidence.", "reviewed_asset_ids": []},
    )
    assert response.status_code == 401
