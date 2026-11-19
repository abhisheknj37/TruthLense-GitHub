import base64

from app.services.activity import (
    delete_user_report,
    get_user_heatmap_path,
    get_user_report,
    list_user_activity,
    save_report_and_activity,
    save_user_report_review,
)
from app.services.integrity import compare_sha256


def test_sha256_compare_match_mismatch_and_missing():
    digest = "a" * 64
    assert compare_sha256(digest, digest)["status"] == "match"
    assert compare_sha256("b" * 64, digest)["status"] == "mismatch"
    assert compare_sha256(None, digest)["status"] == "not_compared"
    assert compare_sha256("invalid", digest)["status"] == "invalid_client_digest"
    assert compare_sha256(123, digest)["status"] == "invalid_client_digest"


def test_report_activity_and_heatmap_are_persisted_per_user(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUTHLENS_DATABASE_PATH", str(tmp_path / "truthlens.sqlite3"))
    monkeypatch.setenv("TRUTHLENS_DATA_DIR", str(tmp_path / "data"))
    overlay = base64.b64encode(b"\x89PNG\r\n\x1a\nheatmap-test").decode("ascii")
    report = {
        "report_id": "case-001",
        "created_at": "2026-11-19T10:00:00+00:00",
        "overall_status": "inconclusive",
        "summary": "Test report",
        "assets": [{
            "asset_id": "asset-001",
            "filename": "face.png",
            "media_type": "image",
            "size_bytes": 100,
            "sha256": "a" * 64,
            "analysis_status": "model_assessed",
            "model_assessment": {
                "prediction": "inconclusive",
                "faces": [{
                    "face_index": 0,
                    "gradcam_overlay_base64": overlay,
                }],
            },
            "integrity_verification": {"status": "match"},
        }],
        "limitations": [],
    }

    event = save_report_and_activity(
        user_uid="firebase-user-a",
        report=report,
        case_title="Example investigation",
    )

    assert event["evidence_count"] == 1
    assert event["verified_count"] == 1
    assert list_user_activity("firebase-user-a")[0]["report_id"] == "case-001"
    assert list_user_activity("firebase-user-b") == []

    saved = get_user_report("firebase-user-a", "case-001")
    assert saved["case_title"] == "Example investigation"
    assert saved["assets"][0]["model_assessment"]["faces"][0]["gradcam_overlay_base64"] == overlay
    assert get_user_heatmap_path("firebase-user-a", "case-001", "asset-001", 0).is_file()
    assert get_user_report("firebase-user-b", "case-001") is None
    assert get_user_heatmap_path("firebase-user-b", "case-001", "asset-001", 0) is None

    assert delete_user_report("firebase-user-a", "case-001") is True
    assert get_user_report("firebase-user-a", "case-001") is None
    assert list_user_activity("firebase-user-a") == []



def test_analyst_review_is_persisted_separately_and_scoped_to_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUTHLENS_DATABASE_PATH", str(tmp_path / "truthlens.sqlite3"))
    monkeypatch.setenv("TRUTHLENS_DATA_DIR", str(tmp_path / "data"))
    report = {
        "report_id": "case-review-001",
        "created_at": "2026-11-20T00:00:00+00:00",
        "overall_status": "model_indicators_available_human_review_required",
        "summary": "Separate model indicators available.",
        "assets": [{
            "asset_id": "asset-a",
            "filename": "clip.mp4",
            "media_type": "video",
            "size_bytes": 123,
            "sha256": "c" * 64,
            "analysis_status": "model_assessed",
            "model_assessment": {},
            "integrity_verification": {"status": "match"},
        }],
        "limitations": ["Human review required."],
        "human_review_required": True,
    }
    save_report_and_activity(user_uid="reviewer-a", report=report, case_title="Review test")

    review = {
        "status": "completed",
        "decision": "manipulation_suspected",
        "notes": "Visible mismatch requires source verification.",
        "reviewed_asset_ids": ["asset-a"],
        "reviewed_by_uid": "reviewer-a",
        "reviewer": "Analyst",
        "reviewed_at": "2026-11-20T00:01:00+00:00",
    }
    assert save_user_report_review("reviewer-a", "case-review-001", review) is True
    assert save_user_report_review("reviewer-b", "case-review-001", review) is False
    saved = get_user_report("reviewer-a", "case-review-001")
    assert saved["analyst_review"]["decision"] == "manipulation_suspected"
    assert saved["analyst_review"]["reviewed_asset_ids"] == ["asset-a"]
    assert get_user_report("reviewer-b", "case-review-001") is None



def test_analyst_review_http_endpoint_persists_and_enforces_case_owner(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.auth import get_current_user
    from app.main import app

    monkeypatch.setenv("TRUTHLENS_DATABASE_PATH", str(tmp_path / "review-api.sqlite3"))
    monkeypatch.setenv("TRUTHLENS_DATA_DIR", str(tmp_path / "review-api-data"))
    report = {
        "report_id": "case-review-api",
        "created_at": "2026-11-20T00:00:00+00:00",
        "overall_status": "inconclusive",
        "summary": "Human review endpoint test.",
        "assets": [{
            "asset_id": "asset-api",
            "filename": "sample.wav",
            "media_type": "audio",
            "size_bytes": 12,
            "sha256": "d" * 64,
            "analysis_status": "inconclusive",
            "model_assessment": {},
            "integrity_verification": {"status": "match"},
        }],
        "limitations": [],
    }
    save_report_and_activity(user_uid="api-owner", report=report, case_title="Review API test")

    app.dependency_overrides[get_current_user] = lambda: {
        "uid": "api-owner", "email": "owner@example.test"
    }
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/reports/case-review-api/review",
                json={
                    "decision": "inconclusive",
                    "notes": "The available evidence is insufficient.",
                    "reviewed_asset_ids": ["asset-api"],
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["analyst_review"]["decision"] == "inconclusive"

            app.dependency_overrides[get_current_user] = lambda: {
                "uid": "different-user", "email": "other@example.test"
            }
            denied = client.post(
                "/api/reports/case-review-api/review",
                json={
                    "decision": "manipulation_suspected",
                    "notes": "Attempt to update another user's case.",
                    "reviewed_asset_ids": ["asset-api"],
                },
            )
            assert denied.status_code == 404
    finally:
        app.dependency_overrides.pop(get_current_user, None)
