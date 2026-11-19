import base64

from app.services.activity import (
    delete_user_report,
    get_user_heatmap_path,
    get_user_report,
    list_user_activity,
    save_report_and_activity,
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
