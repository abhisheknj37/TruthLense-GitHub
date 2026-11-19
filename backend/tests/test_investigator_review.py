import app.services.investigator_review as reviews


def test_reviews_are_user_scoped_and_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(reviews, "DB_PATH", tmp_path / "reviews.sqlite3")
    saved = reviews.save_finding_review("user-a", "case-1", "finding-1", "confirmed", "Reviewed source frame.")
    assert saved["status"] == "confirmed"
    assert reviews.list_finding_reviews("user-a", "case-1")[0]["comment"] == "Reviewed source frame."
    assert reviews.list_finding_reviews("user-a", "case-1")[0]["reviewer_uid"] == "user-a"
    assert reviews.list_finding_reviews("user-b", "case-1") == []


def test_review_rejects_invalid_status_and_oversized_comment(tmp_path, monkeypatch):
    monkeypatch.setattr(reviews, "DB_PATH", tmp_path / "reviews.sqlite3")
    try:
        reviews.save_finding_review("u", "c", "f", "true", "")
        assert False, "invalid status should fail"
    except ValueError:
        pass
    try:
        reviews.save_finding_review("u", "c", "f", "confirmed", "x" * 4001)
        assert False, "oversized comment should fail"
    except ValueError:
        pass



def test_finding_review_api_is_authenticated_owner_scoped_and_validates_finding(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.auth import get_current_user
    from app.main import app
    from app.routes import analysis as analysis_module
    from app.services import evidence_ledger

    monkeypatch.setattr(reviews, "DB_PATH", tmp_path / "api-reviews.sqlite3")
    monkeypatch.setattr(evidence_ledger, "DB_PATH", tmp_path / "api-ledger.sqlite3")
    evidence_ledger.append_case_events("case-api", [], "owner")
    report = {
        "report_id": "case-api",
        "assets": [{
            "asset_id": "asset-1",
            "observations": [{"code": "MODEL_SIGNAL_AVAILABLE"}],
        }],
    }
    monkeypatch.setattr(
        analysis_module,
        "get_user_report",
        lambda uid, report_id: report if uid == "owner" and report_id == "case-api" else None,
    )
    app.dependency_overrides[get_current_user] = lambda: {"uid": "owner", "email": "owner@example.test"}
    try:
        with TestClient(app) as client:
            saved = client.put(
                "/api/reports/case-api/reviews/asset-1:MODEL_SIGNAL_AVAILABLE",
                json={"status": "confirmed", "comment": "Reviewed the original frame."},
            )
            assert saved.status_code == 200
            assert saved.json()["status"] == "confirmed"
            assert saved.json()["reviewer_uid"] == "owner"

            listed = client.get("/api/reports/case-api/reviews")
            assert listed.status_code == 200
            assert listed.json()["reviews"][0]["finding_id"] == "asset-1:MODEL_SIGNAL_AVAILABLE"

            invalid = client.put(
                "/api/reports/case-api/reviews/not-a-finding",
                json={"status": "confirmed", "comment": "Not part of this case."},
            )
            assert invalid.status_code == 404

            app.dependency_overrides[get_current_user] = lambda: {"uid": "other"}
            denied = client.get("/api/reports/case-api/reviews")
            assert denied.status_code == 404
    finally:
        app.dependency_overrides.pop(get_current_user, None)
