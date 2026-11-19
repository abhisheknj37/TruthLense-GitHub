import app.services.investigator_review as reviews


def test_reviews_are_user_scoped_and_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(reviews, "DB_PATH", tmp_path / "reviews.sqlite3")
    saved = reviews.save_finding_review("user-a", "case-1", "finding-1", "confirmed", "Reviewed source frame.")
    assert saved["status"] == "confirmed"
    assert reviews.list_finding_reviews("user-a", "case-1")[0]["comment"] == "Reviewed source frame."
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
