from app.services import evidence_ledger


def test_append_and_verify_hash_chain(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence_ledger, "DB_PATH", tmp_path / "ledger.sqlite3")
    events = evidence_ledger.append_case_events(
        "case-test",
        [{"asset_id": "a1", "filename": "sample.png", "sha256": "a" * 64, "size_bytes": 10}],
        "user-1",
    )
    assert len(events) == 3
    assert evidence_ledger.verify_case_chain("case-test")["status"] == "valid"
    assert len(evidence_ledger.get_case_events("case-test")) == 3
