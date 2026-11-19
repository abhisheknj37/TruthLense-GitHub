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


def test_ledger_delete_is_scoped_to_case_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence_ledger, "DB_PATH", tmp_path / "ledger.sqlite3")
    evidence_ledger.append_case_events("case-owned", [], "user-1")
    assert evidence_ledger.delete_case_events("case-owned", "user-2") is False
    assert evidence_ledger.verify_case_chain("case-owned")["status"] == "valid"
    assert evidence_ledger.delete_case_events("case-owned", "user-1") is True
    assert evidence_ledger.verify_case_chain("case-owned")["status"] == "not_found"
