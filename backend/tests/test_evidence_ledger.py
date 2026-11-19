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


def test_hmac_signature_is_verified_and_detects_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence_ledger, "DB_PATH", tmp_path / "signed-ledger.sqlite3")
    monkeypatch.setenv("TRUTHLENS_LEDGER_HMAC_KEY", "test-secret-not-for-production")
    monkeypatch.setenv("TRUTHLENS_LEDGER_KEY_ID", "test-key-v1")
    events = evidence_ledger.append_case_events(
        "case-signed",
        [{"asset_id": "asset-00000001", "filename": "sample.png", "sha256": "b" * 64, "size_bytes": 25}],
        "user-1",
    )
    assert events[0]["signature"]
    assert events[0]["signature_key_id"] == "test-key-v1"
    result = evidence_ledger.verify_case_chain("case-signed")
    assert result["status"] == "valid"
    assert result["signature_status"] == "hmac_verified"

    with evidence_ledger._connect() as db:
        row = db.execute(
            "SELECT id,payload_json FROM evidence_events WHERE report_id=? ORDER BY id LIMIT 1",
            ("case-signed",),
        ).fetchone()
        payload = row["payload_json"].replace("Investigation created.", "Modified after signing.")
        db.execute("UPDATE evidence_events SET payload_json=? WHERE id=?", (payload, row["id"]))
    assert evidence_ledger.verify_case_chain("case-signed")["status"] == "integrity_error"


def test_unsigned_ledger_is_explicit_about_missing_hmac(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence_ledger, "DB_PATH", tmp_path / "unsigned-ledger.sqlite3")
    monkeypatch.delenv("TRUTHLENS_LEDGER_HMAC_KEY", raising=False)
    evidence_ledger.append_case_events("case-unsigned", [], "user-1")
    result = evidence_ledger.verify_case_chain("case-unsigned")
    assert result["status"] == "valid"
    assert result["signature_status"] == "not_configured"


def test_delete_case_events_removes_only_requested_case(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence_ledger, "DB_PATH", tmp_path / "delete-ledger.sqlite3")
    evidence_ledger.append_case_events("case-one", [], "user-1")
    evidence_ledger.append_case_events("case-two", [], "user-1")
    removed = evidence_ledger.delete_case_events("case-one")
    assert removed == 2
    assert evidence_ledger.verify_case_chain("case-one")["status"] == "not_found"
    assert evidence_ledger.verify_case_chain("case-two")["status"] == "valid"
