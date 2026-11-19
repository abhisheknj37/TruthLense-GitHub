from pathlib import Path

from app.services import evidence_storage


def test_original_retention_is_opt_in_and_owner_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUTHLENS_ORIGINALS_DIR", str(tmp_path / "private"))
    monkeypatch.setenv("TRUTHLENS_ORIGINAL_RETENTION_DAYS", "7")
    source = tmp_path / "source.txt"
    source.write_text("original evidence bytes", encoding="utf-8")
    digest = evidence_storage.sha256_file(source)
    report_id = "12345678-1234-1234-1234-123456789abc"
    asset_id = "abcdef12-1234-1234-1234-123456789abc"

    not_retained = evidence_storage.store_original(
        source, uid="owner-a", report_id=report_id, asset_id=asset_id,
        filename="claim.txt", digest=digest, retain=False,
    )
    assert not_retained["status"] == "not_retained"

    saved = evidence_storage.store_original(
        source, uid="owner-a", report_id=report_id, asset_id=asset_id,
        filename="claim.txt", digest=digest, retain=True,
    )
    assert saved["status"] == "retained"
    assert saved["retention_days"] == 7
    asset = {"asset_id": asset_id, "sha256": digest, "evidence_storage": saved}
    path, state = evidence_storage.resolve_original("owner-a", report_id, asset)
    assert state == "available"
    assert path.read_text(encoding="utf-8") == "original evidence bytes"

    denied, denied_state = evidence_storage.resolve_original("owner-b", report_id, asset)
    assert denied is None
    assert denied_state == "invalid_reference"


def test_original_retention_detects_stored_file_tampering(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUTHLENS_ORIGINALS_DIR", str(tmp_path / "private"))
    source = tmp_path / "source.bin"
    source.write_bytes(b"unmodified evidence")
    digest = evidence_storage.sha256_file(source)
    report_id = "12345678-1234-1234-1234-123456789abc"
    asset_id = "abcdef12-1234-1234-1234-123456789abc"
    saved = evidence_storage.store_original(
        source, uid="owner", report_id=report_id, asset_id=asset_id,
        filename="sample.wav", digest=digest, retain=True,
    )
    asset = {"asset_id": asset_id, "sha256": digest, "evidence_storage": saved}
    stored = evidence_storage._root() / saved["storage_key"]
    stored.write_bytes(b"tampered evidence")
    path, state = evidence_storage.resolve_original("owner", report_id, asset)
    assert path is None
    assert state == "digest_mismatch"


def test_delete_case_originals_removes_owner_case_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUTHLENS_ORIGINALS_DIR", str(tmp_path / "private"))
    source = tmp_path / "source.bin"
    source.write_bytes(b"original")
    digest = evidence_storage.sha256_file(source)
    report_id = "12345678-1234-1234-1234-123456789abc"
    asset_id = "abcdef12-1234-1234-1234-123456789abc"
    saved = evidence_storage.store_original(
        source, uid="owner", report_id=report_id, asset_id=asset_id,
        filename="sample.wav", digest=digest, retain=True,
    )
    evidence_storage.delete_case_originals("owner", report_id)
    asset = {"asset_id": asset_id, "sha256": digest, "evidence_storage": saved}
    path, state = evidence_storage.resolve_original("owner", report_id, asset)
    assert path is None
    assert state == "missing"


def test_expired_originals_are_purged_for_owner_only(tmp_path, monkeypatch):
    import json
    from datetime import datetime, timedelta, timezone

    monkeypatch.setenv("TRUTHLENS_ORIGINALS_DIR", str(tmp_path / "private"))
    source = tmp_path / "source.bin"
    source.write_bytes(b"expire this")
    digest = evidence_storage.sha256_file(source)
    report_id = "12345678-1234-1234-1234-123456789abc"
    asset_id = "abcdef12-1234-1234-1234-123456789abc"
    saved = evidence_storage.store_original(
        source, uid="owner-a", report_id=report_id, asset_id=asset_id,
        filename="sample.bin", digest=digest, retain=True,
    )
    root = evidence_storage._root()
    stored = root / saved["storage_key"]
    sidecar = stored.with_name(stored.name + ".meta.json")
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    manifest["expires_at"] = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    sidecar.write_text(json.dumps(manifest), encoding="utf-8")

    result = evidence_storage.purge_expired_originals("owner-a")
    assert result["manifests_checked"] == 1
    assert result["expired_originals_removed"] == 1
    assert not stored.exists()
    assert not sidecar.exists()
    assert evidence_storage.purge_expired_originals("owner-b")["expired_originals_removed"] == 0
