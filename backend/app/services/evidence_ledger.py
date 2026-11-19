from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(os.getenv("TRUTHLENS_LEDGER_DB", Path(__file__).resolve().parents[2] / "data" / "evidence_ledger.sqlite3"))
ZERO_HASH = "0" * 64


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""CREATE TABLE IF NOT EXISTS evidence_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        report_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        event_time TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        previous_hash TEXT NOT NULL,
        event_hash TEXT NOT NULL UNIQUE,
        signature TEXT,
        signature_key_id TEXT
    )""")
    columns = {row["name"] for row in db.execute("PRAGMA table_info(evidence_events)").fetchall()}
    if "signature" not in columns:
        db.execute("ALTER TABLE evidence_events ADD COLUMN signature TEXT")
    if "signature_key_id" not in columns:
        db.execute("ALTER TABLE evidence_events ADD COLUMN signature_key_id TEXT")
    db.execute("CREATE INDEX IF NOT EXISTS idx_evidence_report ON evidence_events(report_id, id)")
    return db



def _signature(event_hash: str) -> tuple[str | None, str | None]:
    secret = os.getenv("TRUTHLENS_LEDGER_HMAC_KEY", "").encode("utf-8")
    if not secret:
        return None, None
    key_id = os.getenv("TRUTHLENS_LEDGER_KEY_ID", "truthlens-ledger-key-1").strip()[:80]
    return hmac.new(secret, event_hash.encode("ascii"), hashlib.sha256).hexdigest(), key_id


def append_case_events(report_id: str, assets: list[dict[str, Any]], owner_uid: str) -> list[dict[str, Any]]:
    events = [{"event": "case_created", "detail": "Investigation created.", "owner_uid": owner_uid, "asset_count": len(assets)}]
    for asset in assets:
        events.append({"event": "evidence_received", "detail": f"Evidence received: {asset.get('filename', 'unnamed')}.", "asset_id": asset.get("asset_id"), "filename": asset.get("filename"), "sha256": asset.get("sha256"), "size_bytes": asset.get("size_bytes")})
    events.append({"event": "analysis_completed", "detail": "Automated analysis completed; human review required."})
    stored = []
    with _connect() as db:
        row = db.execute("SELECT event_hash FROM evidence_events WHERE report_id=? ORDER BY id DESC LIMIT 1", (report_id,)).fetchone()
        previous = row["event_hash"] if row else ZERO_HASH
        for event in events:
            timestamp = datetime.now(timezone.utc).isoformat()
            payload = {**event, "timestamp": timestamp}
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            digest = hashlib.sha256((previous + canonical).encode("utf-8")).hexdigest()
            signature, key_id = _signature(digest)
            db.execute(
                "INSERT INTO evidence_events(report_id,event_type,event_time,payload_json,previous_hash,event_hash,signature,signature_key_id) VALUES(?,?,?,?,?,?,?,?)",
                (report_id, event["event"], timestamp, canonical, previous, digest, signature, key_id),
            )
            stored.append({**payload, "previous_event_hash": previous, "event_hash": digest, "signature": signature, "signature_key_id": key_id})
            previous = digest
    return stored


def get_case_events(report_id: str) -> list[dict[str, Any]]:
    with _connect() as db:
        rows = db.execute("SELECT payload_json,previous_hash,event_hash,signature,signature_key_id FROM evidence_events WHERE report_id=? ORDER BY id", (report_id,)).fetchall()
    return [{
        **json.loads(row["payload_json"]),
        "previous_event_hash": row["previous_hash"],
        "event_hash": row["event_hash"],
        "signature": row["signature"],
        "signature_key_id": row["signature_key_id"],
    } for row in rows]


def verify_case_chain(report_id: str) -> dict[str, Any]:
    events = get_case_events(report_id)
    if not events:
        return {"status": "not_found", "report_id": report_id, "events_checked": 0}
    previous = ZERO_HASH
    with _connect() as db:
        rows = db.execute("SELECT id,payload_json,previous_hash,event_hash,signature,signature_key_id FROM evidence_events WHERE report_id=? ORDER BY id", (report_id,)).fetchall()
    for row in rows:
        payload = json.loads(row["payload_json"])
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        expected = hashlib.sha256((previous + canonical).encode("utf-8")).hexdigest()
        if row["previous_hash"] != previous or row["event_hash"] != expected:
            return {"status": "integrity_error", "report_id": report_id, "events_checked": len(rows), "failed_event_id": row["id"]}
        secret = os.getenv("TRUTHLENS_LEDGER_HMAC_KEY", "").encode("utf-8")
        if secret:
            if not row["signature"]:
                return {"status": "signature_missing", "report_id": report_id, "events_checked": len(rows), "failed_event_id": row["id"]}
            expected_signature, _ = _signature(expected)
            if not hmac.compare_digest(str(row["signature"]), str(expected_signature)):
                return {"status": "signature_invalid", "report_id": report_id, "events_checked": len(rows), "failed_event_id": row["id"]}
        previous = row["event_hash"]
    signed = bool(os.getenv("TRUTHLENS_LEDGER_HMAC_KEY", ""))
    return {
        "status": "valid",
        "report_id": report_id,
        "events_checked": len(rows),
        "last_event_hash": previous,
        "signature_status": "hmac_verified" if signed else "not_configured",
        "key_id": os.getenv("TRUTHLENS_LEDGER_KEY_ID", "truthlens-ledger-key-1") if signed else None,
        "note": "Local hash-chain verification" + (" with server-secret HMAC signatures." if signed else ". HMAC signing is not configured.") + " This is not an independently trusted timestamp or external anchor.",
    }
