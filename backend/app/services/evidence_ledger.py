from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.database import connect_postgres, is_postgres_enabled

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
        event_hash TEXT NOT NULL UNIQUE
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_evidence_report ON evidence_events(report_id, id)")
    return db


def append_case_events(report_id: str, assets: list[dict[str, Any]], owner_uid: str) -> list[dict[str, Any]]:
    events = [{"event": "case_created", "detail": "Investigation created.", "owner_uid": owner_uid, "asset_count": len(assets)}]
    for asset in assets:
        events.append({"event": "evidence_received", "detail": f"Evidence received: {asset.get('filename', 'unnamed')}.",
                       "owner_uid": owner_uid, "asset_id": asset.get("asset_id"), "filename": asset.get("filename"),
                       "sha256": asset.get("sha256"), "size_bytes": asset.get("size_bytes")})
    events.append({"event": "analysis_completed", "detail": "Automated analysis completed; human review required.",
                   "owner_uid": owner_uid})
    stored = []
    if is_postgres_enabled():
        from psycopg.types.json import Jsonb
        with connect_postgres() as db:
            case = db.execute(
                """INSERT INTO public.truthlens_cases
                   (firebase_uid, report_id, case_title, overall_status, summary, report_json)
                   VALUES (%s, %s, %s, 'inconclusive', '', '{}'::jsonb)
                   ON CONFLICT (firebase_uid, report_id) WHERE report_id IS NOT NULL DO UPDATE SET updated_at = now()
                   RETURNING id""",
                (owner_uid, report_id, "Investigation"),
            ).fetchone()
            case_id = case["id"]
            row = db.execute(
                """SELECT event_hash FROM public.truthlens_evidence_events
                   WHERE case_id = %s AND firebase_uid = %s ORDER BY id DESC LIMIT 1""",
                (case_id, owner_uid),
            ).fetchone()
            previous = row["event_hash"].strip() if row else ZERO_HASH
            for event in events:
                timestamp = datetime.now(timezone.utc).isoformat()
                payload = {**event, "timestamp": timestamp}
                canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                digest = hashlib.sha256((previous + canonical).encode("utf-8")).hexdigest()
                db.execute(
                    """INSERT INTO public.truthlens_evidence_events
                       (case_id, firebase_uid, event_type, payload, previous_hash, event_hash)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (case_id, owner_uid, event["event"], Jsonb(payload), previous, digest),
                )
                stored.append({**payload, "previous_event_hash": previous, "event_hash": digest})
                previous = digest
        return stored
    with _connect() as db:
        row = db.execute("SELECT event_hash FROM evidence_events WHERE report_id=? ORDER BY id DESC LIMIT 1", (report_id,)).fetchone()
        previous = row["event_hash"] if row else ZERO_HASH
        for event in events:
            timestamp = datetime.now(timezone.utc).isoformat()
            payload = {**event, "timestamp": timestamp}
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            digest = hashlib.sha256((previous + canonical).encode("utf-8")).hexdigest()
            db.execute("INSERT INTO evidence_events(report_id,event_type,event_time,payload_json,previous_hash,event_hash) VALUES(?,?,?,?,?,?)", (report_id, event["event"], timestamp, canonical, previous, digest))
            stored.append({**payload, "previous_event_hash": previous, "event_hash": digest})
            previous = digest
    return stored


def get_case_events(report_id: str, owner_uid: str | None = None) -> list[dict[str, Any]]:
    if is_postgres_enabled():
        with connect_postgres() as db:
            if owner_uid is None:
                rows = db.execute(
                    """SELECT e.payload, e.previous_hash, e.event_hash FROM public.truthlens_evidence_events e
                       JOIN public.truthlens_cases c ON c.id = e.case_id
                       WHERE c.report_id = %s ORDER BY e.id""", (report_id,),
                ).fetchall()
            else:
                rows = db.execute(
                    """SELECT e.payload, e.previous_hash, e.event_hash FROM public.truthlens_evidence_events e
                       JOIN public.truthlens_cases c ON c.id = e.case_id
                       WHERE c.report_id = %s AND c.firebase_uid = %s AND e.firebase_uid = %s ORDER BY e.id""",
                    (report_id, owner_uid, owner_uid),
                ).fetchall()
        return [{**(row["payload"] if isinstance(row["payload"], dict) else json.loads(row["payload"])),
                 "previous_event_hash": row["previous_hash"].strip(), "event_hash": row["event_hash"].strip()} for row in rows]
    with _connect() as db:
        rows = db.execute("SELECT payload_json,previous_hash,event_hash FROM evidence_events WHERE report_id=? ORDER BY id", (report_id,)).fetchall()
    return [{**json.loads(row["payload_json"]), "previous_event_hash": row["previous_hash"], "event_hash": row["event_hash"]} for row in rows]


def verify_case_chain(report_id: str, owner_uid: str | None = None) -> dict[str, Any]:
    events = get_case_events(report_id, owner_uid)
    if not events:
        return {"status": "not_found", "report_id": report_id, "events_checked": 0}
    previous = ZERO_HASH
    if is_postgres_enabled():
        with connect_postgres() as db:
            if owner_uid is None:
                rows = db.execute(
                    """SELECT e.id, e.payload, e.previous_hash, e.event_hash
                       FROM public.truthlens_evidence_events e JOIN public.truthlens_cases c ON c.id=e.case_id
                       WHERE c.report_id=%s ORDER BY e.id""", (report_id,),
                ).fetchall()
            else:
                rows = db.execute(
                    """SELECT e.id, e.payload, e.previous_hash, e.event_hash
                       FROM public.truthlens_evidence_events e JOIN public.truthlens_cases c ON c.id=e.case_id
                       WHERE c.report_id=%s AND c.firebase_uid=%s AND e.firebase_uid=%s ORDER BY e.id""",
                    (report_id, owner_uid, owner_uid),
                ).fetchall()
    else:
        with _connect() as db:
            rows = db.execute("SELECT id,payload_json,previous_hash,event_hash FROM evidence_events WHERE report_id=? ORDER BY id", (report_id,)).fetchall()
    for row in rows:
        raw_payload = row["payload"] if is_postgres_enabled() else row["payload_json"]
        payload = raw_payload if isinstance(raw_payload, dict) else json.loads(raw_payload)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        expected = hashlib.sha256((previous + canonical).encode("utf-8")).hexdigest()
        stored_previous = str(row["previous_hash"]).strip()
        stored_hash = str(row["event_hash"]).strip()
        if stored_previous != previous or stored_hash != expected:
            return {"status": "integrity_error", "report_id": report_id, "events_checked": len(rows), "failed_event_id": row["id"]}
        previous = stored_hash
    return {"status": "valid", "report_id": report_id, "events_checked": len(rows), "last_event_hash": previous, "note": "Valid local hash chain. This does not provide trusted timestamps, external anchoring, or protection from replacement of the entire database."}
