from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def database_path() -> Path:
    """Resolve the activity database path; relative paths live under backend/."""
    configured = os.getenv("TRUTHLENS_DATABASE_PATH")
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
    else:
        path = Path(__file__).resolve().parents[2] / "data" / "truthlens.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(database_path(), timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 15000")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS investigation_activity (
            event_id TEXT PRIMARY KEY,
            user_uid TEXT NOT NULL,
            event_type TEXT NOT NULL,
            report_id TEXT NOT NULL,
            case_title TEXT NOT NULL,
            overall_status TEXT NOT NULL,
            evidence_count INTEGER NOT NULL,
            verified_count INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            evidence_json TEXT NOT NULL
        )
    """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_activity_user_created
        ON investigation_activity (user_uid, created_at DESC)
    """)
    return connection


def record_investigation_completed(
    *,
    user_uid: str,
    report_id: str,
    case_title: str,
    overall_status: str,
    evidence: list[dict[str, Any]],
    created_at: str | None = None,
) -> dict[str, Any]:
    """Persist one completed investigation for its authenticated Firebase user."""
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    safe_evidence = [
        {
            "filename": str(item.get("filename") or "evidence"),
            "media_type": str(item.get("media_type") or "unknown"),
            "size_bytes": int(item.get("size_bytes") or 0),
            "sha256": str(item.get("sha256") or ""),
            "integrity_verification": item.get("integrity_verification") or {"status": "not_compared"},
        }
        for item in evidence
    ]
    verified_count = sum(
        item["integrity_verification"].get("status") == "match"
        for item in safe_evidence
    )
    event = {
        "event_id": str(uuid4()),
        "user_uid": user_uid,
        "event_type": "investigation_completed",
        "report_id": report_id,
        "case_title": case_title,
        "overall_status": overall_status,
        "evidence_count": len(safe_evidence),
        "verified_count": verified_count,
        "created_at": timestamp,
        "evidence": safe_evidence,
    }
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO investigation_activity (
                event_id, user_uid, event_type, report_id, case_title,
                overall_status, evidence_count, verified_count, created_at, evidence_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["event_id"], user_uid, event["event_type"], report_id,
                case_title, overall_status, len(safe_evidence), verified_count,
                timestamp, json.dumps(safe_evidence, separators=(",", ":")),
            ),
        )
    return event


def list_user_activity(user_uid: str, limit: int = 20) -> list[dict[str, Any]]:
    """Return the newest activity events for the authenticated user only."""
    safe_limit = max(1, min(int(limit), 50))
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT event_id, event_type, report_id, case_title, overall_status,
                   evidence_count, verified_count, created_at, evidence_json
            FROM investigation_activity
            WHERE user_uid = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            (user_uid, safe_limit),
        ).fetchall()
    events = []
    for row in rows:
        events.append({
            "event_id": row["event_id"],
            "event_type": row["event_type"],
            "report_id": row["report_id"],
            "case_title": row["case_title"],
            "overall_status": row["overall_status"],
            "evidence_count": row["evidence_count"],
            "verified_count": row["verified_count"],
            "created_at": row["created_at"],
            "evidence": json.loads(row["evidence_json"]),
        })
    return events
