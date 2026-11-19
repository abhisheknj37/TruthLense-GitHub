from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(os.getenv("TRUTHLENS_REVIEW_DB", Path(__file__).resolve().parents[2] / "data" / "investigator_reviews.sqlite3"))


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS finding_reviews (
        user_uid TEXT NOT NULL,
        report_id TEXT NOT NULL,
        finding_id TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('unreviewed','confirmed','rejected','inconclusive')),
        comment TEXT NOT NULL DEFAULT '',
        updated_at TEXT NOT NULL,
        PRIMARY KEY(user_uid, report_id, finding_id)
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_reviews_report ON finding_reviews(user_uid, report_id)")
    return db


def save_finding_review(
    user_uid: str, report_id: str, finding_id: str, status: str, comment: str
) -> dict[str, Any]:
    if status not in {"unreviewed", "confirmed", "rejected", "inconclusive"}:
        raise ValueError("Unsupported review status.")
    if not finding_id or len(finding_id) > 160:
        raise ValueError("finding_id must contain 1 to 160 characters.")
    if len(comment) > 4000:
        raise ValueError("Review comment cannot exceed 4,000 characters.")
    timestamp = datetime.now(timezone.utc).isoformat()
    with _connect() as db:
        db.execute(
            """INSERT INTO finding_reviews(user_uid,report_id,finding_id,status,comment,updated_at)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(user_uid,report_id,finding_id) DO UPDATE SET
               status=excluded.status, comment=excluded.comment, updated_at=excluded.updated_at""",
            (user_uid, report_id, finding_id, status, comment.strip(), timestamp),
        )
    return {
        "report_id": report_id, "finding_id": finding_id, "status": status,
        "comment": comment.strip(), "updated_at": timestamp,
    }


def list_finding_reviews(user_uid: str, report_id: str) -> list[dict[str, Any]]:
    with _connect() as db:
        rows = db.execute(
            """SELECT finding_id,status,comment,updated_at FROM finding_reviews
               WHERE user_uid=? AND report_id=? ORDER BY updated_at""",
            (user_uid, report_id),
        ).fetchall()
    return [dict(row) for row in rows]
