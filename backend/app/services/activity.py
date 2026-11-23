from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def data_root() -> Path:
    configured = os.getenv("TRUTHLENS_DATA_DIR")
    root = Path(configured) if configured else Path(__file__).resolve().parents[2] / "data"
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    root.mkdir(parents=True, exist_ok=True)
    return root


def database_path() -> Path:
    configured = os.getenv("TRUTHLENS_DATABASE_PATH")
    path = Path(configured) if configured else data_root() / "truthlens.sqlite3"
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(database_path(), timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 15000")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS case_reports (
            report_id TEXT NOT NULL,
            user_uid TEXT NOT NULL,
            case_title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            report_json TEXT NOT NULL,
            PRIMARY KEY (user_uid, report_id)
        )
    """)
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
            evidence_json TEXT NOT NULL,
            UNIQUE (user_uid, report_id)
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS report_heatmaps (
            user_uid TEXT NOT NULL,
            report_id TEXT NOT NULL,
            asset_id TEXT NOT NULL,
            face_index INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            PRIMARY KEY (user_uid, report_id, asset_id, face_index)
        )
    """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_activity_user_created
        ON investigation_activity (user_uid, created_at DESC)
    """)
    return connection


@contextmanager
def _connection():
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def save_report_and_activity(
    *,
    user_uid: str,
    report: dict[str, Any],
    case_title: str,
) -> dict[str, Any]:
    """Persist report metadata, activity and real model overlays for one Firebase UID."""
    report = json.loads(json.dumps(report))
    report_id = str(report["report_id"])
    timestamp = str(report.get("created_at") or datetime.now(timezone.utc).isoformat())
    evidence_summary: list[dict[str, Any]] = []
    overlay_files: list[tuple[str, int, str]] = []
    user_folder = hashlib.sha256(user_uid.encode("utf-8")).hexdigest()
    report_folder = data_root() / "heatmaps" / user_folder / report_id
    for asset in report.get("assets", []):
        face_summary = []
        for face in (asset.get("model_assessment") or {}).get("faces", []):
            image_b64 = face.get("gradcam_overlay_base64")
            available = False
            if image_b64:
                try:
                    image_bytes = base64.b64decode(image_b64, validate=True)
                    if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n") or len(image_bytes) > 8 * 1024 * 1024:
                        raise ValueError("Heatmap must be a PNG smaller than 8 MB.")
                    report_folder.mkdir(parents=True, exist_ok=True)
                    face_index = int(face.get("face_index", len(face_summary)))
                    destination = report_folder / f"{uuid4().hex}.png"
                    destination.write_bytes(image_bytes)
                    overlay_files.append((str(asset["asset_id"]), face_index, str(destination.resolve())))
                    available = True
                except (ValueError, OSError):
                    available = False
            face["heatmap_available"] = available
            face["gradcam_overlay_base64"] = None
            face_summary.append({"face_index": int(face.get("face_index", len(face_summary))), "heatmap_available": available})
        verification = asset.get("integrity_verification") or {"status": "not_compared"}
        evidence_summary.append({
            "asset_id": str(asset.get("asset_id") or ""),
            "filename": str(asset.get("filename") or "evidence"),
            "media_type": str(asset.get("media_type") or "unknown"),
            "size_bytes": int(asset.get("size_bytes") or 0),
            "sha256": str(asset.get("sha256") or ""),
            "analysis_status": str(asset.get("analysis_status") or "inconclusive"),
            "prediction": str(
                (asset.get("video_analysis") or {}).get("prediction")
                or (asset.get("model_assessment") or {}).get("prediction")
                or "inconclusive"
            ),
            "integrity_verification": verification,
            "heatmap_faces": face_summary,
        })
    report["case_title"] = case_title
    verified_count = sum(item["integrity_verification"].get("status") == "match" for item in evidence_summary)
    event = {
        "event_id": str(uuid4()),
        "event_type": "investigation_completed",
        "report_id": report_id,
        "case_title": case_title,
        "overall_status": str(report.get("overall_status") or "unknown"),
        "evidence_count": len(evidence_summary),
        "verified_count": verified_count,
        "created_at": timestamp,
        "evidence": evidence_summary,
    }
    with _connection() as connection:
        connection.execute(
            """INSERT OR REPLACE INTO case_reports
               (report_id, user_uid, case_title, created_at, report_json)
               VALUES (?, ?, ?, ?, ?)""",
            (report_id, user_uid, case_title, timestamp, json.dumps(report, separators=(",", ":"))),
        )
        connection.execute(
            """INSERT OR REPLACE INTO investigation_activity
               (event_id, user_uid, event_type, report_id, case_title, overall_status,
                evidence_count, verified_count, created_at, evidence_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event["event_id"], user_uid, event["event_type"], report_id, case_title,
             event["overall_status"], event["evidence_count"], verified_count,
             timestamp, json.dumps(evidence_summary, separators=(",", ":"))),
        )
        connection.execute(
            "DELETE FROM report_heatmaps WHERE user_uid = ? AND report_id = ?",
            (user_uid, report_id),
        )
        connection.executemany(
            """INSERT INTO report_heatmaps
               (user_uid, report_id, asset_id, face_index, file_path)
               VALUES (?, ?, ?, ?, ?)""",
            [(user_uid, report_id, asset_id, face_index, file_path)
             for asset_id, face_index, file_path in overlay_files],
        )
    return event


def list_user_activity(user_uid: str, limit: int = 20) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 50))
    with _connect() as connection:
        rows = connection.execute(
            """SELECT event_id, event_type, report_id, case_title, overall_status,
                      evidence_count, verified_count, created_at, evidence_json
               FROM investigation_activity
               WHERE user_uid = ?
               ORDER BY created_at DESC, rowid DESC
               LIMIT ?""",
            (user_uid, safe_limit),
        ).fetchall()
    return [{
        "event_id": row["event_id"],
        "event_type": row["event_type"],
        "report_id": row["report_id"],
        "case_title": row["case_title"],
        "overall_status": row["overall_status"],
        "evidence_count": row["evidence_count"],
        "verified_count": row["verified_count"],
        "created_at": row["created_at"],
        "evidence": json.loads(row["evidence_json"]),
    } for row in rows]


def get_user_activity_summary(user_uid: str) -> dict[str, int]:
    """Aggregate all saved investigations for the account, not just the visible feed page."""
    with _connection() as connection:
        rows = connection.execute(
            """SELECT evidence_count, verified_count, evidence_json
               FROM investigation_activity WHERE user_uid = ?""",
            (user_uid,),
        ).fetchall()
    total_evidence = 0
    total_verified = 0
    manipulation_signals = 0
    for row in rows:
        total_evidence += int(row["evidence_count"])
        total_verified += int(row["verified_count"])
        for evidence in json.loads(row["evidence_json"]):
            if evidence.get("prediction") == "fake_signal":
                manipulation_signals += 1
    return {
        "case_count": len(rows),
        "evidence_count": total_evidence,
        "verified_count": total_verified,
        "manipulation_signals": manipulation_signals,
    }


def get_user_report(user_uid: str, report_id: str) -> dict[str, Any] | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT report_json FROM case_reports WHERE user_uid = ? AND report_id = ?",
            (user_uid, report_id),
        ).fetchone()
        if row is None:
            return None
        report = json.loads(row["report_json"])
        heatmaps = connection.execute(
            """SELECT asset_id, face_index, file_path FROM report_heatmaps
               WHERE user_uid = ? AND report_id = ?""",
            (user_uid, report_id),
        ).fetchall()
    by_asset: dict[str, dict[int, str]] = {}
    for item in heatmaps:
        by_asset.setdefault(item["asset_id"], {})[int(item["face_index"])] = item["file_path"]
    for asset in report.get("assets", []):
        faces = (asset.get("model_assessment") or {}).get("faces", [])
        for face in faces:
            file_path = by_asset.get(str(asset.get("asset_id")), {}).get(int(face.get("face_index", -1)))
            if file_path and Path(file_path).is_file():
                face["gradcam_overlay_base64"] = base64.b64encode(Path(file_path).read_bytes()).decode("ascii")
                face["heatmap_available"] = True
    return report


def get_user_heatmap_path(user_uid: str, report_id: str, asset_id: str, face_index: int) -> Path | None:
    with _connect() as connection:
        row = connection.execute(
            """SELECT file_path FROM report_heatmaps
               WHERE user_uid = ? AND report_id = ? AND asset_id = ? AND face_index = ?""",
            (user_uid, report_id, asset_id, face_index),
        ).fetchone()
    path = Path(row["file_path"]) if row else None
    return path if path and path.is_file() else None


def delete_user_report(user_uid: str, report_id: str) -> bool:
    """Delete one report, its activity record, and heatmap artifacts for its owner."""
    with _connect() as connection:
        rows = connection.execute(
            """SELECT file_path FROM report_heatmaps
               WHERE user_uid = ? AND report_id = ?""",
            (user_uid, report_id),
        ).fetchall()
        existing = connection.execute(
            "SELECT 1 FROM case_reports WHERE user_uid = ? AND report_id = ?",
            (user_uid, report_id),
        ).fetchone()
        if existing is None:
            return False
        connection.execute("DELETE FROM report_heatmaps WHERE user_uid = ? AND report_id = ?", (user_uid, report_id))
        connection.execute("DELETE FROM investigation_activity WHERE user_uid = ? AND report_id = ?", (user_uid, report_id))
        connection.execute("DELETE FROM case_reports WHERE user_uid = ? AND report_id = ?", (user_uid, report_id))
    for row in rows:
        try:
            Path(row["file_path"]).unlink(missing_ok=True)
        except OSError:
            pass
    return True

def save_ocr_correction(
    *, user_uid: str, report_id: str, asset_id: str, corrected_text: str
) -> dict[str, Any]:
    """Persist investigator-corrected OCR separately from raw extraction."""
    text = corrected_text.strip()
    if len(text) > 12000:
        raise ValueError("Corrected text must be 12,000 characters or fewer.")
    with _connection() as connection:
        row = connection.execute(
            "SELECT report_json FROM case_reports WHERE user_uid = ? AND report_id = ?",
            (user_uid, report_id),
        ).fetchone()
        if row is None:
            raise LookupError("Investigation not found.")
        report = json.loads(row["report_json"])
        asset = next((item for item in report.get("assets", []) if item.get("asset_id") == asset_id), None)
        if asset is None:
            raise LookupError("Evidence item not found.")
        extraction = asset.setdefault("text_extraction", {})
        correction = {
            "text": text,
            "reviewer_uid": user_uid,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": "human_investigator",
        }
        extraction["investigator_correction"] = correction
        connection.execute(
            "UPDATE case_reports SET report_json = ? WHERE user_uid = ? AND report_id = ?",
            (json.dumps(report, separators=(",", ":")), user_uid, report_id),
        )
    return correction

