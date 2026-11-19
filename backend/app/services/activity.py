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

from app.services.database import connect_postgres, is_postgres_enabled
from app.services.storage import delete_evidence_objects


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
    overlay_files: list[tuple[str, int, Any]] = []
    use_postgres = is_postgres_enabled()
    if not use_postgres:
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
                    face_index = int(face.get("face_index", len(face_summary)))
                    if use_postgres:
                        overlay_files.append((str(asset["asset_id"]), face_index, image_bytes))
                    else:
                        report_folder.mkdir(parents=True, exist_ok=True)
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
    if use_postgres:
        from psycopg.types.json import Jsonb

        with connect_postgres() as connection:
            case = connection.execute(
                """INSERT INTO public.truthlens_cases
                   (firebase_uid, report_id, case_title, overall_status, summary, report_json, deleted_at)
                   VALUES (%s, %s, %s, %s, %s, %s, NULL)
                   ON CONFLICT (firebase_uid, report_id) WHERE report_id IS NOT NULL DO UPDATE SET
                     case_title = EXCLUDED.case_title, overall_status = EXCLUDED.overall_status,
                     summary = EXCLUDED.summary, report_json = EXCLUDED.report_json,
                     deleted_at = NULL, updated_at = now()
                   RETURNING id""",
                (user_uid, report_id, case_title, str(report.get("overall_status") or "inconclusive"),
                 str(report.get("summary") or ""), Jsonb(report)),
            ).fetchone()
            case_id = case["id"]
            connection.execute("DELETE FROM public.truthlens_assets WHERE case_id = %s", (case_id,))
            for asset in report.get("assets", []):
                connection.execute(
                    """INSERT INTO public.truthlens_assets
                       (case_id, firebase_uid, asset_id, filename, media_type, size_bytes, sha256,
                        analysis_status, storage_path, metadata, observations, model_assessment,
                        video_analysis, forensic_parameters, provenance, text_extraction)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (case_id, user_uid, str(asset.get("asset_id") or ""), str(asset.get("filename") or "evidence"),
                     str(asset.get("media_type") or "image"), int(asset.get("size_bytes") or 0),
                     str(asset.get("sha256") or "0" * 64), str(asset.get("analysis_status") or "inconclusive"),
                     asset.get("storage_path"), Jsonb(asset.get("metadata") or {}), Jsonb(asset.get("observations") or []),
                     Jsonb(asset.get("model_assessment") or {}), Jsonb(asset.get("video_analysis") or {}),
                     Jsonb(asset.get("forensic_parameters") or {}), Jsonb(asset.get("provenance") or {}),
                     Jsonb(asset.get("text_extraction") or {})),
                )
            connection.execute(
                """INSERT INTO public.truthlens_activity
                   (case_id, firebase_uid, event_type, case_title, overall_status,
                    evidence_count, verified_count, evidence_json)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (case_id) DO UPDATE SET firebase_uid = EXCLUDED.firebase_uid,
                     event_type = EXCLUDED.event_type, case_title = EXCLUDED.case_title,
                     overall_status = EXCLUDED.overall_status, evidence_count = EXCLUDED.evidence_count,
                     verified_count = EXCLUDED.verified_count, evidence_json = EXCLUDED.evidence_json,
                     created_at = now()""",
                (case_id, user_uid, event["event_type"], case_title, event["overall_status"],
                 event["evidence_count"], event["verified_count"], Jsonb(evidence_summary)),
            )
            connection.execute("DELETE FROM public.truthlens_heatmaps WHERE case_id = %s AND firebase_uid = %s",
                               (case_id, user_uid))
            connection.executemany(
                """INSERT INTO public.truthlens_heatmaps
                   (case_id, firebase_uid, asset_id, face_index, image_data)
                   VALUES (%s, %s, %s, %s, %s)""",
                [(case_id, user_uid, asset_id, face_index, image_data)
                 for asset_id, face_index, image_data in overlay_files],
            )
        return event

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


def _decode_json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def list_user_activity(user_uid: str, limit: int = 20) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 50))
    if is_postgres_enabled():
        with connect_postgres() as connection:
            rows = connection.execute(
                """SELECT a.id::text AS event_id, a.event_type, c.report_id, a.case_title,
                          a.overall_status, a.evidence_count, a.verified_count, a.created_at, a.evidence_json
                   FROM public.truthlens_activity a JOIN public.truthlens_cases c ON c.id = a.case_id
                   WHERE a.firebase_uid = %s AND c.deleted_at IS NULL
                   ORDER BY a.created_at DESC, a.id DESC LIMIT %s""", (user_uid, safe_limit),
            ).fetchall()
        return [{
            "event_id": row["event_id"], "event_type": row["event_type"], "report_id": row["report_id"],
            "case_title": row["case_title"], "overall_status": row["overall_status"],
            "evidence_count": row["evidence_count"], "verified_count": row["verified_count"],
            "created_at": row["created_at"].isoformat(), "evidence": _decode_json(row["evidence_json"]),
        } for row in rows]
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
    if is_postgres_enabled():
        with connect_postgres() as connection:
            rows = connection.execute(
                """SELECT a.evidence_count, a.verified_count, a.evidence_json
                   FROM public.truthlens_activity a JOIN public.truthlens_cases c ON c.id = a.case_id
                   WHERE a.firebase_uid = %s AND c.deleted_at IS NULL""", (user_uid,),
            ).fetchall()
        total_evidence = sum(int(row["evidence_count"]) for row in rows)
        total_verified = sum(int(row["verified_count"]) for row in rows)
        manipulation_signals = sum(1 for row in rows for item in _decode_json(row["evidence_json"])
                                   if item.get("prediction") == "fake_signal")
        return {"case_count": len(rows), "evidence_count": total_evidence,
                "verified_count": total_verified, "manipulation_signals": manipulation_signals}
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
    if is_postgres_enabled():
        with connect_postgres() as connection:
            row = connection.execute(
                """SELECT id, report_json FROM public.truthlens_cases
                   WHERE firebase_uid = %s AND report_id = %s AND deleted_at IS NULL""", (user_uid, report_id),
            ).fetchone()
            if row is None:
                return None
            report = _decode_json(row["report_json"])
            heatmaps = connection.execute(
                """SELECT asset_id, face_index, image_data FROM public.truthlens_heatmaps
                   WHERE case_id = %s AND firebase_uid = %s""", (row["id"], user_uid),
            ).fetchall()
        by_asset: dict[str, dict[int, bytes]] = {}
        for item in heatmaps:
            by_asset.setdefault(item["asset_id"], {})[int(item["face_index"])] = bytes(item["image_data"])
        for asset in report.get("assets", []):
            for face in (asset.get("model_assessment") or {}).get("faces", []):
                image_data = by_asset.get(str(asset.get("asset_id")), {}).get(int(face.get("face_index", -1)))
                if image_data:
                    face["gradcam_overlay_base64"] = base64.b64encode(image_data).decode("ascii")
                    face["heatmap_available"] = True
        return report
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


def get_user_heatmap_bytes(user_uid: str, report_id: str, asset_id: str, face_index: int) -> bytes | None:
    if is_postgres_enabled():
        with connect_postgres() as connection:
            row = connection.execute(
                """SELECT h.image_data FROM public.truthlens_heatmaps h
                   JOIN public.truthlens_cases c ON c.id = h.case_id
                   WHERE c.firebase_uid = %s AND c.report_id = %s AND c.deleted_at IS NULL
                     AND h.firebase_uid = %s AND h.asset_id = %s AND h.face_index = %s""",
                (user_uid, report_id, user_uid, asset_id, face_index),
            ).fetchone()
        return bytes(row["image_data"]) if row else None
    path = get_user_heatmap_path(user_uid, report_id, asset_id, face_index)
    return path.read_bytes() if path and path.is_file() else None


def get_user_original_info(user_uid: str, report_id: str, asset_id: str) -> dict[str, str] | None:
    if not is_postgres_enabled():
        return None
    with connect_postgres() as connection:
        row = connection.execute(
            """SELECT a.storage_path, a.filename, a.media_type FROM public.truthlens_assets a
               JOIN public.truthlens_cases c ON c.id = a.case_id
               WHERE c.firebase_uid = %s AND c.report_id = %s AND c.deleted_at IS NULL
                 AND a.firebase_uid = %s AND a.asset_id = %s""",
            (user_uid, report_id, user_uid, asset_id),
        ).fetchone()
    if not row or not row["storage_path"]:
        return None
    return {"storage_path": row["storage_path"], "filename": row["filename"], "media_type": row["media_type"]}


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
    """Soft-delete a case, retain its evidence audit chain, and remove originals/heatmaps."""
    if is_postgres_enabled():
        with connect_postgres() as connection:
            case = connection.execute(
                """SELECT id FROM public.truthlens_cases
                   WHERE firebase_uid = %s AND report_id = %s AND deleted_at IS NULL""",
                (user_uid, report_id),
            ).fetchone()
            if case is None:
                return False
            paths = [row["storage_path"] for row in connection.execute(
                "SELECT storage_path FROM public.truthlens_assets WHERE case_id = %s AND storage_path IS NOT NULL",
                (case["id"],),
            ).fetchall()]
            connection.execute(
                "UPDATE public.truthlens_cases SET deleted_at = now(), updated_at = now() WHERE id = %s AND firebase_uid = %s",
                (case["id"], user_uid),
            )
            connection.execute("DELETE FROM public.truthlens_heatmaps WHERE case_id = %s AND firebase_uid = %s",
                               (case["id"], user_uid))
        if paths:
            try:
                delete_evidence_objects(paths)
            except Exception:
                pass
        return True
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
