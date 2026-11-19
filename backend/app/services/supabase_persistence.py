from __future__ import annotations

import os
from typing import Any

import httpx


class SupabasePersistenceError(RuntimeError):
    pass


def is_configured() -> bool:
    return bool(os.getenv("SUPABASE_URL", "").strip() and os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip())


def persist_case(report: dict[str, Any], firebase_uid: str, case_title: str) -> bool:
    """Mirror a saved case to Supabase using a server-only service-role key.

    Firebase remains the identity provider. The service role must never be exposed
    to the browser; ownership is bound to the verified Firebase UID by this API.
    """
    if not is_configured():
        return False

    base_url = os.environ["SUPABASE_URL"].rstrip("/")
    service_key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    report_id = str(report["report_id"])
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }
    case = {
        "id": report_id,
        "firebase_uid": firebase_uid,
        "case_title": case_title,
        "overall_status": report.get("overall_status", "inconclusive"),
        "summary": report.get("summary", ""),
        "report_json": report,
        "created_at": report.get("created_at"),
    }
    assets = []
    for asset in report.get("assets", []):
        assets.append({
            "case_id": report_id,
            "firebase_uid": firebase_uid,
            "asset_id": asset["asset_id"],
            "filename": asset["filename"],
            "media_type": asset["media_type"],
            "size_bytes": asset["size_bytes"],
            "sha256": asset["sha256"],
            "analysis_status": asset.get("analysis_status", "inconclusive"),
            "metadata": asset.get("metadata", {}),
            "observations": asset.get("observations", []),
            "model_assessment": asset.get("model_assessment", {}),
            "video_analysis": asset.get("video_analysis", {}),
            "forensic_parameters": asset.get("forensic_parameters", {}),
            "provenance": asset.get("provenance", {}),
            "text_extraction": asset.get("text_extraction", {}),
        })
    events = []
    for event in report.get("evidence_events", []):
        events.append({
            "case_id": report_id,
            "firebase_uid": firebase_uid,
            "event_type": event.get("event", "evidence_event"),
            "payload": event,
            "previous_hash": event.get("previous_event_hash", "0" * 64),
            "event_hash": event["event_hash"],
            "created_at": event.get("timestamp"),
        })

    try:
        with httpx.Client(timeout=25.0) as client:
            for table, rows in (
                ("truthlens_cases", [case]),
                ("truthlens_assets", assets),
                ("truthlens_evidence_events", events),
            ):
                if not rows:
                    continue
                response = client.post(
                    f"{base_url}/rest/v1/{table}",
                    headers=headers,
                    json=rows,
                )
                if response.is_error:
                    raise SupabasePersistenceError(
                        f"Supabase write to {table} failed with HTTP {response.status_code}."
                    )
    except httpx.HTTPError as exc:
        raise SupabasePersistenceError("Could not reach Supabase while syncing the case.") from exc
    return True
