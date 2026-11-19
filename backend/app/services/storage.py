from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import quote

import httpx


def storage_is_configured() -> bool:
    return bool(
        os.getenv("DATABASE_URL", "").strip()
        and os.getenv("SUPABASE_URL", "").strip()
        and os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    )


def _settings() -> tuple[str, str, str]:
    base = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "truthlens-evidence").strip()
    if not base or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required for private evidence storage.")
    return base, key, bucket


def _headers(key: str, content_type: str = "application/octet-stream") -> dict[str, str]:
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": content_type}


def upload_original_media(
    source: Path, *, owner_uid: str, asset_id: str, filename: str, content_type: str
) -> str | None:
    """Upload original bytes into the private bucket; return its opaque object path."""
    if not storage_is_configured():
        return None
    base, key, bucket = _settings()
    safe_name = filename.replace("\\", "/").split("/")[-1] or "evidence.bin"
    owner_folder = hashlib.sha256(owner_uid.encode("utf-8")).hexdigest()[:32]
    object_path = f"{owner_folder}/{asset_id}/{safe_name}"
    url = f"{base}/storage/v1/object/{quote(bucket, safe='')}/{quote(object_path, safe='/')}"
    headers = _headers(key, content_type)
    headers["x-upsert"] = "true"
    with source.open("rb") as media_file, httpx.Client(timeout=120) as client:
        response = client.post(url, content=media_file.read(), headers=headers)
        response.raise_for_status()
    return object_path


def download_original_media(object_path: str) -> bytes:
    base, key, bucket = _settings()
    url = f"{base}/storage/v1/object/{quote(bucket, safe='')}/{quote(object_path, safe='/')}"
    with httpx.Client(timeout=120) as client:
        response = client.get(url, headers=_headers(key))
        response.raise_for_status()
        return response.content


def delete_evidence_objects(object_paths: list[str]) -> None:
    if not object_paths or not storage_is_configured():
        return
    base, key, bucket = _settings()
    url = f"{base}/storage/v1/object/{quote(bucket, safe='')}"
    with httpx.Client(timeout=60) as client:
        response = client.request(
            "DELETE", url, json={"prefixes": object_paths}, headers=_headers(key, "application/json")
        )
        response.raise_for_status()
