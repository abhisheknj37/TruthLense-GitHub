from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "originals"


def _root() -> Path:
    configured = os.getenv("TRUTHLENS_ORIGINALS_DIR", "").strip()
    root = Path(configured).expanduser() if configured else DEFAULT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    try:
        root.chmod(0o700)
    except OSError:
        pass
    return root.resolve()


def _owner(uid: str) -> str:
    return hashlib.sha256(uid.encode("utf-8")).hexdigest()


def _id(value: str) -> str:
    if not re.fullmatch(r"[A-Fa-f0-9-]{16,64}", str(value or "")):
        raise ValueError("Invalid evidence identifier.")
    return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def store_original(
    source: Path, *, uid: str, report_id: str, asset_id: str,
    filename: str, digest: str, retain: bool,
) -> dict[str, Any]:
    if not retain:
        return {"status": "not_retained", "retention_requested": False}
    report_id, asset_id = _id(report_id), _id(asset_id)
    extension = Path(filename).suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,8}", extension):
        extension = ".evidence"
    root = _root()
    relative = Path(_owner(uid)) / report_id / (asset_id + extension)
    destination = (root / relative).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("Invalid private evidence storage path.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.parent.chmod(0o700)
    except OSError:
        pass
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    actual = sha256_file(temporary)
    if actual.lower() != digest.lower():
        temporary.unlink(missing_ok=True)
        raise ValueError("Retained original does not match the received SHA-256 digest.")
    try:
        temporary.chmod(0o600)
    except OSError:
        pass
    os.replace(temporary, destination)
    try:
        days = max(1, min(int(os.getenv("TRUTHLENS_ORIGINAL_RETENTION_DAYS", "30")), 3650))
    except ValueError:
        days = 30
    now = datetime.now(timezone.utc)
    return {
        "status": "retained",
        "storage_key": relative.as_posix(),
        "original_filename": Path(filename).name,
        "sha256": actual,
        "size_bytes": destination.stat().st_size,
        "stored_at": now.isoformat(),
        "expires_at": (now + timedelta(days=days)).isoformat(),
        "retention_days": days,
        "media_type": mimetypes.guess_type(filename)[0] or "application/octet-stream",
    }


def resolve_original(uid: str, report_id: str, asset: dict[str, Any]) -> tuple[Path | None, str]:
    record = asset.get("evidence_storage") or {}
    if record.get("status") != "retained":
        return None, "not_retained"
    try:
        report_id = _id(report_id)
        asset_id = _id(str(asset.get("asset_id") or ""))
        relative = Path(str(record.get("storage_key") or ""))
        root = _root()
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            return None, "invalid_reference"
        if tuple(relative.parts[:2]) != (_owner(uid), report_id) or not relative.name.startswith(asset_id + "."):
            return None, "invalid_reference"
        expiry = datetime.fromisoformat(str(record.get("expires_at")))
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if expiry <= datetime.now(timezone.utc):
            path.unlink(missing_ok=True)
            return None, "expired"
        if not path.is_file():
            return None, "missing"
        if sha256_file(path).lower() != str(asset.get("sha256") or "").lower():
            return None, "digest_mismatch"
        return path, "available"
    except (OSError, TypeError, ValueError):
        return None, "invalid_reference"


def delete_case_originals(uid: str, report_id: str) -> None:
    try:
        report_id = _id(report_id)
    except ValueError:
        return
    root = _root()
    folder = (root / _owner(uid) / report_id).resolve()
    if folder.is_relative_to(root) and folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)


def purge_expired_originals() -> dict[str, int]:
    root = _root()
    checked = removed = 0
    now = datetime.now(timezone.utc)
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix == ".tmp":
            continue
        checked += 1
        # Expiry is also checked at authenticated download; bulk purge is intentionally conservative.
    return {"files_checked": checked, "expired_originals_removed": removed}
