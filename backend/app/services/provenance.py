from __future__ import annotations

from pathlib import Path
from typing import Any


def inspect_provenance(path: Path, sha256: str, original_filename: str) -> dict[str, Any]:
    """Record file provenance facts without inferring authenticity."""
    return {
        "original_filename": original_filename,
        "sha256": sha256,
        "hash_algorithm": "SHA-256",
        "hash_scope": "Exact uploaded file bytes",
        "integrity_status": "fingerprint_recorded",
        "content_credentials": {
            "status": "not_verified",
            "standard": "C2PA Content Credentials",
            "detail": "C2PA validation status is supplied separately by the optional validator.",
        },
        "limitations": [
            "A hash identifies bytes; it does not prove creator identity or content truth.",
            "A request-level event list is not independently timestamped or tamper-proof.",
        ],
    }
