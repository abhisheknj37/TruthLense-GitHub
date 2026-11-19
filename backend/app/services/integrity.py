from __future__ import annotations

import hmac
import re
from typing import Any


_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def compare_sha256(client_digest: str | None, server_digest: str) -> dict[str, Any]:
    """Compare a client pre-upload SHA-256 with the digest of received bytes."""
    server_value = (server_digest or "").strip().lower()
    client_value = (client_digest or "").strip().lower()
    if not _SHA256_PATTERN.fullmatch(server_value):
        raise ValueError("The server SHA-256 digest is invalid.")
    if not client_value:
        return {
            "status": "not_compared",
            "server_sha256": server_value,
            "client_sha256": None,
        }
    if not _SHA256_PATTERN.fullmatch(client_value):
        return {
            "status": "invalid_client_digest",
            "server_sha256": server_value,
            "client_sha256": None,
        }
    return {
        "status": "match" if hmac.compare_digest(client_value, server_value) else "mismatch",
        "server_sha256": server_value,
        "client_sha256": client_value,
    }
