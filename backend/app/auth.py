from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import firebase_admin
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth as firebase_auth
from firebase_admin import exceptions as firebase_exceptions
from firebase_admin import credentials

bearer_scheme = HTTPBearer(auto_error=False)


def _firebase_app() -> firebase_admin.App:
    """Return the configured Firebase Admin app; fail closed if credentials are absent."""
    project_id = os.getenv("FIREBASE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise RuntimeError("FIREBASE_PROJECT_ID is not configured.")

    try:
        return firebase_admin.get_app()
    except ValueError:
        service_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        service_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
        if service_json:
            try:
                credential_info = json.loads(service_json)
            except json.JSONDecodeError as exc:
                raise RuntimeError("FIREBASE_SERVICE_ACCOUNT_JSON must contain valid JSON.") from exc
            credential = credentials.Certificate(credential_info)
        elif service_path:
            path = Path(service_path)
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[1] / path
            if not path.is_file():
                raise RuntimeError(f"Firebase service-account file not found: {path}")
            credential = credentials.Certificate(str(path))
        else:
            # On Google Cloud, use Application Default Credentials.
            credential = credentials.ApplicationDefault()

        try:
            return firebase_admin.initialize_app(
                credential,
                options={"projectId": project_id},
            )
        except ValueError:
            return firebase_admin.get_app()


def get_current_user(
    bearer: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, Any]:
    """Verify a Firebase-issued JWT ID token and return trusted identity claims."""
    if bearer is None or bearer.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in to access this endpoint.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        app = _firebase_app()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firebase authentication is not configured on this server.",
        ) from exc

    try:
        decoded = firebase_auth.verify_id_token(
            bearer.credentials,
            app=app,
            check_revoked=True,
        )
    except (firebase_exceptions.FirebaseError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Your sign-in token is invalid or expired. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return {
        "uid": decoded["uid"],
        "email": decoded.get("email"),
        "email_verified": bool(decoded.get("email_verified", False)),
        "name": decoded.get("name"),
    }
