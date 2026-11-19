from __future__ import annotations

import os


def is_postgres_enabled() -> bool:
    """Use Supabase PostgreSQL when DATABASE_URL is configured; keep SQLite for local dev."""
    return bool(os.getenv("DATABASE_URL", "").strip())


def connect_postgres():
    """Open an SSL-required backend-only connection to Supabase PostgreSQL."""
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured.")
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError('Install the pinned psycopg[binary] dependency to use Supabase PostgreSQL.') from exc
    return psycopg.connect(database_url, row_factory=dict_row, sslmode="require", connect_timeout=10)
