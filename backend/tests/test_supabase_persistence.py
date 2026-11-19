import json

from app.services import supabase_persistence


class FakeResponse:
    is_error = False
    status_code = 201


class FakeClient:
    def __init__(self, timeout):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, headers, json):
        self.calls.append((url, json))
        return FakeResponse()


def test_supabase_persistence_is_optional_when_secrets_are_missing(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    assert supabase_persistence.persist_case({"report_id": "case"}, "firebase-user", "Case") is False


def test_supabase_persistence_writes_case_assets_and_events(monkeypatch):
    fake = FakeClient
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only-test-key")
    monkeypatch.setattr(supabase_persistence.httpx, "Client", fake)
    report = {
        "report_id": "00000000-0000-0000-0000-000000000001",
        "created_at": "2026-11-19T00:00:00+00:00",
        "overall_status": "inconclusive",
        "summary": "Review required",
        "assets": [{
            "asset_id": "asset-1", "filename": "sample.png", "media_type": "image",
            "size_bytes": 10, "sha256": "a" * 64, "analysis_status": "inconclusive",
        }],
        "evidence_events": [{
            "event": "case_created", "event_hash": "b" * 64,
            "previous_event_hash": "0" * 64, "timestamp": "2026-11-19T00:00:00+00:00",
        }],
    }
    assert supabase_persistence.persist_case(report, "firebase-user", "Case") is True
