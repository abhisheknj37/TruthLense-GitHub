from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_capabilities_disclose_model_status():
    response = client.get("/api/capabilities")
    assert response.status_code == 200
    assert response.json()["authenticity_models"] == "not_configured"
