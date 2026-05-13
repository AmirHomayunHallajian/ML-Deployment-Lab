from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def valid_payload() -> dict:
    return {
        "country": "Netherlands",
        "therapeutic_area": "Neuroscience",
        "study_phase": "Phase III",
        "site_tier": "A",
        "planned_enrollment": 40,
        "previous_trials_count": 8,
        "avg_query_resolution_days": 4.2,
        "missing_documents_count": 2,
        "startup_cycle_days": 65,
        "staff_training_completion_rate": 85,
        "edc_adoption_score": 78,
        "historical_enrollment_rate": 0.72,
        "monitoring_findings_count": 3,
    }


def test_root_endpoint() -> None:
    resp = client.get("/")
    assert resp.status_code == 200


def test_health_endpoint() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "model_loaded" in resp.json()


def test_predict_valid_or_model_missing() -> None:
    resp = client.post("/predict", json=valid_payload())
    assert resp.status_code in (200, 503)


def test_predict_invalid_payload() -> None:
    bad = valid_payload()
    bad["planned_enrollment"] = -1
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_monitoring_summary() -> None:
    resp = client.get("/monitoring/summary")
    assert resp.status_code == 200
