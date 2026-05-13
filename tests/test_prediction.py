from pathlib import Path

import pytest

from app.model_loader import load_model
from app.prediction import predict_from_records


@pytest.mark.skipif(not Path("models/site_delay_model.joblib").exists(), reason="Model artifact not found")
def test_model_loading_and_prediction_format() -> None:
    model = load_model()
    sample = {
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
    out = predict_from_records(model, [sample])[0]
    assert set(out.keys()) == {"site_delay_risk", "risk_probability", "risk_label"}
    assert out["site_delay_risk"] in [0, 1]
