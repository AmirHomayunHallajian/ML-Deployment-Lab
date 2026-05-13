# API Usage

## GET /
Purpose: Returns project metadata.

## GET /health
Purpose: Service and model health status.

## POST /predict
Purpose: Predict delay risk for one site.

Example:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "monitoring_findings_count": 3
  }'
```

Response format:
```json
{
  "site_delay_risk": 0,
  "risk_probability": 0.241,
  "risk_label": "Low Risk"
}
```

## POST /predict_batch
Purpose: Predict delay risk for multiple sites.

## GET /monitoring/summary
Purpose: Returns runtime prediction counters and aggregate statistics.
