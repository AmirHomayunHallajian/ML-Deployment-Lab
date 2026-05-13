# ML-Deployment-Lab: FastAPI + Docker + Monitoring Demo

## Overview
This project demonstrates an end-to-end ML deployment workflow: synthetic data generation, model training, artifact persistence, FastAPI model serving, Dockerization, tests, and basic monitoring.

## Problem
Many data-science projects stop at notebooks. In real organizations, models need to be served through APIs, tested, containerized, monitored, and documented.

## Solution
This repository trains a simple site-delay risk model using synthetic clinical-trial operations data and exposes predictions through a FastAPI service.

## Why This Project Matters
- API serving
- Docker
- model artifact management
- inference endpoint
- unit testing
- monitoring/logging
- reproducible commands
- separation between training and serving code

## Use Case
Synthetic clinical-trial site delay prediction. Clinical operations teams could use similar models to prioritize site support, identify delayed start-up risk, and monitor operational bottlenecks.

## Tech Stack
Python, pandas, scikit-learn, FastAPI, Pydantic, Docker, pytest, joblib, uvicorn.

## Repository Structure
See project folders under `app/`, `scripts/`, `tests/`, `docs/`, `data/`, `models/`, and `reports/`.

## Quickstart
1. Clone repository and enter directory.
2. Create virtual environment:
   - `python -m venv .venv`
   - `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
3. Install dependencies: `pip install -r requirements.txt` or `make install`
4. Generate synthetic data: `make data`
5. Train model: `make train`
6. Run tests: `make test`
7. Start API: `make run`
8. Open `http://localhost:8000/docs`

## API Usage
Example request:
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

Example response:
```json
{
  "site_delay_risk": 0,
  "risk_probability": 0.241,
  "risk_label": "Low Risk"
}
```

## Docker Usage
- `docker build -t ml-deployment-lab .`
- `docker run -p 8000:8000 ml-deployment-lab`
- `docker compose up --build`

## Endpoints
- GET /
- GET /health
- POST /predict
- POST /predict_batch
- GET /monitoring/summary

## Model Training
Uses synthetic dataset, preprocessing pipeline (imputation + one-hot encoding + scaling), RandomForest classifier, saved artifact, and standard classification metrics.

## Monitoring
Prediction logs are written to `logs/predictions.log`. Runtime counters are available at `/monitoring/summary`.

## Testing
Run `pytest`.

## Future Improvements
Prometheus/Grafana metrics, MLflow tracking, CI/CD with GitHub Actions, model versioning, drift detection, batch scoring, auth, cloud deployment, Kubernetes, database-backed logging, async inference queue, and richer simulation.

