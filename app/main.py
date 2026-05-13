from fastapi import FastAPI, HTTPException

from app.config import APP_NAME, APP_VERSION, MODEL_PATH
from app.model_loader import load_model
from app.monitoring import monitoring_store
from app.prediction import predict_from_records
from app.schemas import BatchPredictionResponse, PredictionResponse, SiteFeatures

app = FastAPI(title=APP_NAME, version=APP_VERSION)


@app.get("/")
def root() -> dict:
    return {
        "project": APP_NAME,
        "description": "End-to-end model training, API serving, Dockerization, and monitoring demo.",
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH.relative_to(MODEL_PATH.parent.parent)),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: SiteFeatures):
    try:
        model = load_model()
        result = predict_from_records(model, [payload.model_dump()])[0]
        monitoring_store.record_prediction(
            probability=result["risk_probability"],
            predicted_class=result["site_delay_risk"],
            risk_label=result["risk_label"],
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(payload: list[SiteFeatures]):
    try:
        model = load_model()
        records = [item.model_dump() for item in payload]
        results = predict_from_records(model, records)
        for result in results:
            monitoring_store.record_prediction(
                probability=result["risk_probability"],
                predicted_class=result["site_delay_risk"],
                risk_label=result["risk_label"],
            )
        return {"predictions": results}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/monitoring/summary")
def monitoring_summary() -> dict:
    return monitoring_store.summary()
