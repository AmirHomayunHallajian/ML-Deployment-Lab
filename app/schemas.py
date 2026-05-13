from typing import List

from pydantic import BaseModel, Field


class SiteFeatures(BaseModel):
    country: str
    therapeutic_area: str
    study_phase: str
    site_tier: str
    planned_enrollment: int = Field(ge=0)
    previous_trials_count: int = Field(ge=0)
    avg_query_resolution_days: float = Field(ge=0)
    missing_documents_count: int = Field(ge=0)
    startup_cycle_days: float = Field(ge=0)
    staff_training_completion_rate: float = Field(ge=0, le=100)
    edc_adoption_score: float = Field(ge=0, le=100)
    historical_enrollment_rate: float = Field(ge=0, le=1)
    monitoring_findings_count: int = Field(ge=0)


class PredictionResponse(BaseModel):
    site_delay_risk: int
    risk_probability: float
    risk_label: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
