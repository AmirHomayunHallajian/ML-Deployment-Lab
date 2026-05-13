import logging
from datetime import datetime, timezone

from app.config import LOG_DIR, PREDICTION_LOG_PATH

LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("prediction_logger")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(PREDICTION_LOG_PATH)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


class MonitoringStore:
    def __init__(self) -> None:
        self.prediction_count = 0
        self.high_risk_count = 0
        self.low_risk_count = 0
        self.cumulative_probability = 0.0
        self.service_start_time = datetime.now(timezone.utc).isoformat()

    def record_prediction(self, probability: float, predicted_class: int, risk_label: str) -> None:
        self.prediction_count += 1
        self.cumulative_probability += probability
        if predicted_class == 1:
            self.high_risk_count += 1
        else:
            self.low_risk_count += 1

        log_line = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "risk_probability": round(probability, 4),
            "predicted_class": int(predicted_class),
            "risk_label": risk_label,
        }
        logger.info(str(log_line))

    def summary(self) -> dict:
        avg_prob = (
            self.cumulative_probability / self.prediction_count if self.prediction_count else 0.0
        )
        return {
            "number_of_predictions_made": self.prediction_count,
            "average_predicted_risk_probability": round(avg_prob, 4),
            "high_risk_prediction_count": self.high_risk_count,
            "low_risk_prediction_count": self.low_risk_count,
            "service_start_time": self.service_start_time,
        }


monitoring_store = MonitoringStore()
