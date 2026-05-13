from pathlib import Path

APP_NAME = "ML-Deployment-Lab"
APP_VERSION = "1.0.0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "site_delay_model.joblib"
LOG_DIR = PROJECT_ROOT / "logs"
PREDICTION_LOG_PATH = LOG_DIR / "predictions.log"
