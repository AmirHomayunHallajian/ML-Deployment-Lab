import joblib

from app.config import MODEL_PATH


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run training first: make data && make train"
        )
    return joblib.load(MODEL_PATH)
