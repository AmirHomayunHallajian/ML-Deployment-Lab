from typing import Iterable, List

import pandas as pd


def predict_from_records(model, records: Iterable[dict]) -> List[dict]:
    data = pd.DataFrame(list(records))
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]

    outputs = []
    for pred, prob in zip(predictions, probabilities):
        pred_int = int(pred)
        prob_float = float(prob)
        outputs.append(
            {
                "site_delay_risk": pred_int,
                "risk_probability": round(prob_float, 4),
                "risk_label": "High Risk" if pred_int == 1 else "Low Risk",
            }
        )
    return outputs
