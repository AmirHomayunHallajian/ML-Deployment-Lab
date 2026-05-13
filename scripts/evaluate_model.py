import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def main() -> None:
    data_path = Path("data/processed/site_delay_data.csv")
    model_path = Path("models/site_delay_model.joblib")
    metrics_path = Path("reports/metrics.json")
    report_path = Path("reports/model_report.md")

    df = pd.read_csv(data_path)
    X = df.drop(columns=["site_delay_risk", "site_id", "study_id"])
    y = df["site_delay_risk"]

    model = joblib.load(model_path)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_prob),
        "evaluation_samples": int(len(df)),
    }

    importances_text = ""
    clf = model.named_steps.get("classifier")
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        top_features = sorted(enumerate(importances), key=lambda x: x[1], reverse=True)[:10]
        importances_text = "\n## Top Feature Importances (transformed feature index)\n"
        importances_text += "\n".join([f"- Feature_{idx}: {score:.4f}" for idx, score in top_features])

    metrics_path.write_text(json.dumps(metrics, indent=2))
    report = ["# Model Report", "", "## Evaluation Metrics"]
    report.extend([f"- **{k}**: {v:.4f}" if isinstance(v, float) else f"- **{k}**: {v}" for k, v in metrics.items()])
    report_path.write_text("\n".join(report) + "\n" + importances_text + "\n")

    print("Evaluation complete")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
