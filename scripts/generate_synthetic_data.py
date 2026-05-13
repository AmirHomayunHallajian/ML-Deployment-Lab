from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_dataset(n_rows: int = 2200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    countries = ["United States", "Germany", "Netherlands", "Spain", "Poland", "Brazil", "Japan"]
    therapeutic_areas = ["Oncology", "Cardiology", "Immunology", "Neuroscience", "Endocrinology"]
    study_phases = ["Phase I", "Phase II", "Phase III", "Phase IV"]
    site_tiers = ["A", "B", "C"]

    df = pd.DataFrame(
        {
            "site_id": [f"SITE_{i:04d}" for i in range(n_rows)],
            "study_id": [f"STUDY_{rng.integers(1, 220):04d}" for _ in range(n_rows)],
            "country": rng.choice(countries, n_rows),
            "therapeutic_area": rng.choice(therapeutic_areas, n_rows),
            "study_phase": rng.choice(study_phases, n_rows, p=[0.15, 0.35, 0.4, 0.1]),
            "site_tier": rng.choice(site_tiers, n_rows, p=[0.3, 0.5, 0.2]),
            "planned_enrollment": rng.integers(10, 180, n_rows),
            "previous_trials_count": rng.integers(0, 35, n_rows),
            "avg_query_resolution_days": rng.normal(7.5, 3.0, n_rows).clip(0.5, 25),
            "missing_documents_count": rng.poisson(3.0, n_rows),
            "startup_cycle_days": rng.normal(80, 25, n_rows).clip(20, 220),
            "staff_training_completion_rate": rng.normal(82, 12, n_rows).clip(30, 100),
            "edc_adoption_score": rng.normal(76, 14, n_rows).clip(20, 100),
            "historical_enrollment_rate": rng.normal(0.66, 0.18, n_rows).clip(0.1, 1.0),
            "monitoring_findings_count": rng.poisson(4.0, n_rows),
        }
    )

    risk_score = (
        0.28 * df["missing_documents_count"]
        + 0.03 * df["startup_cycle_days"]
        + 0.22 * df["avg_query_resolution_days"]
        - 0.045 * df["staff_training_completion_rate"]
        - 0.03 * df["edc_adoption_score"]
        - 0.04 * df["previous_trials_count"]
        - 1.8 * df["historical_enrollment_rate"]
        + 0.15 * df["monitoring_findings_count"]
        + rng.normal(0, 1.2, n_rows)
    )

    threshold = np.quantile(risk_score, 0.56)
    df["site_delay_risk"] = (risk_score > threshold).astype(int)
    return df


def main() -> None:
    output_path = Path("data/processed/site_delay_data.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_dataset()
    df.to_csv(output_path, index=False)

    print(f"Saved synthetic dataset to {output_path} with shape={df.shape}")
    print("Class balance:")
    print(df["site_delay_risk"].value_counts(normalize=True).rename("proportion"))
    print("Numeric summary:")
    print(df.describe(include="all").transpose().head(15))


if __name__ == "__main__":
    main()
