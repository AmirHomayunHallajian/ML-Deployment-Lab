from scripts.generate_synthetic_data import generate_synthetic_dataset


def test_generate_synthetic_data_required_columns() -> None:
    df = generate_synthetic_dataset(n_rows=200, seed=123)
    required_columns = {
        "site_id", "study_id", "country", "therapeutic_area", "study_phase", "site_tier",
        "planned_enrollment", "previous_trials_count", "avg_query_resolution_days",
        "missing_documents_count", "startup_cycle_days", "staff_training_completion_rate",
        "edc_adoption_score", "historical_enrollment_rate", "monitoring_findings_count",
        "site_delay_risk",
    }
    assert required_columns.issubset(df.columns)


def test_target_has_both_classes() -> None:
    df = generate_synthetic_dataset(n_rows=400, seed=55)
    assert set(df["site_delay_risk"].unique()) == {0, 1}


def test_value_ranges() -> None:
    df = generate_synthetic_dataset(n_rows=200, seed=7)
    assert (df["planned_enrollment"] >= 0).all()
    assert (df["staff_training_completion_rate"].between(0, 100)).all()
    assert (df["edc_adoption_score"].between(0, 100)).all()
    assert (df["historical_enrollment_rate"].between(0, 1)).all()
