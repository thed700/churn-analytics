"""
Unit tests for feature engineering module.
"""
import pytest
import pandas as pd
import numpy as np

from src.features.engineer import ChurnFeatureEngineer


@pytest.fixture
def sample_df():
    """Minimal DataFrame mimicking preprocessed Telco data."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "tenure": np.random.randint(1, 72, n),
        "MonthlyCharges": np.random.uniform(20, 120, n),
        "TotalCharges": np.random.uniform(100, 8000, n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "PhoneService": np.random.randint(0, 2, n),
        "MultipleLines": np.random.randint(0, 2, n),
        "OnlineSecurity": np.random.randint(0, 2, n),
        "OnlineBackup": np.random.randint(0, 2, n),
        "DeviceProtection": np.random.randint(0, 2, n),
        "TechSupport": np.random.randint(0, 2, n),
        "StreamingTV": np.random.randint(0, 2, n),
        "StreamingMovies": np.random.randint(0, 2, n),
        "Churn": np.random.randint(0, 2, n),
    })


def test_engineer_adds_all_features(sample_df):
    """All 5 engineered features must be present after transform."""
    eng = ChurnFeatureEngineer()
    eng.fit(sample_df)
    result = eng.transform(sample_df)

    expected_features = [
        "charge_volatility_ratio",
        "service_adoption_density",
        "tenure_contract_interaction",
        "support_recency_score",
        "cohort_clv_percentile",
    ]
    for feat in expected_features:
        assert feat in result.columns, f"Missing feature: {feat}"


def test_service_density_range(sample_df):
    """service_adoption_density must be between 0 and 1."""
    eng = ChurnFeatureEngineer()
    eng.fit(sample_df)
    result = eng.transform(sample_df)
    assert result["service_adoption_density"].between(0, 1).all()


def test_charge_volatility_no_negative(sample_df):
    """charge_volatility_ratio must be non-negative."""
    eng = ChurnFeatureEngineer()
    eng.fit(sample_df)
    result = eng.transform(sample_df)
    assert (result["charge_volatility_ratio"] >= 0).all()


def test_cohort_percentile_range(sample_df):
    """cohort_clv_percentile must be in (0, 1]."""
    eng = ChurnFeatureEngineer()
    eng.fit(sample_df)
    result = eng.transform(sample_df)
    assert result["cohort_clv_percentile"].between(0, 1, inclusive="right").all()


def test_tenure_contract_interaction_positive(sample_df):
    """tenure_contract_interaction must be positive (tenure > 0, contract months > 0)."""
    eng = ChurnFeatureEngineer()
    eng.fit(sample_df)
    result = eng.transform(sample_df)
    assert (result["tenure_contract_interaction"] > 0).all()


def test_no_data_leakage(sample_df):
    """Fit on train, transform on test — no errors, no NaN in key features."""
    train = sample_df.iloc[:80]
    test = sample_df.iloc[80:]

    eng = ChurnFeatureEngineer()
    eng.fit(train)
    result = eng.transform(test)

    key_features = ["charge_volatility_ratio", "service_adoption_density",
                     "tenure_contract_interaction", "cohort_clv_percentile"]
    for feat in key_features:
        assert result[feat].isna().sum() == 0, f"NaN in {feat} after transform"
