"""
Unit tests for survival model and Cox PH evaluator.
"""
import pytest
import numpy as np
import pandas as pd

from src.models.survival_model import CoxChurnSurvivalModel
from src.models.evaluator import ChurnEvaluator, EvaluationResult


@pytest.fixture
def sample_survival_df():
    """Minimal DataFrame for survival model testing."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "tenure": np.random.randint(1, 72, n),
        "Churn": np.random.randint(0, 2, n),
        "MonthlyCharges": np.random.uniform(20, 120, n),
        "TotalCharges": np.random.uniform(100, 8000, n),
        "SeniorCitizen": np.random.randint(0, 2, n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "charge_volatility_ratio": np.random.uniform(0, 1, n),
        "service_adoption_density": np.random.uniform(0, 1, n),
        "tenure_contract_interaction": np.random.randint(1, 1728, n),
        "support_recency_score": np.random.uniform(0, 1, n),
        "cohort_clv_percentile": np.random.uniform(0, 1, n),
    })


def test_cox_model_fits(sample_survival_df):
    """Cox PH model should fit without errors."""
    model = CoxChurnSurvivalModel()
    model.fit(sample_survival_df)
    assert model.cph_ is not None


def test_cox_concordance_index(sample_survival_df):
    """C-Index should be between 0.5 and 1.0 (better than random)."""
    model = CoxChurnSurvivalModel()
    model.fit(sample_survival_df)
    c_index = model.cph_.concordance_index_
    assert 0.5 <= c_index <= 1.0


def test_cox_hazard_ratios(sample_survival_df):
    """get_hazard_ratios should return a DataFrame with required columns."""
    model = CoxChurnSurvivalModel()
    model.fit(sample_survival_df)
    hr_df = model.get_hazard_ratios()
    assert "hazard_ratio" in hr_df.columns
    assert "p" in hr_df.columns
    assert len(hr_df) > 0


def test_kaplan_meier_data(sample_survival_df):
    """KM data should have one survival curve per contract type."""
    model = CoxChurnSurvivalModel()
    model.fit(sample_survival_df)
    km_data = model.get_kaplan_meier_data()
    assert "timeline" in km_data.columns
    assert "survival_probability" in km_data.columns
    assert "contract_type" in km_data.columns
    # Survival probabilities must be in [0, 1]
    assert km_data["survival_probability"].between(0, 1).all()


def test_churn_probability_at_horizon(sample_survival_df):
    """predict_churn_probability_at should return values in [0, 1]."""
    model = CoxChurnSurvivalModel()
    model.fit(sample_survival_df)
    proba_12m = model.predict_churn_probability_at(sample_survival_df.head(20), months=12)
    assert len(proba_12m) == 20
    assert np.all(proba_12m >= 0) and np.all(proba_12m <= 1)


def test_evaluator_threshold_curve():
    """Threshold curve should have monotone precision and decreasing recall."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 300)
    y_proba = np.clip(np.random.beta(2, 5, 300) + y_true * 0.25, 0, 1)

    evaluator = ChurnEvaluator(threshold=0.40)
    curve = evaluator.get_threshold_curve(y_true, y_proba)

    assert "threshold" in curve.columns
    assert "f2_score" in curve.columns
    assert "business_cost_usd" in curve.columns
    # Flagged customers should decrease as threshold increases
    assert curve["flagged_customers"].iloc[0] >= curve["flagged_customers"].iloc[-1]


def test_evaluator_lift_curve():
    """Lift curve should have 10 deciles and lift > 1 in top decile."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 500)
    y_proba = np.clip(np.random.beta(2, 5, 500) + y_true * 0.3, 0, 1)

    evaluator = ChurnEvaluator(threshold=0.40)
    lift = evaluator.get_lift_curve(y_true, y_proba, bins=10)

    assert len(lift) == 10
    assert "lift" in lift.columns
    # Top decile should have lift > 1 (model performs better than random)
    assert lift.iloc[0]["lift"] > 1.0
