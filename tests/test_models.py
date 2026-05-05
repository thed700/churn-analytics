"""
Unit tests for evaluator and helper functions.
"""
import pytest
import numpy as np
import pandas as pd

from src.models.evaluator import ChurnEvaluator
from src.utils.helpers import business_cost_matrix, fbeta_score, get_high_risk_customers


def test_fbeta_perfect_model():
    """F2-score should be 1.0 for perfect predictions."""
    assert fbeta_score(1.0, 1.0, beta=2.0) == pytest.approx(1.0)


def test_fbeta_zero_recall():
    """F2-score should be 0.0 if recall is 0."""
    assert fbeta_score(1.0, 0.0, beta=2.0) == pytest.approx(0.0)


def test_business_cost_matrix_structure():
    """Cost matrix should return correct keys."""
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])
    result = business_cost_matrix(y_true, y_pred, cost_fn=500, cost_fp=50)

    assert "true_positives" in result
    assert "false_negatives" in result
    assert "total_business_cost_usd" in result
    assert "estimated_revenue_saved_usd" in result


def test_business_cost_fn_dominant():
    """Missing a churner (FN) should cost more than a false alarm (FP)."""
    y_true = np.array([1, 1, 0, 0])
    y_pred_all_positive = np.array([1, 1, 1, 1])   # 0 FN, 2 FP
    y_pred_all_negative = np.array([0, 0, 0, 0])   # 2 FN, 0 FP

    cost_fp = business_cost_matrix(y_true, y_pred_all_positive, 500, 50)
    cost_fn = business_cost_matrix(y_true, y_pred_all_negative, 500, 50)

    # 2 FN × $500 = $1000  >  2 FP × $50 = $100
    assert cost_fn["total_business_cost_usd"] > cost_fp["total_business_cost_usd"]


def test_evaluator_produces_result():
    """ChurnEvaluator should return EvaluationResult with valid fields."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 200)
    y_proba = np.clip(np.random.beta(2, 5, 200) + y_true * 0.3, 0, 1)

    evaluator = ChurnEvaluator(threshold=0.40)
    result = evaluator.evaluate(y_true, y_proba)

    assert 0 <= result.auc_roc <= 1
    assert 0 <= result.f2_score <= 1
    assert 0 <= result.precision <= 1
    assert 0 <= result.recall <= 1
    assert result.threshold_used == 0.40


def test_get_high_risk_customers():
    """High-risk filter should only return customers above threshold."""
    df = pd.DataFrame({
        "customerID": ["A", "B", "C", "D"],
        "churn_probability": [0.1, 0.5, 0.7, 0.3],
        "estimated_clv": [500, 1000, 800, 600],
    })
    result = get_high_risk_customers(df, threshold=0.40)
    assert len(result) == 2
    assert all(result["churn_probability"] >= 0.40)
    # Should be sorted by revenue_at_risk descending
    assert result.iloc[0]["churn_probability"] >= result.iloc[1]["churn_probability"] or \
           result.iloc[0]["revenue_at_risk"] >= result.iloc[1]["revenue_at_risk"]
