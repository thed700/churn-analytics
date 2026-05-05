"""
Utility functions: config loading, timing, cost matrix computation.
"""
import time
import functools
from pathlib import Path
from typing import Any

import yaml
import numpy as np
import pandas as pd
from loguru import logger


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Config loaded from {config_path}")
    return config


def timer(func):
    """Decorator: logs execution time of any function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


def business_cost_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float = 500.0,
    cost_fp: float = 50.0,
) -> dict:
    """
    Compute dollar-denominated confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_pred : np.ndarray
        Predicted binary labels (at chosen threshold).
    cost_fn : float
        Cost of a False Negative (missed churner) — avg customer CLV.
    cost_fp : float
        Cost of a False Positive (unnecessary retention offer).

    Returns
    -------
    dict with TP, TN, FP, FN counts and total business cost.
    """
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    total_cost = (fn * cost_fn) + (fp * cost_fp)
    revenue_saved = tp * cost_fn

    return {
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "total_business_cost_usd": round(total_cost, 2),
        "estimated_revenue_saved_usd": round(revenue_saved, 2),
    }


def fbeta_score(precision: float, recall: float, beta: float = 2.0) -> float:
    """F-beta score: recall weighted beta^2 times more than precision."""
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)


def get_high_risk_customers(
    df: pd.DataFrame,
    churn_proba_col: str = "churn_probability",
    threshold: float = 0.40,
    clv_col: str = "estimated_clv",
) -> pd.DataFrame:
    """
    Filter and rank high-risk customers for the retention team.

    Returns DataFrame sorted by revenue at risk (descending).
    """
    high_risk = df[df[churn_proba_col] >= threshold].copy()
    high_risk["revenue_at_risk"] = high_risk[churn_proba_col] * high_risk[clv_col]
    return high_risk.sort_values("revenue_at_risk", ascending=False).reset_index(drop=True)
