"""
Data cleaning and preprocessing pipeline.

Handles type coercion, missing values, and target encoding
before feature engineering begins.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from loguru import logger

from src.utils.helpers import timer


class ChurnPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible preprocessing transformer.

    Steps:
    1. Fix TotalCharges (stored as string in raw data)
    2. Encode binary target
    3. Handle missing values
    4. Encode binary Yes/No columns
    """

    BINARY_COLS = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "Churn", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]

    def __init__(self):
        self.median_total_charges_ = None

    @timer
    def fit(self, df: pd.DataFrame, y=None) -> "ChurnPreprocessor":
        """Learn statistics from training data."""
        df_temp = df.copy()
        df_temp["TotalCharges"] = pd.to_numeric(df_temp["TotalCharges"], errors="coerce")
        self.median_total_charges_ = df_temp["TotalCharges"].median()
        logger.info(f"Preprocessor fitted. Median TotalCharges: {self.median_total_charges_:.2f}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned transformations."""
        df = df.copy()

        # Fix TotalCharges — raw data has spaces for new customers
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        n_missing = df["TotalCharges"].isna().sum()
        if n_missing > 0:
            logger.warning(f"Imputing {n_missing} missing TotalCharges with median ({self.median_total_charges_:.2f})")
            df["TotalCharges"] = df["TotalCharges"].fillna(self.median_total_charges_)

        # Encode binary target
        df["Churn"] = (df["Churn"] == "Yes").astype(int)

        # Encode Yes/No columns to 1/0
        yes_no_cols = [c for c in self.BINARY_COLS if c in df.columns and c != "Churn"]
        for col in yes_no_cols:
            if df[col].dtype == object:
                df[col] = df[col].map({"Yes": 1, "No": 1, "No phone service": 0, "No internet service": 0}).fillna(0).astype(int)

        # SeniorCitizen already 0/1
        logger.info(f"Preprocessing complete. Shape: {df.shape}")
        return df
