"""
Data ingestion and validation module.

Handles loading raw Telco Churn CSV and performing
schema validation before any processing begins.
"""
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

from src.utils.helpers import timer


EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


class DataLoader:
    """
    Loads and validates the Telco Churn dataset.

    Parameters
    ----------
    data_path : str
        Path to raw CSV file.
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    @timer
    def load(self) -> pd.DataFrame:
        """Load CSV and perform basic validation."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}.\n"
                "Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
                "Place in: data/raw/telco_churn.csv"
            )

        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

        self._validate_schema(df)
        return df

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Check that all expected columns are present."""
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")

        logger.info(f"Schema validation passed. Target distribution:\n{df['Churn'].value_counts(normalize=True).round(3).to_dict()}")
