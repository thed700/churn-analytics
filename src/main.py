"""
Main pipeline orchestrator for the Customer Churn Intelligence System.

Run with:
    python -m src.main

Or with custom config:
    python -m src.main --config configs/config.yaml
"""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from loguru import logger

from src.utils.logger import setup_logger
from src.utils.helpers import load_config, get_high_risk_customers
from src.data.loader import DataLoader
from src.data.preprocessor import ChurnPreprocessor
from src.features.engineer import ChurnFeatureEngineer
from src.models.churn_model import LGBMChurnModel, CatBoostChurnModel
from src.models.evaluator import ChurnEvaluator


def run_pipeline(config_path: str = "configs/config.yaml") -> dict:
    """
    Execute the full churn prediction pipeline end-to-end.

    Steps:
    1. Load & validate data
    2. Preprocess (type fix, target encode)
    3. Feature engineering (5 advanced features)
    4. Train-test split
    5. Train LightGBM (5-fold CV)
    6. Train CatBoost (challenger)
    7. Business-aware evaluation
    8. Generate high-risk customer list
    9. Save artifacts

    Returns
    -------
    dict with eval results, model objects, and predictions DataFrame.
    """
    setup_logger()
    cfg = load_config(config_path)

    logger.info("=" * 60)
    logger.info("CUSTOMER CHURN INTELLIGENCE PIPELINE")
    logger.info("=" * 60)

    # ── Step 1: Load ────────────────────────────────────────────────
    loader = DataLoader(cfg["paths"]["raw_data"])
    df_raw = loader.load()

    # ── Step 2: Preprocess ──────────────────────────────────────────
    preprocessor = ChurnPreprocessor()
    preprocessor.fit(df_raw)
    df_clean = preprocessor.transform(df_raw)

    # ── Step 3: Feature engineering ─────────────────────────────────
    engineer = ChurnFeatureEngineer()
    engineer.fit(df_clean)
    df_features = engineer.transform(df_clean)

    # ── Step 4: Prepare features & split ────────────────────────────
    target = cfg["data"]["target_col"]
    drop_cols = [cfg["data"]["customer_id_col"], target, "contract_months"]
    categorical_cols = ["Contract", "PaymentMethod", "InternetService", "gender"]

    feature_cols = [c for c in df_features.columns if c not in drop_cols
                    and c in df_features.columns]

    X = df_features[feature_cols].copy()
    y = df_features[target]

    # Encode remaining categoricals for LightGBM
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )
    logger.info(f"Split: {len(X_train):,} train | {len(X_test):,} test | "
                f"Churn rate: train={y_train.mean():.2%} | test={y_test.mean():.2%}")

    # ── Step 5: Train LightGBM ───────────────────────────────────────
    lgbm_model = LGBMChurnModel(cfg["model"], threshold=cfg["evaluation"]["threshold"])
    lgbm_model.fit(X_train, y_train)

    # ── Step 6: Train CatBoost (challenger) ─────────────────────────
    X_train_cb = X_train.copy()
    X_test_cb = X_test.copy()
    for col in categorical_cols:
        if col in X_train_cb.columns:
            X_train_cb[col] = X_train_cb[col].astype(str)
            X_test_cb[col] = X_test_cb[col].astype(str)

    cb_model = CatBoostChurnModel(cfg["model"], threshold=cfg["evaluation"]["threshold"])
    cat_idx = [i for i, c in enumerate(X_train_cb.columns) if c in categorical_cols]
    cb_model.fit(X_train_cb, y_train, categorical_features=cat_idx)

    # ── Step 7: Evaluate ─────────────────────────────────────────────
    evaluator = ChurnEvaluator(
        threshold=cfg["evaluation"]["threshold"],
        cost_fn=cfg["evaluation"]["cost_fn"],
        cost_fp=cfg["evaluation"]["cost_fp"],
        beta=cfg["evaluation"]["beta"],
    )

    lgbm_proba = lgbm_model.predict_proba(X_test)
    eval_result = evaluator.evaluate(y_test.values, lgbm_proba)

    # Threshold curve for dashboard
    threshold_df = evaluator.get_threshold_curve(y_test.values, lgbm_proba)

    # ── Step 8: High-risk customer list ─────────────────────────────
    test_customers = df_features.loc[X_test.index, [cfg["data"]["customer_id_col"], "Contract",
                                                      "MonthlyCharges", "tenure"]].copy()
    test_customers["churn_probability"] = lgbm_proba
    test_customers["estimated_clv"] = test_customers["MonthlyCharges"] * (test_customers["tenure"] + 6)
    test_customers["revenue_at_risk"] = test_customers["churn_probability"] * test_customers["estimated_clv"]

    high_risk_df = get_high_risk_customers(
        test_customers,
        churn_proba_col="churn_probability",
        threshold=cfg["evaluation"]["threshold"],
        clv_col="estimated_clv",
    )

    logger.info(f"\nTop 5 high-risk customers:\n{high_risk_df[['customerID','churn_probability','revenue_at_risk']].head().to_string()}")

    # ── Step 9: Save artifacts ───────────────────────────────────────
    Path(cfg["paths"]["processed_data"]).parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(cfg["paths"]["processed_data"], index=False)
    lgbm_model.save("models/lgbm_churn_final.pkl")
    high_risk_df.to_csv("reports/high_risk_customers.csv", index=False)
    threshold_df.to_csv("reports/threshold_curve.csv", index=False)

    importance_df = lgbm_model.get_feature_importance()
    importance_df.to_csv("reports/feature_importance.csv", index=False)

    logger.info("\nAll artifacts saved.")
    logger.info("Run the dashboard with: python -m src.visualization.dashboard")

    return {
        "eval_result": eval_result,
        "lgbm_model": lgbm_model,
        "cb_model": cb_model,
        "predictions": test_customers,
        "high_risk": high_risk_df,
        "threshold_df": threshold_df,
        "importance_df": importance_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Churn Intelligence Pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    results = run_pipeline(args.config)
    logger.info("Pipeline complete.")
