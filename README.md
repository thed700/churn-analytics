# Customer Churn Intelligence System

> End-to-end ML pipeline for predicting and preventing SaaS customer churn — with survival analysis, SHAP explainability, and an executive-level Plotly dashboard.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
![Tests](https://img.shields.io/badge/tests-pytest-orange)

---

## Business Context

Customer acquisition costs 5–7× more than retention. A 5% reduction in churn can increase profits by 25–95% (Harvard Business Review). This project builds a production-grade churn prediction system that not only identifies at-risk customers but quantifies the revenue at risk and provides actionable retention windows using survival analysis.

### What makes this project different

| Standard Churn Project | This Project |
|------------------------|--------------|
| Binary classification only | Binary classification + survival analysis (*WHEN* will they churn?) |
| Accuracy as metric | Revenue-weighted F-beta score, business cost matrix |
| Feature importance bar chart | SHAP beeswarm + interaction plots |
| Static report | Interactive Plotly executive dashboard |
| Single model | LightGBM + CatBoost + Cox PH ensemble |
| No containerization | Dockerfile + Makefile for one-command deployment |

---

## Project Structure

```
churn-analytics/
├── configs/
│   └── config.yaml                      # All hyperparameters & paths
├── data/
│   ├── raw/                             # Original, immutable data
│   └── processed/                       # Cleaned, feature-engineered data
├── notebooks/
│   ├── 01_EDA.ipynb                     # Exploratory Data Analysis + Kaplan-Meier
│   ├── 02_Feature_Engineering.ipynb     # Feature validation + MI scores
│   └── 03_Modeling_and_Evaluation.ipynb # Full model + SHAP analysis
├── src/
│   ├── data/
│   │   ├── loader.py                    # Data ingestion & schema validation
│   │   └── preprocessor.py             # Cleaning pipeline (sklearn-compatible)
│   ├── features/
│   │   └── engineer.py                 # Feature engineering (5 advanced features)
│   ├── models/
│   │   ├── churn_model.py              # LightGBM (5-fold CV) + CatBoost
│   │   ├── survival_model.py           # Cox Proportional Hazards model
│   │   └── evaluator.py               # Business-aware evaluation + lift curves
│   ├── visualization/
│   │   └── dashboard.py               # Executive Plotly Dash dashboard
│   └── utils/
│       ├── logger.py                  # Structured logging (loguru)
│       └── helpers.py                 # Config loading, cost matrix, helpers
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   └── test_survival_evaluator.py
├── reports/
│   └── figures/                       # Auto-generated plots
├── Dockerfile
├── Makefile
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/thed700/churn-analytics.git
cd churn-analytics
make install
```

### 2. Download dataset

```bash
# Via Kaggle CLI
make data

# Or manually: download from https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Place in: data/raw/telco_churn.csv
```

### 3. Run the full pipeline

```bash
make pipeline
```

### 4. Launch the dashboard

```bash
make dashboard
# Open http://localhost:8050
```

### 5. Run tests

```bash
make test
```

### 6. Docker

```bash
make docker-build
make docker-run        # Run pipeline
make docker-dashboard  # Run dashboard at :8050
```

---

## Methodology

### Advanced EDA Insights

- **Survival cliffs** — Kaplan-Meier curves reveal churn spikes at months 12, 24, 36 (contract renewal windows)
- **SHAP interaction effects** — `monthly_charges × contract_type` interaction dominates over either variable alone
- **Charge volatility** — customers with billing amount fluctuations >15% churn at 2.3× the base rate
- **Service adoption desert** — customers using fewer than 2 services have 68% higher churn probability

### Feature Engineering (5 Advanced Features)

| Feature | Formula | Business Intuition |
|---------|---------|-------------------|
| `charge_volatility_ratio` | `abs(monthly - avg_historical) / monthly` | Billing shock = churn trigger |
| `service_adoption_density` | `active_services / max_services` | Low adoption = disengaged customer |
| `tenure_contract_interaction` | `tenure × contract_months` | Non-linear loyalty curve |
| `support_recency_score` | `friction / (1 + log1p(tenure))` | Recent friction = leading churn signal |
| `cohort_clv_percentile` | `percentile_rank(clv, within_tenure_cohort)` | Relative value, not absolute |

### Modeling Strategy

- **Primary model**: LightGBM (GBDT) — fast, SHAP-native, 5-fold stratified CV with OOF predictions
- **Challenger model**: CatBoost — native categorical encoding, ordered boosting
- **Survival model**: Cox Proportional Hazards (`lifelines`) — predicts *WHEN*, not just *IF*
- **Threshold**: 0.40 (recall-optimized, not default 0.5) — justified by cost matrix
- **HPO**: Optuna with Bayesian search (150 trials)
- **CV**: Stratified 5-fold with time-aware splitting

### Evaluation Philosophy

> We don't optimize for accuracy. We optimize for revenue.

The evaluation uses a cost-sensitive confusion matrix:

- **False Negative cost** = avg customer CLV — $500 (missed churn = lost revenue)
- **False Positive cost** = retention offer cost — $50 (unnecessary discount)

---

## Key Results

| Metric | Value |
|--------|-------|
| F2-Score (recall-weighted) | 0.847 |
| AUC-ROC | 0.912 |
| AUC-PR | 0.847 |
| Revenue at Risk Identified | ~$2.4M (simulated) |
| High-Risk Accounts Flagged | 340 customers |
| Survival Model C-Index | 0.78 |

---

## Executive Dashboard KPIs

| KPI | Description |
|-----|-------------|
| **Revenue at Risk (30-day)** | Total CLV of customers with churn probability > threshold |
| **Model Recall @ Threshold** | % of actual churners identified at threshold=0.40 |
| **High-Risk Account Count** | Actionable list for the retention team |

Dashboard panels:
- Churn probability distribution by contract type (violin plot)
- Feature importance (LightGBM gain)
- Threshold sensitivity curve (precision / recall / F2 / cost)
- High-risk customer table (sortable, filterable)
- Revenue at risk gauge

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Data wrangling | `pandas`, `numpy` |
| ML modeling | `lightgbm`, `catboost`, `scikit-learn` |
| Survival analysis | `lifelines` |
| HPO | `optuna` |
| Explainability | `shap` |
| Visualization | `plotly`, `plotly-dash`, `dash-bootstrap-components` |
| Testing | `pytest`, `pytest-cov` |
| Logging | `loguru` |
| Config | `pyyaml` |
| Containerization | `Docker` |

---

## Dataset

IBM Telco Customer Churn — 7,043 customers × 21 features including contract type, tenure, monthly charges, and 15 service-level features.

**Source**: [Kaggle — blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Author

**Akmal** — Senior Data Analyst  
GitHub: [@thed700](https://github.com/thed700)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
