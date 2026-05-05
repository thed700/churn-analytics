from setuptools import setup, find_packages

setup(
    name="churn-analytics",
    version="1.0.0",
    author="Akmal",
    description="End-to-end Customer Churn Intelligence System",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.5.0",
        "lightgbm>=4.5.0",
        "catboost>=1.2.0",
        "lifelines>=0.29.0",
        "optuna>=3.6.0",
        "shap>=0.45.0",
        "plotly>=5.22.0",
        "dash>=2.17.0",
        "dash-bootstrap-components>=1.6.0",
        "loguru>=0.7.0",
        "pyyaml>=6.0.0",
    ],
)
