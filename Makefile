.PHONY: help install install-dev data pipeline dashboard test lint format clean docker-build docker-run

PYTHON := python
PIP := pip
PROJECT := churn-analytics

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	$(PIP) install -r requirements.txt

install-dev:  ## Install all dependencies including dev tools
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

data:  ## Download Telco Churn dataset via Kaggle CLI
	@echo "Downloading Telco Churn dataset..."
	kaggle datasets download -d blastchar/telco-customer-churn -p data/raw/ --unzip
	@echo "Dataset ready at data/raw/"

pipeline:  ## Run the full ML pipeline end-to-end
	$(PYTHON) -m src.main --config configs/config.yaml

dashboard:  ## Launch the Plotly Dash executive dashboard
	$(PYTHON) -m src.visualization.dashboard

test:  ## Run test suite with coverage report
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html:reports/coverage

lint:  ## Run flake8 linter
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503

format:  ## Auto-format code with Black
	black src/ tests/ --line-length=100

clean:  ## Remove generated artifacts, cache, logs
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov reports/coverage
	rm -rf models/*.pkl logs/*.log
	@echo "Cleaned."

docker-build:  ## Build Docker image
	docker build -t $(PROJECT):latest .

docker-run:  ## Run pipeline in Docker container
	docker run --rm -v $(PWD)/data:/app/data -v $(PWD)/models:/app/models $(PROJECT):latest

docker-dashboard:  ## Run dashboard in Docker (http://localhost:8050)
	docker run --rm -p 8050:8050 -v $(PWD)/data:/app/data -v $(PWD)/models:/app/models \
		$(PROJECT):latest python -m src.visualization.dashboard
