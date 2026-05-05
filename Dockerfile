FROM python:3.11-slim

LABEL maintainer="Akmal <thed700>"
LABEL description="Customer Churn Intelligence System"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .
RUN pip install -e . --no-deps

# Create necessary directories
RUN mkdir -p data/raw data/processed models logs reports/figures

# Default: run pipeline
CMD ["python", "-m", "src.main", "--config", "configs/config.yaml"]

EXPOSE 8050
