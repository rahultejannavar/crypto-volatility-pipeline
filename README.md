# Crypto Volatility Pipeline

Real-time BTC-USD volatility spike detection using streaming data,
machine learning, and production monitoring.

**Course:** 94-879 Fundamentals of Operationalizing AI, CMU Heinz College  
**Author:** Rahul Tejannavar  
**Date:** April 2026

---

## Overview

This pipeline connects to Coinbase's live WebSocket API, collects
BTC-USD ticker data, streams it through Apache Kafka, computes
rolling window features, and uses an XGBoost classifier to detect
short-term volatility spikes. The system includes experiment
tracking (MLflow), drift monitoring (Evidently), and full Docker
containerization.

**Prediction task:** Will 60-second forward volatility exceed
threshold τ? (Binary classification)

**Best model:** XGBoost — PR-AUC 0.998, Precision 0.959, Recall 0.998

---

## Architecture

```
Coinbase WebSocket
       │
       ▼
  ws_ingest.py ──► Kafka (ticks.raw) ──► featurizer.py ──► Kafka (ticks.features)
       │                                       │
       ▼                                       ▼
  data/raw/*.ndjson                    data/processed/features.parquet
                                               │
                                               ▼
                                          train.py ──► MLflow
                                               │
                                               ▼
                                          infer.py ──► Predictions
                                               │
                                               ▼
                                     evidently_report.py ──► Drift Reports
```

---

## Quick Start

### Prerequisites

- Docker Desktop
- Python 3.10+
- Git

### Setup

```bash
git clone https://github.com/rahultejannavar/crypto-volatility-pipeline.git
cd crypto-volatility-pipeline

python -m venv .venv
source .venv/bin/activate        # Mac/Linux
pip install -r requirements.txt

cp .env.example .env
```

### Run Everything with Docker

```bash
docker compose -f docker/compose.yaml up -d
docker compose -f docker/compose.yaml ps
```

This starts Kafka, MLflow, the ingestor, and the featurizer
automatically.

### Run Components Manually

```bash
# Ingest live data (15 minutes)
python scripts/ws_ingest.py --pair BTC-USD --minutes 15

# Replay raw data through feature pipeline
python scripts/replay.py --raw data/raw/*.ndjson --out data/processed/features.parquet

# Train models (z-score, LogReg, XGBoost)
python models/train.py

# Run batch inference
python models/infer.py --data data/processed/features.parquet --model xgboost

# Generate monitoring reports
python reports/evidently_report.py

# Shut down
docker compose -f docker/compose.yaml down
```

### View Dashboards

- **MLflow:** http://localhost:5001
- **Evidently reports:** Open `reports/evidently/*.html` in browser

---

## Project Structure

```
crypto-volatility-pipeline/
├── config.yaml                  # Central configuration
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
│
├── docker/
│   ├── compose.yaml             # Kafka, MLflow, Ingestor, Featurizer
│   ├── Dockerfile.ingestor      # WebSocket ingestor container
│   └── Dockerfile.featurizer    # Feature pipeline container
│
├── scripts/
│   ├── ws_ingest.py             # Coinbase WebSocket → Kafka + NDJSON
│   ├── kafka_consume_check.py   # Kafka consumer sanity check
│   └── replay.py                # Replay NDJSON through featurizer
│
├── features/
│   └── featurizer.py            # Feature engineering (live + replay)
│
├── models/
│   ├── train.py                 # Train and evaluate models
│   ├── infer.py                 # Batch and single-record inference
│   └── artifacts/               # Saved models and preprocessing
│
├── notebooks/
│   └── eda.ipynb                # Exploratory data analysis
│
├── reports/
│   ├── evidently_report.py      # Drift and performance monitoring
│   └── evidently/               # Generated HTML reports
│
├── docs/
│   ├── scoping_brief.pdf        # Business case
│   ├── feature_spec.md          # Feature definitions
│   ├── model_card_v1.md         # Model documentation
│   └── genai_appendix.md        # AI usage documentation
│
└── handoff/
    └── HANDOFF.md               # Handoff documentation
```

---

## Model Results

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|-------|--------|---------|-----------|--------|----|
| Z-Score Baseline | 0.505 | 0.766 | 0.565 | 0.564 | 0.564 |
| Logistic Regression | 0.538 | 0.813 | 0.335 | 0.714 | 0.456 |
| **XGBoost** | **0.998** | **0.999** | **0.959** | **0.998** | **0.978** |

---

## Features

Eight features computed over a 60-second sliding window:

| Feature | Description | Spike Correlation |
|---------|-------------|------------------|
| return_std | Realized volatility | 0.59 (strong) |
| price_range_pct | Normalized price range | 0.55 (strong) |
| return_mean | Directional drift | 0.35 (strong) |
| tick_count | Trading activity | 0.28 (moderate) |
| midprice | Current price level | 0.25 (moderate) |
| volume_24h | 24-hour volume | 0.23 (moderate) |
| return_skew | Return asymmetry | 0.21 (moderate) |
| spread_pct | Normalized bid-ask spread | 0.04 (weak) |

---

## Monitoring

Evidently reports detect three types of drift:

1. **Data drift** — Feature distribution changes across time
2. **Target drift** — Spike prevalence shifted from 9.2% to 20.8%
   between early and late data
3. **Model performance** — Classification metrics on current vs
   reference data

---

## Tools & Technologies

| Tool | Purpose |
|------|---------|
| Apache Kafka | Real-time stream processing |
| MLflow | Experiment tracking and model logging |
| Evidently | Data drift and model monitoring |
| Docker | Containerization and orchestration |
| XGBoost | Gradient-boosted tree classification |
| scikit-learn | Logistic Regression, evaluation metrics |
| Coinbase WebSocket | Live BTC-USD market data |

---

## Known Limitations

1. **Volatility clustering** — Model detects ongoing spikes, not
   new spike onset. High PR-AUC is partly driven by autocorrelation.
2. **Narrow training window** — All data from a ~26-hour period.
   Performance on different market conditions is unknown.
3. **MLflow artifact uploads** — Artifacts save locally due to
   Docker path conflicts. Params and metrics log to server.

See `docs/model_card_v1.md` for full limitations analysis.

---

## AI Usage

This project used Claude (Anthropic, 2026) as a development
assistant. Full documentation in `docs/genai_appendix.md`.

> Anthropic. (2026). Claude [Large language model]. https://claude.ai