# Project Handoff — Crypto Volatility Pipeline

**From:** Rahul Tejannavar  
**Date:** April 3, 2026  
**Project:** Real-Time BTC-USD Volatility Spike Detection  
**Repo:** https://github.com/rahultejannavar/crypto-volatility-pipeline

---

## 1. What This Project Does

This pipeline connects to Coinbase's live WebSocket feed, collects
BTC-USD ticker data, streams it through Kafka, computes rolling
window features, and uses an XGBoost model to detect short-term
volatility spikes in real time. It achieves 0.998 PR-AUC on
held-out test data.

---

## 2. Quick Start

### Prerequisites
- Docker Desktop installed and running
- Python 3.10+ with virtual environment
- Git

### First-Time Setup
```bash
# Clone the repo
git clone https://github.com/rahultejannavar/crypto-volatility-pipeline.git
cd crypto-volatility-pipeline

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate    # Mac/Linux
# .venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment file and add your keys (if needed)
cp .env.example .env
```

### Run the Full Pipeline
```bash
# Start all services (Kafka, MLflow, Ingestor, Featurizer)
docker compose -f docker/compose.yaml up -d

# Verify everything is running
docker compose -f docker/compose.yaml ps

# View MLflow dashboard
open http://localhost:5001
```

### Run Individual Components Manually
```bash
# Ingest data (15 minutes)
python scripts/ws_ingest.py --pair BTC-USD --minutes 15

# Replay raw data through featurizer
python scripts/replay.py --raw data/raw/*.ndjson --out data/processed/features.parquet

# Train models
python models/train.py

# Run inference (batch)
python models/infer.py --data data/processed/features.parquet --model xgboost

# Generate monitoring reports
python reports/evidently_report.py

# Shut down
docker compose -f docker/compose.yaml down
```

---

## 3. Repository Structure
crypto-volatility-pipeline/
├── config.yaml                  ← Central configuration
├── requirements.txt             ← Python dependencies
├── .env.example                 ← Environment variable template
│
├── docker/
│   ├── compose.yaml             ← Docker Compose (Kafka, MLflow, Ingestor, Featurizer)
│   ├── Dockerfile.ingestor      ← Container for WebSocket ingestor
│   └── Dockerfile.featurizer    ← Container for feature pipeline
│
├── scripts/
│   ├── ws_ingest.py             ← Coinbase WebSocket → Kafka + NDJSON
│   ├── kafka_consume_check.py   ← Kafka consumer sanity check
│   └── replay.py                ← Replay NDJSON files through featurizer
│
├── features/
│   └── featurizer.py            ← Feature engineering (live + replay modes)
│
├── models/
│   ├── train.py                 ← Train z-score, LogReg, XGBoost + MLflow logging
│   ├── infer.py                 ← Batch and single-record inference
│   └── artifacts/               ← Saved models, scaler, feature list
│
├── notebooks/
│   └── eda.ipynb                ← Exploratory data analysis
│
├── reports/
│   ├── evidently_report.py      ← Drift and performance monitoring
│   ├── evidently/               ← Generated HTML reports
│   ├── eval_*.png               ← Model evaluation plots
│   └── model_comparison.png     ← Side-by-side model comparison
│
├── docs/
│   ├── scoping_brief.pdf        ← Business case and prediction goal
│   ├── feature_spec.md          ← Feature definitions and selection rationale
│   ├── model_card_v1.md         ← Model documentation and limitations
│   ├── genai_appendix.md        ← AI usage documentation
│   └── mlflow_troubleshooting_log.md ← Infrastructure debugging log
│
├── data/
│   ├── raw/                     ← Raw NDJSON ticks (gitignored)
│   └── processed/               ← Feature parquet files (gitignored)
│
└── handoff/
└── HANDOFF.md               ← This file

---

## 4. Key Technical Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Streaming | Kafka (KRaft mode) | No ZooKeeper dependency, auto-topic creation |
| Data format | NDJSON (raw), Parquet (features) | NDJSON for streaming, Parquet for fast columnar queries |
| Feature window | 60-second sliding | Matches prediction horizon |
| Spike threshold | 85th percentile of future_vol | Yields ~15% prevalence, sufficient positive examples |
| Primary metric | PR-AUC | Handles class imbalance better than ROC-AUC |
| Selected model | XGBoost | PR-AUC 0.998 vs LogReg 0.538 vs Z-Score 0.505 |
| Experiment tracking | MLflow 2.x | Pinned to <3 due to 3.x breaking changes |
| Monitoring | Evidently (legacy API) | Data drift, target drift, model performance reports |

---

## 5. Model Summary

| Metric | XGBoost |
|--------|---------|
| PR-AUC | 0.998 |
| ROC-AUC | 0.999 |
| Precision | 0.959 |
| Recall | 0.998 |
| F1-Score | 0.978 |
| False Positives | 49 |
| False Negatives | 2 |

**Top features by importance:** return_std (0.29), volume_24h (0.15),
price_range_pct (0.15), spread_pct (0.12), midprice (0.11)

---

## 6. Known Issues and Limitations

1. **Volatility clustering inflates metrics** — The model detects
   ongoing spike regimes rather than predicting new onset. PR-AUC
   is partly driven by autocorrelation. See model card Section 8.1.

2. **MLflow artifact uploads fail** — The Docker MLflow server
   cannot receive artifacts from the local client due to path
   conflicts. Params and metrics log correctly. Artifacts are
   saved locally. See `docs/mlflow_troubleshooting_log.md`.

3. **Narrow training data** — All data from a ~26-hour window.
   Performance on different market conditions is unknown.

4. **Evidently API version** — Using `evidently.legacy` imports
   for v0.7.x compatibility. May need updating if Evidently
   is upgraded.

---

## 7. Maintenance Checklist

For ongoing operation of this pipeline:

- [ ] Monitor Evidently reports weekly for data drift
- [ ] Retrain if spike prevalence shifts beyond ±5 percentage points
- [ ] Recalibrate τ after significant new data collection
- [ ] Check MLflow dashboard for experiment comparison
- [ ] Review Docker container logs for connectivity issues
- [ ] Update MLflow version pin when artifact issue is resolved

---

## 8. Contact

**Author:** Rahul Tejannavar  
**Date:** April 3, 2026  
**Email:** rahultejannavar@cmu.edu  
**Course:** 94-879, Spring 2026, Prof. Anand S Rao