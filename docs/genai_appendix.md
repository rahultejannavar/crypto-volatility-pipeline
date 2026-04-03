# Generative AI Usage Appendix

**Project:** Detecting Crypto Volatility in Real-Time  
**Course:** 94-879 Fundamentals of Operationalizing AI, CMU Heinz College  
**Author:** Rahul Tejannavar 
**Date:** April 3, 2026  

## AI Tool Used

Anthropic. (2026). Claude [Large language model]. https://claude.ai

**Model:** Claude (Opus 4.6)  
**Interaction dates:** April 2–3, 2026  
**Full conversation transcripts:** Available upon request. Key exchanges
are summarized below.

---

## Summary of AI Usage by Component

### 1. Streaming & Infrastructure (Milestone 1)

**Where used:** Design of `ws_ingest.py`, `docker/compose.yaml`,
Kafka configuration, WebSocket reconnection logic.

**How used:** Discussed architecture decisions for the streaming
pipeline — KRaft vs ZooKeeper for Kafka, WebSocket heartbeat
handling, NDJSON backup strategy. Claude provided code structure
and explanations for each design choice.

**Verification:** Tested live against Coinbase WebSocket API.
Confirmed tick ingestion rates (~250 ticks/min), Kafka consumer
checks via `kafka_consume_check.py`, and Docker container
stability across multiple collection sessions.

---

### 2. Feature Engineering (Milestone 2)

**Where used:** Design of `features/featurizer.py`, feature
selection rationale, EDA notebook cells.

**How used:** Claude helped design the 14-feature set based on
market microstructure principles. Each EDA cell was provided
one at a time with explanations of purpose and expected findings.
Statistical interpretations (correlation analysis, autocorrelation,
run-length analysis) were discussed collaboratively.

**Key prompts:**
- "What features should we compute from tick-level crypto data?"
- "Help me build EDA cells to choose the spike threshold τ"
- "Explain what the autocorrelation and transition matrix results mean"

**Verification:** All feature computations verified by inspecting
`describe()` output against known BTC-USD market properties.
Correlation values cross-checked with distribution plots.
Threshold τ = 85th percentile validated by checking spike
prevalence (~15%) and positive example count (~5,800).

---

### 3. Modeling & Evaluation (Milestone 3)

**Where used:** `models/train.py`, `models/infer.py`, MLflow
integration, model evaluation logic.

**How used:** Claude provided `train.py` block by block, with
each block explained before coding. Three models were designed
collaboratively: z-score baseline, Logistic Regression, and
XGBoost. Evaluation function with PR-AUC, confusion matrix,
and PR curve plotting was built with explanations of why
PR-AUC was chosen over ROC-AUC for imbalanced classification.

**Key prompts:**
- "Build train.py block by block, explaining each block"
- "Why is XGBoost's PR-AUC so high? Is it legitimate?"
- "Build infer.py with both batch and single-record modes"

**Verification:**
- Z-score baseline PR-AUC (0.50) confirmed as reasonable for
  single-feature rule
- LogReg PR-AUC (0.54) confirmed marginal improvement, consistent
  with weak linear separability in EDA
- XGBoost PR-AUC (0.998) flagged as potentially inflated due to
  autocorrelation — documented as limitation in model card
- Inference pipeline tested end-to-end: batch scoring produced
  6,046 predicted spikes vs 5,804 actual (consistent with
  training confusion matrix)

---

### 4. Monitoring & Drift Detection

**Where used:** `reports/evidently_report.py`

**How used:** Claude designed the Evidently monitoring script with
three reports (data drift, target drift, model performance).
Significant debugging was required due to Evidently API changes
in version 0.7.x — Claude helped navigate the `evidently.legacy`
module imports.

**Key prompts:**
- "Build the Evidently drift report block by block"
- "The import is failing — help debug the Evidently API"

**Verification:** All three reports generated successfully. Key
finding validated: target drift from 9.2% to 20.8% spike
prevalence between early and late data confirmed by manual
inspection of the time series plot.

---

### 5. Docker Packaging

**Where used:** `docker/Dockerfile.featurizer`, updated
`docker/compose.yaml`

**How used:** Claude designed the multi-service Docker Compose
configuration with health checks, dependency ordering, and
environment variable passthrough for Kafka broker addresses.

**Verification:** Both images built successfully
(`docker compose build`). Service dependency chain validated:
Kafka → Ingestor → Featurizer.

---

### 6. Documentation

**Where used:** `docs/feature_spec.md`, `docs/model_card_v1.md`,
`README.md`, this appendix.

**How used:** Claude drafted document templates based on findings
from EDA, training, and monitoring. All content was reviewed
and validated against actual pipeline outputs before inclusion.

---

### 7. Troubleshooting

**Where used:** MLflow Docker server configuration

**How used:** Encountered five cascading errors when connecting
the Python MLflow client to the Docker MLflow server. Claude
helped diagnose each issue: MLflow 3.x security middleware
(403 errors), database migration incompatibility, SQLite path
issues, and artifact storage path conflicts.

**Documentation:** Full troubleshooting log saved in
`docs/mlflow_troubleshooting_log.md`.

---

## What I Verified Independently

- All Coinbase WebSocket data validated against live market prices
- Feature computation spot-checked against manual calculations
- Model metrics cross-verified between training output and MLflow UI
- Evidently reports opened and inspected in browser
- Docker images built and service dependencies confirmed
- All code executed on my local machine; no pre-built solutions used

## What I Learned Through AI Collaboration

- The importance of volatility clustering and its implications for
  model evaluation honesty
- Why PR-AUC is preferred over ROC-AUC for imbalanced classification
- How train-serve skew manifests (the scaler bug in `infer.py` where
  XGBoost received scaled data and predicted zero spikes)
- Real-world dependency management challenges (MLflow 2.x vs 3.x
  breaking changes)
- The difference between detecting ongoing regimes vs predicting
  new events