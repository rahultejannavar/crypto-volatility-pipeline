# Model Card — Crypto Volatility Spike Detection (v1)

**Project:** Real-Time BTC-USD Volatility Spike Detection  
**Course:** 94-879 Fundamentals of Operationalizing AI, Heinz College of Information Systems and Public Policy, CMU
**Author:** Rahul Tejannavar  
**Date:** April 3, 2026  
**Version:** 1.0

---

## 1. Model Overview

This model detects short-term volatility spikes in BTC-USD trading
by classifying whether 60-second forward volatility will exceed a
threshold τ. It is part of a real-time streaming pipeline that
ingests live Coinbase WebSocket data, computes rolling window
features, and generates binary spike/no-spike predictions.

**Task:** Binary classification  
**Target:** spike (1 if future_vol ≥ τ, else 0)  
**Primary Metric:** PR-AUC (Precision-Recall Area Under Curve)  
**Selected Model:** XGBoost  

---

## 2. Intended Use

**Primary use case:** Real-time alerting for BTC-USD volatility
spikes to support risk management and trading decisions.

**Intended users:** Traders, risk managers, or automated trading
systems that need early warning of elevated market volatility.

**Out-of-scope uses:**
- Price direction prediction (the model predicts volatility, not
  whether price goes up or down)
- Multi-asset volatility (trained only on BTC-USD)
- Long-horizon forecasting (prediction window is 60 seconds)

---

## 3. Training Data

**Source:** Coinbase Advanced Trade WebSocket API (ticker channel)  
**Asset:** BTC-USD  
**Collection period:** April 2–3, 2026 (multiple sessions)  
**Total rows:** 38,692 feature rows (after dropping NaN labels)  
**Midprice range:** ~$66,927 – $68,500  
**Train/test split:** 80/20 stratified random split  
- Train: 30,953 rows (15.0% spikes)  
- Test: 7,739 rows (15.0% spikes)

**Spike threshold:** τ = 0.00003164 (85th percentile of future_vol)  
**Spike prevalence:** 15.0% (5,804 positive examples)  
**Independent spike events:** 41 contiguous spike runs

---

## 4. Features

Eight features selected based on EDA correlation analysis and
distribution separation tests. See `docs/feature_spec.md` for
full specifications.

**Strong predictors (ρ > 0.3 with spike):**
- `return_std` (ρ = 0.59) — current realized volatility
- `price_range_pct` (ρ = 0.55) — normalized price range in window
- `return_mean` (ρ = 0.35) — directional drift magnitude

**Moderate predictors (0.15 < ρ < 0.3):**
- `tick_count` (ρ = 0.28) — trading activity proxy
- `midprice` (ρ = 0.25) — current price level
- `volume_24h` (ρ = 0.23) — market-wide activity

**Weak but included:**
- `return_skew` (ρ = 0.21) — return distribution asymmetry
- `spread_pct` (ρ = 0.04) — normalized bid-ask spread

**Dropped features:**
- `book_imbalance` — near-zero correlation (ρ = 0.004)
- `price_range` — redundant with `price_range_pct` (ρ = 1.00)
- `spread` — redundant with `spread_pct` (ρ = 1.00)

---

## 5. Model Comparison

Three models were evaluated. All logged to MLflow.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|-------|--------|---------|-----------|--------|----|
| Z-Score Baseline | 0.505 | 0.766 | 0.565 | 0.564 | 0.564 |
| Logistic Regression | 0.538 | 0.813 | 0.335 | 0.714 | 0.456 |
| **XGBoost** | **0.998** | **0.999** | **0.959** | **0.998** | **0.978** |

**Selected model:** XGBoost based on highest PR-AUC.

### XGBoost Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_estimators | 200 | Sufficient boosting rounds |
| max_depth | 4 | Shallow trees to prevent overfitting |
| learning_rate | 0.1 | Standard starting point |
| scale_pos_weight | 5.67 | Compensates for class imbalance |
| min_child_weight | 5 | Prevents overfitting on small leaves |
| subsample | 0.8 | Row sampling for regularization |
| colsample_bytree | 0.8 | Feature sampling for regularization |

### XGBoost Feature Importance (Gain)

| Feature | Importance |
|---------|-----------|
| return_std | 0.290 |
| volume_24h | 0.150 |
| price_range_pct | 0.148 |
| spread_pct | 0.116 |
| midprice | 0.110 |
| tick_count | 0.067 |
| return_skew | 0.063 |
| return_mean | 0.058 |

---

## 6. Evaluation Details

### Confusion Matrix (Test Set — 7,739 rows)

|  | Predicted No Spike | Predicted Spike |
|--|-------------------|----------------|
| **Actual No Spike** | 6,529 | 49 |
| **Actual Spike** | 2 | 1,159 |

- **False positives (unnecessary alerts):** 49
- **False negatives (missed spikes):** 2

### Why PR-AUC Over ROC-AUC

With 15% spike prevalence, ROC-AUC can be misleadingly high
because the true negative rate is inflated by the large majority
class. PR-AUC focuses on the positive class — when the model
says "spike," how often is it right, and how many actual spikes
does it catch. This aligns with the business use case where
missed spikes are costly.

---

## 7. Monitoring and Drift

Evidently reports were generated comparing the first and second
halves of the data (chronological split):

**Data drift findings:**
- Reference (early): 19,346 rows, 9.2% spikes
- Current (late): 19,346 rows, 20.8% spikes
- Significant target drift detected — the late data has more
  than double the spike rate of the early data
- This confirms different collection sessions captured different
  market conditions

**Implication:** If the model were trained only on early (calm)
data and deployed during the later (volatile) period, its
threshold τ would be miscalibrated. Regular retraining and
threshold recalibration are essential.

Reports available in `reports/evidently/`.

---

## 8. Limitations and Risks

### 8.1 Volatility Clustering and Autocorrelation

Future_vol shows autocorrelation > 0.83 at lag 100. Spike
windows have a 99.5% probability of being followed by another
spike window. The model primarily detects ongoing volatility
regimes rather than predicting new spike onset. Think of it
as detecting that it's currently raining, not forecasting
rain from a clear sky.

This means:
- A naive "predict same state as current" baseline performs well
- The high PR-AUC (0.998) is partly driven by volatility
  persistence rather than the model learning subtle precursors
- Performance on truly novel volatility events may be lower

### 8.2 Train/Test Leakage from Autocorrelation

Random train/test split places adjacent (nearly identical) ticks
into both sets. This inflates test metrics because the model
sees near-duplicate data points. A time-based split or
spike-block-level split would give more conservative estimates.

### 8.3 Limited Market Conditions

All data was collected during a ~26-hour window. The BTC-USD
midprice ranged from ~$66,927 to ~$68,500. Model performance
during flash crashes, exchange outages, or major news events
is unknown and likely degraded.

### 8.4 Training–Serving Skew

Features are computed identically in live and replay modes using
the same `compute_features()` function. However, tick arrival
timing differs — live mode has network jitter; replay processes
ticks instantaneously. This may cause subtle differences in
window composition.

---

## 9. Recommendations

1. **Collect more diverse data** across different market conditions,
   times of day, and volatility regimes before production deployment.
2. **Implement time-based train/test splits** to get more honest
   evaluation metrics.
3. **Monitor for target drift** using Evidently on a daily basis —
   if spike prevalence shifts beyond ±5 percentage points, trigger
   model retraining.
4. **Recalibrate τ periodically** — the 85th percentile threshold
   should be recomputed as more data accumulates.
5. **Add regime-onset features** (e.g., rate of change of return_std)
   to improve prediction of new spike events rather than just
   detecting ongoing ones.

---

## 10. Ethical Considerations

- The model is used for market monitoring, not autonomous trading.
  Predictions should inform human decisions, not replace them.
- No personally identifiable information is used.
- Model performance may vary across market conditions, creating
  risk if used without monitoring in production.