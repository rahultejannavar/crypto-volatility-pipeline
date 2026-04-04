# Feature Specification — Crypto Volatility Pipeline

**Project:** Detecting Crypto Volatility in Real-Time  
**Course:** 94-879 Fundamentals of Operationalizing AI, CMU Heinz College  
**Author:** Rahul Tejannavar  
**Last Updated:** April 2, 2026  

---

## 1. Overview

This document specifies the features used in the crypto volatility spike
detection pipeline. The pipeline ingests live BTC-USD ticker data from
Coinbase's Advanced Trade WebSocket API, computes rolling window features,
and uses them to predict whether 60-second forward volatility will exceed
a threshold τ (binary classification).

**Data source:** Coinbase Advanced Trade WebSocket (`ticker` channel)  
**Asset:** BTC-USD  
**Sliding window:** 60 seconds  
**Prediction horizon:** 60 seconds forward  
**Primary metric:** PR-AUC (Precision-Recall Area Under Curve)

## 2. Raw Input Fields

Each tick received from the Coinbase WebSocket contains the following
fields used by the featurizer:

| Field | Type | Description |
|-------|------|-------------|
| `price` | string → float | Last trade price (BTC-USD) |
| `best_bid` | string → float | Highest current buy order |
| `best_ask` | string → float | Lowest current sell order |
| `best_bid_quantity` | string → float | Size at best bid |
| `best_ask_quantity` | string → float | Size at best ask |
| `volume_24_h` | string → float | Rolling 24-hour trade volume |
| `timestamp` | ISO 8601 string | Coinbase server timestamp |

All numeric fields arrive as strings and are cast to float during parsing
in `features/featurizer.py::parse_tick()`. A `received_at` timestamp is
appended locally to measure ingestion latency.

## 3. Computed Features

Features are computed over a 60-second sliding window of ticks. The window
advances with each new tick (not fixed intervals), so feature rows are
tick-aligned, not time-aligned.

### 3.1 Price-Based Features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `midprice` | (best_bid + best_ask) / 2 | More stable than last trade price; standard microstructure measure |
| `spread` | best_ask − best_bid | Measures market tightness; wider spreads can indicate uncertainty |
| `spread_pct` | spread / midprice | Normalized spread for cross-session comparability |
| `price_range` | max(midprice) − min(midprice) in window | Absolute price movement within the window |
| `price_range_pct` | price_range / midprice | Normalized price range; strong spike correlate (ρ = 0.55) |

### 3.2 Return-Based Features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `return_mean` | mean of tick-to-tick midprice returns in window | Captures directional drift; spike correlation ρ = 0.35 |
| `return_std` | std of tick-to-tick midprice returns in window | Current realized volatility; strongest predictor (ρ = 0.59) |
| `return_skew` | skewness of returns in window | Detects asymmetric return distributions; weak signal (ρ = 0.21) |

### 3.3 Microstructure Features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `book_imbalance` | (bid_qty − ask_qty) / (bid_qty + ask_qty) | Order book pressure indicator; range [−1, +1]. Near-zero spike correlation (ρ = 0.004) — included for completeness |
| `tick_count` | number of ticks in window | Proxy for trading activity; moderate signal (ρ = 0.28) |
| `volume_24h` | 24-hour volume (from Coinbase) | Market-wide activity level; moderate signal (ρ = 0.23) |

### 3.4 Label

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `future_vol` | std of midprice returns over NEXT 60 seconds | Continuous target; converted to binary spike label |
| `spike` | 1 if future_vol ≥ τ, else 0 | Binary classification target |

## 4. Threshold Selection (τ)

The spike threshold τ was chosen based on percentile analysis of `future_vol`
across the full dataset.

| Percentile | future_vol value |
|------------|-----------------|
| 50th | 0.00001979 |
| 75th | 0.00002684 |
| 80th | 0.00002828 |
| 85th | 0.00003321 |
| 90th | 0.00003813 |
| 95th | 0.00004381 |

**Selected threshold:** τ = 0.00003321 (85th percentile)

**Rationale:**
- Yields ~15% spike prevalence, providing ~1,474 positive examples in the
  initial dataset — sufficient for training and evaluation.
- The 90th percentile was considered but produces only ~980 positives,
  reducing test-set reliability.
- The 75th percentile labels too many windows as spikes (25%), diluting the
  definition of "elevated" volatility.

> **Note:** τ will be recomputed after additional data collection. If the
> 85th percentile shifts materially, the updated value will be documented here.

## 5. Feature Selection Notes

### 5.1 Recommended Model Features

Based on EDA (correlation analysis, distribution separation, and
autocorrelation analysis), the following features are recommended for
modeling:

**Strong signal (ρ > 0.3 with spike):**
- `return_std` — strongest predictor; captures volatility persistence
- `price_range_pct` — normalized price movement; highly correlated with spikes
- `return_mean` — directional drift; signal is in magnitude (nonlinear)

**Moderate signal (0.15 < ρ < 0.3):**
- `tick_count` — trading activity proxy
- `midprice` — moderate correlation, but may reflect confounding trends
- `volume_24h` — market-wide activity

**Weak signal (ρ < 0.15):**
- `return_skew` — modest distributional signal; may help tree-based models
- `spread` / `spread_pct` — near-zero correlation; BTC-USD spread is almost
  always $0.01 in this dataset
- `book_imbalance` — ρ = 0.004; effectively noise

### 5.2 Multicollinearity

Two feature pairs are perfectly correlated (ρ = 1.00):
- `price_range` ↔ `price_range_pct`
- `spread` ↔ `spread_pct`

**Decision:** Drop `price_range` and `spread` (keep the normalized `_pct`
versions for cross-session stability).

### 5.3 Dropped Features

| Feature | Reason |
|---------|--------|
| `price_range` | Redundant with `price_range_pct` (ρ = 1.00) |
| `spread` | Redundant with `spread_pct` (ρ = 1.00) |
| `book_imbalance` | Near-zero predictive signal (ρ = 0.004) |

## 6. Known Limitations and Risks

### 6.1 Volatility Clustering / Autocorrelation

`future_vol` shows autocorrelation > 0.83 at lag 100. Spike windows have
a 99.5% probability of being followed by another spike window. This means:
- Models partly rely on volatility persistence (detecting ongoing regimes)
  rather than predicting regime *onset*.
- A naive "predict same state as current" baseline will perform well.
- Model metrics may overstate ability to predict genuinely new spike events.

### 6.2 Limited Independent Spike Events

The initial dataset contains only 8 independent spike runs (contiguous blocks
of spike=1). The longest run is 1,187 ticks; the shortest is 3 ticks.
Random train/test splits may yield unstable PR-AUC depending on which
events land in the test set.

**Mitigation:** Additional data collection sessions are planned to increase
the number of independent spike events.

### 6.3 Narrow Market Conditions

All data was collected during a single multi-hour window. The BTC-USD
midprice ranged from ~$66,927 to ~$68,500 — a relatively calm period.
Model performance during flash crashes, exchange outages, or high-news
periods is unknown.

### 6.4 Training–Serving Skew Risk

Features are computed identically in live mode (Kafka consumer) and replay
mode (NDJSON files) using the same `compute_features()` function. However,
tick arrival timing differs — live mode has network jitter; replay mode
processes ticks instantaneously. This may cause subtle differences in
window composition.
