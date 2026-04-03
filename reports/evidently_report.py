"""
evidently_report.py — Data Drift & Model Performance Monitoring
Generates Evidently reports comparing feature distributions across
time windows and evaluating model prediction quality.

This is the Value Stewardship component (L9) of the pipeline —
monitoring for data drift that could degrade model performance
over time.

Usage:
    python reports/evidently_report.py
"""

import os
import sys
import pandas as pd
import numpy as np

from evidently.legacy.report import Report
from evidently.legacy.metric_preset import (
    DataDriftPreset,
    ClassificationPreset,
)
from evidently.legacy.utils.data_preprocessing import ColumnMapping

sys.path.insert(0, ".")

# ── Data Preparation ────────────────────────────────────────────
# WHY: Evidently compares two datasets — a "reference" dataset and
# a "current" dataset. The idea is simple: the reference is what
# the model was trained on (or what "normal" looks like), and the
# current is new incoming data. If the distributions differ
# significantly, that's drift — and it means the model might be
# making predictions on data it wasn't designed for.
#
# For our assignment, we don't have a true production vs training
# split (the model hasn't been deployed yet). So we simulate drift
# detection by splitting our data chronologically:
#   - Reference = first half of the data (earlier collection sessions)
#   - Current = second half (later collection sessions)
#
# This is realistic — in production, you'd compare "last week's
# data" against "this week's data" to catch distribution shifts.
# If BTC-USD's behavior changed between our collection sessions
# (different time of day, different market conditions), Evidently
# will flag it.

def load_and_split_data(parquet_path="data/processed/features.parquet"):
    """Load features and split chronologically into reference and current."""

    print(f"[DATA] Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=["future_vol"])

    # Sort by timestamp to ensure chronological split
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"[DATA] {len(df):,} rows loaded")
    print(f"[DATA] Time range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Compute spike label
    tau = np.percentile(df["future_vol"], 85)
    df["spike"] = (df["future_vol"] >= tau).astype(int)
    print(f"[DATA] τ = {tau:.8f}, spike prevalence = {df['spike'].mean()*100:.1f}%")

    # Chronological 50/50 split
    midpoint = len(df) // 2
    reference = df.iloc[:midpoint].copy()
    current = df.iloc[midpoint:].copy()

    print(f"[DATA] Reference (early): {len(reference):,} rows, "
          f"spikes = {reference['spike'].mean()*100:.1f}%")
    print(f"[DATA] Current (late):    {len(current):,} rows, "
          f"spikes = {current['spike'].mean()*100:.1f}%")

    return df, reference, current

# ── Data Drift Report ───────────────────────────────────────────
# WHY: Data drift means the input feature distributions have changed
# between reference and current data. This is the most common reason
# ML models degrade in production — the model learned patterns from
# one distribution but is now seeing a different one.
#
# Evidently runs statistical tests on each feature:
#   - For numerical features: Kolmogorov-Smirnov test (compares
#     whether two samples come from the same distribution)
#   - It flags a feature as "drifted" if the p-value falls below
#     a threshold (default: 0.05)
#
# For our crypto pipeline, drift could happen because:
#   - Different times of day have different trading patterns
#   - Market regime changes (calm → volatile or vice versa)
#   - Exchange infrastructure changes (affecting tick frequency)
#
# The output is an HTML report you can open in a browser — it
# shows per-feature distribution comparisons with drift scores.

def generate_data_drift_report(reference, current):
    """Generate Evidently data drift report."""

    print("\n[EVIDENTLY] Generating Data Drift Report...")

    # Select only the feature columns (not timestamps or labels)
    feature_cols = ["midprice", "spread_pct", "return_mean", "return_std",
                    "return_skew", "tick_count", "price_range_pct", "volume_24h"]

    ref_features = reference[feature_cols]
    cur_features = current[feature_cols]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_features, current_data=cur_features)

    # Save HTML report
    os.makedirs("reports/evidently", exist_ok=True)
    output_path = "reports/evidently/data_drift_report.html"
    report.save_html(output_path)
    print(f"[EVIDENTLY] Data drift report saved: {output_path}")

    return report


# ── Target Drift Report ─────────────────────────────────────────
# WHY: Even if input features don't drift, the TARGET variable
# might. If spikes were 10% of the reference data but 25% of the
# current data, the model's threshold τ might be miscalibrated.
#
# Target drift is especially dangerous because:
#   - The model might still produce confident predictions
#   - But those predictions are based on an outdated understanding
#     of what "normal" vs "spike" looks like
#   - You wouldn't notice unless you actively monitor the label
#     distribution
#
# In production, you often don't have ground truth labels
# immediately (you have to wait 60 seconds for future_vol to be
# observed). But when labels DO become available, this report
# tells you whether the world has changed.

def generate_target_drift_report(reference, current):
    """Generate target drift report using DataDriftPreset on the spike label."""

    print("\n[EVIDENTLY] Generating Target Drift Report...")

    # Include spike column alongside features so DataDriftPreset
    # checks whether the label distribution has shifted too
    cols = ["midprice", "spread_pct", "return_mean", "return_std",
            "return_skew", "tick_count", "price_range_pct", "volume_24h",
            "spike", "future_vol"]

    ref_data = reference[cols].copy()
    cur_data = current[cols].copy()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_data, current_data=cur_data)

    output_path = "reports/evidently/target_drift_report.html"
    report.save_html(output_path)
    print(f"[EVIDENTLY] Target drift report saved: {output_path}")

    return report

# ── Model Performance Report ────────────────────────────────────
# WHY: The first two reports check whether the DATA has changed.
# This report checks whether the MODEL is still performing well.
#
# It computes classification metrics (precision, recall, F1, etc.)
# on the current data, using predictions from the trained model.
# If data has drifted AND performance has dropped, that's a clear
# signal the model needs retraining.
#
# We simulate this by scoring the current data with our XGBoost
# model and comparing predictions against actual labels. In
# production, this report would run on a schedule (e.g., daily)
# to catch performance degradation early.

def generate_model_performance_report(reference, current):
    """Generate Evidently classification performance report."""

    print("\n[EVIDENTLY] Generating Model Performance Report...")

    # Load the trained XGBoost model
    import joblib

    model_path = "models/artifacts/xgboost_model.joblib"
    if not os.path.exists(model_path):
        print(f"[WARN] Model not found at {model_path}")
        print(f"[WARN] Skipping model performance report")
        return None

    model = joblib.load(model_path)

    feature_cols = ["midprice", "spread_pct", "return_mean", "return_std",
                    "return_skew", "tick_count", "price_range_pct", "volume_24h"]

    # Score both reference and current data
    ref_data = reference[feature_cols + ["spike"]].copy()
    cur_data = current[feature_cols + ["spike"]].copy()

    ref_data["prediction"] = model.predict(ref_data[feature_cols])
    cur_data["prediction"] = model.predict(cur_data[feature_cols])

    report = Report(metrics=[ClassificationPreset()])
    column_mapping = ColumnMapping()
    column_mapping.target = "spike"
    column_mapping.prediction = "prediction"

    report.run(
        reference_data=ref_data,
        current_data=cur_data,
        column_mapping=column_mapping
    )

    output_path = "reports/evidently/model_performance_report.html"
    report.save_html(output_path)
    print(f"[EVIDENTLY] Model performance report saved: {output_path}")

    return report

# ── Main: Run All Reports ───────────────────────────────────────
# WHY: Orchestrates all three reports in sequence and prints a
# summary. In production, this script would run on a schedule
# (e.g., daily cron job) and trigger alerts if drift exceeds
# thresholds. For the assignment, we run it once to demonstrate
# the monitoring capability.

def main():
    print("="*60)
    print("EVIDENTLY MONITORING REPORTS")
    print("="*60)

    # Load and split data
    df, reference, current = load_and_split_data()

    # Generate all three reports
    drift_report = generate_data_drift_report(reference, current)
    target_report = generate_target_drift_report(reference, current)
    perf_report = generate_model_performance_report(reference, current)

    # Summary
    print("\n" + "="*60)
    print("REPORT SUMMARY")
    print("="*60)
    print(f"Reports saved to: reports/evidently/")
    print(f"  1. data_drift_report.html     — Feature distribution comparison")
    print(f"  2. target_drift_report.html   — Spike label distribution comparison")
    print(f"  3. model_performance_report.html — Classification metrics comparison")
    print(f"\nOpen these HTML files in a browser to view interactive charts.")
    print(f"\n[DONE] Monitoring reports complete.")


if __name__ == "__main__":
    main()