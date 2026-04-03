"""
infer.py — Model Inference Pipeline
Loads a trained model from MLflow or local artifacts and scores
new data for volatility spike detection.

This is the Inference Pipeline in the FTI (Feature–Training–Inference)
pattern. It ensures that new data is transformed identically to
training data before scoring, preventing train-serve skew.

Usage:
    # Batch: score a parquet file
    python models/infer.py --data data/processed/features.parquet

    # Single record: pass a JSON string (simulates real-time)
    python models/infer.py --json '{"return_std": 0.00004, ...}'

    # Specify model type (default: xgboost)
    python models/infer.py --data features.parquet --model logistic_regression
"""

import os
import sys
import json
import argparse
import warnings
import yaml
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib

warnings.filterwarnings("ignore")

# ── Model Loading ───────────────────────────────────────────────
# WHY: In production, the inference pipeline needs to load whatever
# model was selected as "best" during training. We support two
# loading paths:
#
#   1. MLflow Model Registry: Load by run ID or model name. This is
#      the production-grade approach — the model lives in MLflow's
#      artifact store, versioned and traceable.
#   2. Local artifacts: Load from models/artifacts/ directory. This
#      is the fallback for development or when MLflow isn't running.
#
# We also load the scaler (for LogReg) and the feature column list.
# These MUST be the same objects from training — using a different
# scaler or different feature order would silently produce garbage
# predictions. This is the train-serve skew problem from L8
# (Value Delivery).

def load_model_from_mlflow(model_name, tracking_uri="http://localhost:5001",
                            experiment_name="crypto-volatility"):
    """Load the latest run of a specific model type from MLflow."""

    mlflow.set_tracking_uri(tracking_uri)

    # Search for the most recent run of this model type
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.model_type = '{model_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        raise ValueError(f"No runs found for model_type='{model_name}'")

    run_id = runs.iloc[0]["run_id"]
    pr_auc = runs.iloc[0].get("metrics.pr_auc", "N/A")
    print(f"[LOAD] Found run: {run_id}")
    print(f"[LOAD] Model: {model_name}, PR-AUC: {pr_auc}")

    # Load the model
    model_uri = f"runs:/{run_id}/model"
    if model_name == "logistic_regression":
        model = mlflow.sklearn.load_model(model_uri)
    elif model_name == "xgboost":
        model = mlflow.xgboost.load_model(model_uri)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    # Load preprocessing artifacts
    client = mlflow.tracking.MlflowClient()

    # Download scaler (if it exists — XGBoost doesn't need one)
    scaler = None
    try:
        scaler_dir = client.download_artifacts(run_id, "preprocessing")
        scaler_path = os.path.join(scaler_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"[LOAD] Scaler loaded from MLflow")
    except Exception:
        print(f"[LOAD] No scaler found (expected for {model_name})")

    # Download feature column list
    feature_cols = None
    try:
        feature_path = os.path.join(scaler_dir, "feature_cols.json")
        if os.path.exists(feature_path):
            with open(feature_path, "r") as f:
                feature_cols = json.load(f)
            print(f"[LOAD] Feature columns loaded: {len(feature_cols)} features")
    except Exception:
        print(f"[LOAD] No feature column list found")

    return model, scaler, feature_cols, run_id


def load_model_local(model_name):
    """Fallback: load model from local artifacts directory."""

    artifacts_dir = "models/artifacts"

    # Load feature columns
    feature_path = os.path.join(artifacts_dir, "feature_cols.json")
    if os.path.exists(feature_path):
        with open(feature_path, "r") as f:
            feature_cols = json.load(f)
        print(f"[LOAD] Feature columns loaded from local: {len(feature_cols)} features")
    else:
        raise FileNotFoundError(f"Feature columns not found at {feature_path}")

    # Load scaler (only for models that need it — not tree-based models)
    scaler = None
    if model_name == "logistic_regression":
        scaler_path = os.path.join(artifacts_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"[LOAD] Scaler loaded from local")
    else:
        print(f"[LOAD] No scaler needed for {model_name}")

    # Load model
    model_path = os.path.join(artifacts_dir, f"{model_name}_model.joblib")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"[LOAD] Model loaded from local: {model_path}")
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")

    return model, scaler, feature_cols, "local"

# ── Data Preparation for Inference ──────────────────────────────
# WHY: The most common source of bugs in ML systems isn't the model
# itself — it's the data transformation pipeline. If training
# applied features in the order [return_std, price_range_pct, ...]
# but inference accidentally reorders them, every prediction is
# wrong. If training scaled the data but inference doesn't, the
# model sees inputs in a completely different range.
#
# This function enforces three guarantees:
#   1. SAME COLUMNS: Only the features used in training are selected,
#      in the exact same order (loaded from feature_cols.json).
#   2. SAME SCALING: If a scaler was used in training, the same
#      scaler object (with the same mean/std) is applied here.
#   3. MISSING FEATURE DETECTION: If the input data is missing a
#      feature the model expects, we fail loudly rather than
#      silently producing garbage predictions.
#
# This is the train-serve skew prevention that L8 (Value Delivery)
# emphasizes. In production ML systems, more incidents come from
# data pipeline mismatches than from model bugs.

def prepare_for_inference(data, feature_cols, scaler=None):
    """Transform input data to match training format.

    Args:
        data: DataFrame with raw features (from parquet or JSON)
        feature_cols: List of feature column names from training
        scaler: Fitted StandardScaler from training (None for XGBoost)

    Returns:
        numpy array or DataFrame ready for model.predict()
    """

    # Check for missing features
    missing = [col for col in feature_cols if col not in data.columns]
    if missing:
        raise ValueError(
            f"Input data is missing {len(missing)} required feature(s): {missing}\n"
            f"Available columns: {list(data.columns)}"
        )

    # Select features in training order
    X = data[feature_cols].copy()

    # Check for NaN values
    nan_counts = X.isna().sum()
    if nan_counts.any():
        nan_cols = nan_counts[nan_counts > 0]
        print(f"[WARN] NaN values detected in {len(nan_cols)} column(s):")
        for col, count in nan_cols.items():
            print(f"  {col}: {count} NaN(s)")
        print(f"[WARN] Filling NaNs with column medians")
        X = X.fillna(X.median())

    # Apply scaling if scaler was used in training
    if scaler is not None:
        X_transformed = scaler.transform(X)
        print(f"[PREP] Scaled {len(X)} rows using training scaler")
    else:
        X_transformed = X.values
        print(f"[PREP] No scaling applied (tree-based model)")

    return X_transformed, X

# ── Batch Inference ─────────────────────────────────────────────
# WHY: Batch inference scores an entire dataset at once — typically
# a parquet file of historical features. This is useful for:
#   1. BACKTESTING: "How would the model have performed on last
#      week's data?" — run batch inference and compare to actual
#      spike labels.
#   2. GENERATING EVIDENTLY REPORTS: The monitoring tool needs
#      predictions alongside actuals to detect model drift.
#   3. VALIDATING RETRAINING: After collecting new data and
#      retraining, batch-score the old test set to confirm the
#      new model isn't worse.
#
# Output is a copy of the input DataFrame with two new columns:
#   - spike_prob: model's predicted probability of a spike (0 to 1)
#   - spike_pred: binary prediction (1 if prob >= 0.5)

def batch_inference(model, data_path, feature_cols, scaler=None,
                    output_path=None, tau_percentile=85):
    """Score an entire parquet file and optionally save results."""

    print(f"\n[BATCH] Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    df = df.dropna(subset=["future_vol"])
    print(f"[BATCH] {len(df):,} rows loaded")

    # Prepare features
    X_transformed, X_raw = prepare_for_inference(df, feature_cols, scaler)

    # Generate predictions
    y_prob = model.predict_proba(X_transformed)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Attach predictions to original dataframe
    results = df.copy()
    results["spike_prob"] = y_prob
    results["spike_pred"] = y_pred

    # If actuals exist, compute metrics
    tau = np.percentile(df["future_vol"], tau_percentile)
    results["spike_actual"] = (df["future_vol"] >= tau).astype(int)

    actual_spikes = results["spike_actual"].sum()
    predicted_spikes = results["spike_pred"].sum()

    print(f"[BATCH] Actual spikes:    {actual_spikes:,}")
    print(f"[BATCH] Predicted spikes: {predicted_spikes:,}")

    # Compute PR-AUC if actuals are available
    from sklearn.metrics import average_precision_score
    pr_auc = average_precision_score(results["spike_actual"], results["spike_prob"])
    print(f"[BATCH] PR-AUC: {pr_auc:.4f}")

    # Save results
    if output_path:
        results.to_parquet(output_path, index=False)
        print(f"[BATCH] Results saved to {output_path}")
    else:
        # Default output path
        output_path = data_path.replace(".parquet", "_scored.parquet")
        results.to_parquet(output_path, index=False)
        print(f"[BATCH] Results saved to {output_path}")

    return results

# ── Single-Record Inference ─────────────────────────────────────
# WHY: In a real production deployment, the model doesn't score
# parquet files — it scores ONE feature vector at a time as each
# new tick arrives from Kafka. This function simulates that
# real-time scoring path.
#
# The flow in production would be:
#   Kafka (ticks.raw) → featurizer → feature vector → infer.py → alert
#
# This function takes a single JSON record (like what the featurizer
# would produce) and returns a prediction. It's the bridge between
# our batch-oriented development workflow and the real-time system
# the assignment envisions.
#
# By supporting both batch and single-record inference in the same
# script, we guarantee that the same preprocessing logic is used
# in both cases — eliminating a common source of train-serve skew
# where the batch pipeline works fine but the real-time pipeline
# silently transforms data differently.

def single_inference(model, record, feature_cols, scaler=None):
    """Score a single feature record.

    Args:
        model: Trained model object
        record: dict with feature values (e.g., from Kafka or JSON input)
        feature_cols: List of feature column names from training
        scaler: Fitted StandardScaler (None for XGBoost)

    Returns:
        dict with prediction results
    """

    # Convert single record to DataFrame (model expects 2D input)
    df = pd.DataFrame([record])

    # Prepare features using the same logic as batch
    X_transformed, X_raw = prepare_for_inference(df, feature_cols, scaler)

    # Generate prediction
    y_prob = model.predict_proba(X_transformed)[:, 1][0]
    y_pred = int(y_prob >= 0.5)

    # Determine alert level based on probability
    if y_prob >= 0.8:
        alert_level = "HIGH"
    elif y_prob >= 0.5:
        alert_level = "MEDIUM"
    elif y_prob >= 0.3:
        alert_level = "LOW"
    else:
        alert_level = "NONE"

    result = {
        "spike_probability": round(float(y_prob), 6),
        "spike_prediction": y_pred,
        "alert_level": alert_level,
        "features_used": {col: round(float(record.get(col, 0)), 8)
                          for col in feature_cols},
    }

    # Print formatted output
    print(f"\n[INFER] Single-Record Prediction")
    print(f"  Spike probability: {y_prob:.4f}")
    print(f"  Prediction:        {'SPIKE' if y_pred else 'NO SPIKE'}")
    print(f"  Alert level:       {alert_level}")
    print(f"  Key features:")

    # Show the most important features for this prediction
    for col in ["return_std", "price_range_pct", "return_mean"]:
        if col in record:
            print(f"    {col}: {record[col]:.8f}")

    return result

# ── Main: Entry Point ───────────────────────────────────────────
# WHY: The main function handles argument parsing and routes to
# either batch or single-record inference. It tries MLflow first
# for model loading, and falls back to local artifacts if MLflow
# isn't available. This makes the script work both in the full
# Docker environment (MLflow running) and in a lightweight
# development setup.

def main():
    parser = argparse.ArgumentParser(
        description="Score new data using a trained volatility spike model"
    )
    parser.add_argument(
        "--data", default=None,
        help="Path to parquet file for batch inference"
    )
    parser.add_argument(
        "--json", default=None,
        help="JSON string for single-record inference"
    )
    parser.add_argument(
        "--model", default="xgboost",
        choices=["logistic_regression", "xgboost"],
        help="Which model to load (default: xgboost)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for batch results (default: input_scored.parquet)"
    )
    args = parser.parse_args()

    # ── Validate arguments ───────────────────────────────────────
    if args.data is None and args.json is None:
        print("[ERROR] Must specify either --data (batch) or --json (single record)")
        print("Examples:")
        print("  python models/infer.py --data data/processed/features.parquet")
        print('  python models/infer.py --json \'{"return_std": 0.00004, ...}\'')
        sys.exit(1)

    # ── Load config ──────────────────────────────────────────────
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        tracking_uri = config.get("mlflow", {}).get("tracking_uri", "http://localhost:5001")
        experiment_name = config.get("mlflow", {}).get("experiment_name", "crypto-volatility")
    else:
        tracking_uri = "http://localhost:5001"
        experiment_name = "crypto-volatility"

    # ── Load model ───────────────────────────────────────────────
    print(f"[INFER] Loading model: {args.model}")
    try:
        model, scaler, feature_cols, run_id = load_model_from_mlflow(
            args.model, tracking_uri, experiment_name
        )
        print(f"[INFER] Loaded from MLflow (run: {run_id})")
    except Exception as e:
        print(f"[INFER] MLflow load failed: {e}")
        print(f"[INFER] Falling back to local artifacts...")
        try:
            model, scaler, feature_cols, run_id = load_model_local(args.model)
            print(f"[INFER] Loaded from local artifacts")
        except Exception as e2:
            print(f"[ERROR] Could not load model: {e2}")
            print(f"[ERROR] Run train.py first to generate model artifacts")
            sys.exit(1)

    # ── Route to appropriate inference mode ──────────────────────
    if args.data:
        # Batch inference
        results = batch_inference(
            model, args.data, feature_cols, scaler, args.output
        )
        print(f"\n[DONE] Batch inference complete. {len(results):,} rows scored.")

    elif args.json:
        # Single-record inference
        try:
            record = json.loads(args.json)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON: {e}")
            sys.exit(1)

        result = single_inference(model, record, feature_cols, scaler)

        # Output as formatted JSON
        print(f"\n[RESULT]")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()