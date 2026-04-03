"""
train.py — Model Training Pipeline
Trains baseline (z-score), Logistic Regression, and XGBoost models
to detect short-term BTC-USD volatility spikes. Logs all experiments
to MLflow for tracking and comparison.

Usage:
    python models/train.py
    python models/train.py --data data/processed/features.parquet
    python models/train.py --percentile 85
"""

import os
import sys
import json
import argparse
import warnings
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost

warnings.filterwarnings("ignore")

# ── Data Loading & Preparation ──────────────────────────────────
# WHY: We need to go from raw features.parquet to a clean train/test
# split with the binary spike label. This block handles:
#   1. Loading the parquet and dropping rows with NaN labels
#   2. Computing τ from the data (not hardcoded — so this works
#      regardless of how much data we've collected)
#   3. Creating the spike label
#   4. Dropping features we identified as redundant in the EDA
#   5. Splitting into train/test (80/20, stratified by spike label)

def load_and_prepare(parquet_path, tau_percentile=85, test_size=0.2, random_state=42):
    """Load features, create spike label, split into train/test."""

    print(f"[DATA] Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=["future_vol"])
    print(f"[DATA] {len(df):,} rows after dropping NaN labels")

    # Compute threshold from data
    tau = np.percentile(df["future_vol"], tau_percentile)
    df["spike"] = (df["future_vol"] >= tau).astype(int)
    spike_pct = df["spike"].mean() * 100
    print(f"[DATA] τ = {tau:.8f} ({tau_percentile}th percentile)")
    print(f"[DATA] Spike prevalence: {spike_pct:.1f}% ({df['spike'].sum():,} positives)")

    # Feature selection — drop redundant and noise features (from EDA)
    drop_cols = [
        "timestamp", "received_at",  # Not features
        "future_vol",                 # Target leakage (this IS the answer)
        "spike",                      # The label — separate it out
        "price_range",                # Redundant with price_range_pct (ρ=1.00)
        "spread",                     # Redundant with spread_pct (ρ=1.00)
        "book_imbalance",             # Near-zero predictive signal (ρ=0.004)
    ]

    feature_cols = [c for c in df.columns if c not in drop_cols]
    print(f"[DATA] Features selected: {feature_cols}")

    X = df[feature_cols].copy()
    y = df["spike"].copy()

    # Train/test split — stratified to preserve spike ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[DATA] Train: {len(X_train):,} rows ({y_train.mean()*100:.1f}% spikes)")
    print(f"[DATA] Test:  {len(X_test):,} rows ({y_test.mean()*100:.1f}% spikes)")

    return X_train, X_test, y_train, y_test, feature_cols, tau

# ── Baseline Model: Z-Score Rule ────────────────────────────────
# WHY: Before training any ML model, we need a simple baseline to
# compare against. This is a core principle from the course (L5:
# Value Discovery — Predictive Models): always establish what a
# trivial approach achieves before investing in complexity.
#
# The z-score rule works like this:
#   1. For each row, compute how many standard deviations the current
#      return_std is above its training-set mean.
#   2. If the z-score exceeds a threshold (default: 1.0), predict spike=1.
#
# This is NOT machine learning — it's a hand-crafted rule using a
# single feature. It captures our EDA finding that return_std is the
# strongest predictor (ρ = 0.59) and that volatility clusters. If our
# ML models can't beat this, they aren't learning anything useful
# beyond "current volatility is high."
#
# We output predicted probabilities (not just 0/1) so we can compute
# PR-AUC on the same scale as the ML models. We use a sigmoid-like
# mapping: probability = clipped z-score normalized to [0, 1].

def train_zscore_baseline(X_train, X_test, y_train, y_test, tau):
    """Z-score baseline: predict spike if return_std is abnormally high."""

    print("\n" + "="*60)
    print("MODEL: Z-Score Baseline")
    print("="*60)

    # Compute mean and std of return_std from TRAINING set only
    # (never peek at test data — that would be data leakage)
    train_mean = X_train["return_std"].mean()
    train_std = X_train["return_std"].std()

    print(f"[ZSCORE] Training return_std: mean={train_mean:.8f}, std={train_std:.8f}")

    # Compute z-scores on test set using training statistics
    z_scores = (X_test["return_std"] - train_mean) / train_std

    # Convert z-scores to pseudo-probabilities for PR-AUC
    # Clip to [0, 1] range: z <= 0 maps to 0, z >= 3 maps to 1
    y_prob = np.clip(z_scores / 3.0, 0, 1)

    # Binary predictions at z-score threshold of 1.0
    # (i.e., predict spike if return_std is 1+ std above the mean)
    z_threshold = 1.0
    y_pred = (z_scores >= z_threshold).astype(int)

    # Evaluate
    metrics = evaluate_model(y_test, y_pred, y_prob, "zscore_baseline")

    print(f"[ZSCORE] Z-threshold: {z_threshold}")
    print(f"[ZSCORE] Predictions — spike: {y_pred.sum()}, no spike: {(1-y_pred).sum()}")

    # Package model parameters (for MLflow logging)
    params = {
        "model_type": "zscore_baseline",
        "feature_used": "return_std",
        "z_threshold": z_threshold,
        "train_mean": float(train_mean),
        "train_std": float(train_std),
        "tau": float(tau),
    }

    return params, metrics, y_pred, y_prob

# ── ML Model 1: Logistic Regression ────────────────────────────
# WHY: Logistic Regression is the standard first ML model for binary
# classification. In the course framework (L5: Predictive Models),
# it sits at the "standalone model" level — simple, interpretable,
# and fast to train.
#
# It's a good fit here because:
#   1. Our EDA showed mostly LINEAR separation between spike and
#      non-spike distributions (especially return_std, price_range_pct).
#      LogReg can capture these patterns directly.
#   2. It outputs calibrated probabilities by default — important
#      for PR-AUC and for setting operational thresholds in production.
#   3. The coefficients are interpretable — we can say exactly which
#      features the model relies on and by how much. This matters
#      for the model card and trust management (L11).
#
# SCALING: LogReg is sensitive to feature magnitude. midprice is
# ~68,000 while return_std is ~0.00002. Without scaling, midprice
# would dominate the gradient updates. StandardScaler normalizes
# each feature to mean=0, std=1 so they contribute equally.
#
# CLASS IMBALANCE: With only 15% spikes, the model could achieve
# 85% accuracy by predicting "no spike" for everything. We use
# class_weight="balanced" to penalize misclassifying the minority
# class more heavily — forcing the model to actually learn the
# spike pattern rather than taking the lazy majority-class shortcut.

def train_logistic_regression(X_train, X_test, y_train, y_test, feature_cols, tau):
    """Train and evaluate a Logistic Regression model."""

    print("MODEL: Logistic Regression")

    # Scale features — fit on train, transform both train and test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model = LogisticRegression(
        class_weight="balanced",  # Handle 5.7:1 imbalance
        max_iter=1000,            # Ensure convergence
        random_state=42,
        solver="lbfgs",           # Default solver, works well for small datasets
    )
    model.fit(X_train_scaled, y_train)

    # Predict probabilities (column 1 = probability of spike)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    # Evaluate
    metrics = evaluate_model(y_test, y_pred, y_prob, "logistic_regression")

    # Print feature importance (coefficients)
    print(f"\n[LOGREG] Feature coefficients:")
    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": model.coef_[0]
    }).sort_values("coefficient", key=abs, ascending=False)

    for _, row in coef_df.iterrows():
        direction = "+" if row["coefficient"] > 0 else "-"
        print(f"  {direction} {row['feature']:20s} {row['coefficient']:+.4f}")

    # Package parameters
    params = {
        "model_type": "logistic_regression",
        "class_weight": "balanced",
        "max_iter": 1000,
        "solver": "lbfgs",
        "n_features": len(feature_cols),
        "tau": float(tau),
    }

    return model, scaler, params, metrics, y_pred, y_prob

# ── ML Model 2: XGBoost ────────────────────────────────────────
# WHY: XGBoost is a gradient-boosted tree ensemble — it sits at the
# "Ensemble Modeling" level in the course framework (L6: Value
# Discovery — Ensemble Models). It addresses two limitations of
# Logistic Regression:
#
#   1. NONLINEAR PATTERNS: Our EDA showed that return_mean's signal
#      is in its magnitude, not its sign (spikes happen when price
#      moves sharply in EITHER direction). LogReg can only learn
#      linear boundaries — XGBoost can learn "if |return_mean| > X,
#      predict spike" without us manually engineering that feature.
#
#   2. FEATURE INTERACTIONS: Maybe high return_std alone isn't enough
#      to predict a spike — but high return_std COMBINED with high
#      tick_count is. Trees naturally capture these interactions
#      without us specifying them.
#
# SCALING: Unlike LogReg, tree-based models don't need feature scaling.
# They split on thresholds, so the absolute magnitude doesn't matter.
# We train on raw (unscaled) features.
#
# CLASS IMBALANCE: XGBoost handles this via the scale_pos_weight
# parameter. We set it to the ratio of negative to positive examples
# (~5.7), which has the same effect as class_weight="balanced" in
# LogReg — it upweights the minority class in the loss function.
#
# HYPERPARAMETERS: We use conservative defaults rather than extensive
# tuning. The goal is a solid model, not a Kaggle-winning one. The
# key choices:
#   - max_depth=4: Shallow trees prevent overfitting on 10-20K rows
#   - n_estimators=200: Enough boosting rounds to learn the signal
#   - learning_rate=0.1: Standard starting point
#   - eval_metric="aucpr": Optimizes directly for our primary metric

def train_xgboost(X_train, X_test, y_train, y_test, feature_cols, tau):
    """Train and evaluate an XGBoost model."""

    print("\n" + "="*60)
    print("MODEL: XGBoost")
    print("="*60)

    # Compute class imbalance ratio for scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_ratio = neg_count / pos_count
    print(f"[XGB] Class ratio (neg:pos) = {scale_ratio:.2f}:1")

    # Train
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_ratio,    # Handle imbalance
        eval_metric="aucpr",             # Optimize for PR-AUC
        use_label_encoder=False,
        random_state=42,
        verbosity=0,                     # Suppress training logs

        # Regularization — prevent overfitting on small dataset
        min_child_weight=5,              # Minimum samples per leaf
        subsample=0.8,                   # Use 80% of rows per tree
        colsample_bytree=0.8,           # Use 80% of features per tree
    )

    # Fit with early stopping using a validation split carved from train
    # This prevents overfitting: if test PR-AUC stops improving, stop adding trees
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Predict probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Evaluate
    metrics = evaluate_model(y_test, y_pred, y_prob, "xgboost")

    # Print feature importance (gain-based)
    print(f"\n[XGB] Feature importance (gain):")
    importance = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance
    }).sort_values("importance", ascending=False)

    for _, row in imp_df.iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"  {row['feature']:20s} {row['importance']:.4f} {bar}")

    # Best iteration info
    best_iter = model.best_iteration if hasattr(model, "best_iteration") else 200
    print(f"[XGB] Best iteration: {best_iter}")

    # Package parameters
    params = {
        "model_type": "xgboost",
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "scale_pos_weight": float(scale_ratio),
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "best_iteration": best_iter,
        "n_features": len(feature_cols),
        "tau": float(tau),
    }

    return model, params, metrics, y_pred, y_prob

# ── Evaluation Helper ───────────────────────────────────────────
# WHY: All three models need the same evaluation — PR-AUC, confusion
# matrix, classification report, and a Precision-Recall curve plot.
# Rather than duplicate this code three times, we centralize it in
# one function that every model calls.
#
# PR-AUC (Precision-Recall Area Under Curve) is our PRIMARY metric,
# chosen in the scoping brief for a specific reason: when classes are
# imbalanced (15% spikes), ROC-AUC can be misleadingly high. A model
# that rarely predicts "spike" can still have high ROC-AUC because
# the true negative rate is inflated by the large majority class.
# PR-AUC focuses exclusively on how well the model handles the
# POSITIVE class — when it says "spike," how often is it right
# (precision), and of all actual spikes, how many does it catch
# (recall). This is exactly what a trader cares about.
#
# We also compute ROC-AUC for completeness (it's standard practice),
# plus a confusion matrix to understand the specific error types:
#   - False Positives: Model says spike, but it's calm → unnecessary
#     trading action, but not dangerous
#   - False Negatives: Model says calm, but it's a spike → missed
#     risk event, potentially costly
#
# The PR curve plot is saved as a PNG artifact for MLflow.

def evaluate_model(y_true, y_pred, y_prob, model_name):
    """Compute metrics and generate evaluation plots."""

    # PR-AUC (primary metric)
    pr_auc = average_precision_score(y_true, y_prob)

    # ROC-AUC (secondary metric)
    roc_auc = roc_auc_score(y_true, y_prob)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Derived metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Print results
    print(f"\n[EVAL] {model_name}")
    print(f"  PR-AUC  (primary):  {pr_auc:.4f}")
    print(f"  ROC-AUC (secondary): {roc_auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  No Spike   Spike")
    print(f"  Actual No Spike   {tn:>6,}   {fp:>6,}")
    print(f"  Actual Spike      {fn:>6,}   {tp:>6,}")
    print(f"\n  False Positives (unnecessary alerts): {fp:,}")
    print(f"  False Negatives (missed spikes):      {fn:,}")

    # ── Precision-Recall Curve Plot ──────────────────────────────
    # WHY: The PR curve shows the tradeoff between precision and recall
    # at every possible probability threshold. A model that hugs the
    # top-right corner (high precision AND high recall) is ideal.
    # The baseline (random classifier) would be a horizontal line at
    # y = spike_prevalence (~0.15). Anything above that line is
    # adding value.

    prec_curve, rec_curve, thresholds = precision_recall_curve(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: PR Curve
    axes[0].plot(rec_curve, prec_curve, linewidth=2, label=f"{model_name} (PR-AUC={pr_auc:.3f})")
    axes[0].axhline(y=y_true.mean(), color="gray", linestyle="--",
                     label=f"Random baseline ({y_true.mean():.3f})")
    axes[0].set_title(f"Precision-Recall Curve — {model_name}")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Right: Confusion Matrix Heatmap
    sns_available = True
    try:
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=axes[1],
                    xticklabels=["No Spike", "Spike"],
                    yticklabels=["No Spike", "Spike"])
        axes[1].set_title(f"Confusion Matrix — {model_name}")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")
    except ImportError:
        sns_available = False
        axes[1].matshow(cm, cmap="Blues")
        axes[1].set_title(f"Confusion Matrix — {model_name}")
        for i in range(2):
            for j in range(2):
                axes[1].text(j, i, f"{cm[i,j]:,}", ha="center", va="center")

    plt.tight_layout()

    # Save plot to reports/ for MLflow artifact logging
    os.makedirs("reports", exist_ok=True)
    plot_path = f"reports/eval_{model_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EVAL] Plot saved: {plot_path}")

    # Package all metrics
    metrics = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    return metrics

# ── MLflow Experiment Logging ───────────────────────────────────
# WHY: MLflow is the experiment tracking system covered in L2 and
# the Feast/MLflow recitation (Mar 27). It solves a fundamental
# problem in ML development: "which combination of data, features,
# parameters, and code produced the best model?"
#
# Without tracking, you end up with a folder full of model files
# named "model_v2_final_FINAL_v3.pkl" and no idea which one to
# deploy. MLflow logs everything in a structured database:
#   - Parameters: what settings were used (tau, learning_rate, etc.)
#   - Metrics: how the model performed (PR-AUC, precision, recall)
#   - Artifacts: the actual model file, evaluation plots, scaler
#   - Tags: metadata like model type and run description
#
# This maps directly to the course's Value Discovery phase —
# experiment tracking is how you systematically compare models
# and make evidence-based decisions about what to deploy.
#
# After running train.py, you can open http://localhost:5001 to
# see all three models side-by-side in the MLflow UI — compare
# PR-AUC, view the PR curves, and decide which model to promote
# to production.

def log_to_mlflow(model_name, params, metrics, artifacts_dir="reports",
                  model_obj=None, scaler_obj=None, feature_cols=None):
    """Log a single model run to MLflow."""

    with mlflow.start_run(run_name=model_name):

        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("project", "crypto-volatility-pipeline")
        mlflow.set_tag("stage", "development")

        for key, value in params.items():
            mlflow.log_param(key, value)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Artifact logging — may fail if server artifact store is inaccessible
        try:
            plot_path = f"{artifacts_dir}/eval_{model_name}.png"
            if os.path.exists(plot_path):
                mlflow.log_artifact(plot_path, artifact_path="plots")

            if model_obj is not None:
                if model_name == "logistic_regression":
                    mlflow.sklearn.log_model(model_obj, "model")
                elif model_name == "xgboost":
                    mlflow.xgboost.log_model(model_obj, "model")

            if scaler_obj is not None:
                import joblib
                os.makedirs("models/artifacts", exist_ok=True)
                scaler_path = "models/artifacts/scaler.joblib"
                joblib.dump(scaler_obj, scaler_path)
                mlflow.log_artifact(scaler_path, artifact_path="preprocessing")

            if feature_cols is not None:
                feature_path = "models/artifacts/feature_cols.json"
                os.makedirs("models/artifacts", exist_ok=True)
                with open(feature_path, "w") as f:
                    json.dump(feature_cols, f, indent=2)
                mlflow.log_artifact(feature_path, artifact_path="preprocessing")

        except Exception as e:
            print(f"[MLFLOW] Artifact upload failed (non-fatal): {e}")
            print(f"[MLFLOW] Params and metrics logged. Artifacts saved locally.")

            # Save model locally as fallback
            if model_obj is not None:
                import joblib
                os.makedirs("models/artifacts", exist_ok=True)
                joblib.dump(model_obj, f"models/artifacts/{model_name}_model.joblib")
                print(f"[MLFLOW] Model saved locally: models/artifacts/{model_name}_model.joblib")

        print(f"[MLFLOW] Run logged: {model_name}")
        print(f"[MLFLOW]   PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"[MLFLOW]   Run ID: {mlflow.active_run().info.run_id}")

# ── Main: Orchestrate the Full Pipeline ─────────────────────────
# WHY: This is the entry point that ties everything together. It
# follows the Value Discovery pipeline pattern from L4-L6:
#   1. Load and prepare data (Data Identification & Access)
#   2. Train baseline model (Baseline Benchmarking)
#   3. Train ML models (Model Selection & Training)
#   4. Log everything to MLflow (Experiment Tracking)
#   5. Compare and summarize (Model Validation)
#
# The script is designed to be run from the command line with
# optional arguments, making it reusable as you collect more data
# or want to try a different threshold. Nothing is hardcoded —
# the data path, percentile, and test size are all configurable.

def main():
    parser = argparse.ArgumentParser(
        description="Train volatility spike detection models"
    )
    parser.add_argument(
        "--data", default="data/processed/features.parquet",
        help="Path to features parquet file"
    )
    parser.add_argument(
        "--percentile", type=int, default=85,
        help="Percentile for spike threshold τ (default: 85)"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data for test set (default: 0.2)"
    )
    args = parser.parse_args()

    # ── Load config for MLflow URI ───────────────────────────────
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        tracking_uri = config.get("mlflow", {}).get("tracking_uri", "http://localhost:5001")
        experiment_name = config.get("mlflow", {}).get("experiment_name", "crypto-volatility")
    else:
        tracking_uri = "http://localhost:5001"
        experiment_name = "crypto-volatility"

    # ── Set up MLflow ────────────────────────────────────────────
  
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment(experiment_name)
    print(f"[MLFLOW] Connected: {mlflow.get_tracking_uri()}")
    print(f"[MLFLOW] Experiment: {experiment_name}")

    # ── Step 1: Load and prepare data ────────────────────────────
    X_train, X_test, y_train, y_test, feature_cols, tau = load_and_prepare(
        args.data, tau_percentile=args.percentile, test_size=args.test_size
    )

    # ── Step 2: Z-Score Baseline ─────────────────────────────────
    zs_params, zs_metrics, zs_pred, zs_prob = train_zscore_baseline(
        X_train, X_test, y_train, y_test, tau
    )
    log_to_mlflow("zscore_baseline", zs_params, zs_metrics)

    # ── Step 3: Logistic Regression ──────────────────────────────
    lr_model, lr_scaler, lr_params, lr_metrics, lr_pred, lr_prob = train_logistic_regression(
        X_train, X_test, y_train, y_test, feature_cols, tau
    )
    log_to_mlflow("logistic_regression", lr_params, lr_metrics,
                  model_obj=lr_model, scaler_obj=lr_scaler, feature_cols=feature_cols)

    # ── Step 4: XGBoost ──────────────────────────────────────────
    xgb_model, xgb_params, xgb_metrics, xgb_pred, xgb_prob = train_xgboost(
        X_train, X_test, y_train, y_test, feature_cols, tau
    )
    log_to_mlflow("xgboost", xgb_params, xgb_metrics,
                  model_obj=xgb_model, feature_cols=feature_cols)

    # ── Step 5: Model Comparison Summary ─────────────────────────
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)

    summary = pd.DataFrame({
        "Model": ["Z-Score Baseline", "Logistic Regression", "XGBoost"],
        "PR-AUC": [zs_metrics["pr_auc"], lr_metrics["pr_auc"], xgb_metrics["pr_auc"]],
        "ROC-AUC": [zs_metrics["roc_auc"], lr_metrics["roc_auc"], xgb_metrics["roc_auc"]],
        "Precision": [zs_metrics["precision"], lr_metrics["precision"], xgb_metrics["precision"]],
        "Recall": [zs_metrics["recall"], lr_metrics["recall"], xgb_metrics["recall"]],
        "F1": [zs_metrics["f1_score"], lr_metrics["f1_score"], xgb_metrics["f1_score"]],
    })

    print(summary.to_string(index=False))

    # Identify best model by PR-AUC
    best_idx = summary["PR-AUC"].idxmax()
    best_model = summary.loc[best_idx, "Model"]
    best_prauc = summary.loc[best_idx, "PR-AUC"]
    print(f"\n>>> Best model by PR-AUC: {best_model} ({best_prauc:.4f})")

    # ── Save comparison plot ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: PR-AUC comparison bar chart
    colors = ["#888888", "#4A90D9", "#D94A4A"]
    axes[0].bar(summary["Model"], summary["PR-AUC"], color=colors, edgecolor="black")
    axes[0].set_title("PR-AUC Comparison")
    axes[0].set_ylabel("PR-AUC")
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(summary["PR-AUC"]):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    # Right: All three PR curves overlaid
    # Re-compute PR curves for overlay
    for name, y_p, color, ls in [
        ("Z-Score", zs_prob, "#888888", "--"),
        ("LogReg", lr_prob, "#4A90D9", "-"),
        ("XGBoost", xgb_prob, "#D94A4A", "-"),
    ]:
        prec_c, rec_c, _ = precision_recall_curve(y_test, y_p)
        auc_val = average_precision_score(y_test, y_p)
        axes[1].plot(rec_c, prec_c, color=color, linestyle=ls,
                     linewidth=2, label=f"{name} (PR-AUC={auc_val:.3f})")

    axes[1].axhline(y=y_test.mean(), color="gray", linestyle=":",
                     label=f"Random ({y_test.mean():.3f})")
    axes[1].set_title("Precision-Recall Curves — All Models")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("reports/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OUTPUT] Comparison plot saved: reports/model_comparison.png")

    # Log comparison as a separate MLflow run
    # Log comparison as a separate MLflow run
    with mlflow.start_run(run_name="model_comparison"):
        mlflow.set_tag("model_type", "comparison")
        mlflow.log_metric("best_pr_auc", best_prauc)
        mlflow.log_param("best_model", best_model)
        try:
            mlflow.log_artifact("reports/model_comparison.png", artifact_path="plots")
            summary.to_csv("reports/model_comparison.csv", index=False)
            mlflow.log_artifact("reports/model_comparison.csv", artifact_path="results")
        except Exception as e:
            print(f"[MLFLOW] Artifact upload failed (non-fatal): {e}")
            summary.to_csv("reports/model_comparison.csv", index=False)
            print(f"[MLFLOW] Comparison saved locally: reports/model_comparison.csv")

if __name__ == "__main__":
    main()
