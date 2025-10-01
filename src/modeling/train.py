"""
This script trains an XGBoost classifier on preprocessed fraud detection data.

It performs the following steps:
1.  Loads a Parquet dataset using Polars.
2.  Applies basic feature engineering (one-hot encoding, log transform).
3.  Splits data into training and testing sets, stratified by the target variable.
4.  Uses SMOTE to handle class imbalance on the training data.
5.  Trains a final XGBoost model and evaluates it on the test set.
6.  Performs 5-fold cross-validation with SMOTE to check for model stability.
7.  Saves the trained model, performance metrics, and evaluation plots to an 'output' directory.
"""
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import polars as pl
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

# --- Constants ---
TARGET_COLUMN = "isFraud"
OUTPUT_DIR = Path("output")
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"
RANDOM_STATE = 42


def load_dataset(parquet_path: Path) -> Tuple[pl.DataFrame, np.ndarray]:
    """Loads a Parquet file and separates features from the target."""
    data = pl.read_parquet(parquet_path)
    y = data[TARGET_COLUMN].to_numpy()
    X = data.drop(TARGET_COLUMN)
    return X, y


def basic_feature_engineering(data: pl.DataFrame) -> pl.DataFrame:
    """Applies basic transformations to the feature set."""
    # Drop high-cardinality identifiers that are not useful as features
    columns_to_drop = ["nameOrig", "nameDest"]
    columns_to_drop = [c for c in columns_to_drop if c in data.columns]
    data = data.drop(columns_to_drop)

    # One-hot encode the transaction 'type' column
    if "type" in data.columns:
        data = data.to_dummies(columns=["type"], drop_first=True)

    # Apply a log1p transform to the 'amount' to reduce skewness, a common
    # practice for financial data. Using native polars expression is faster.
    if "amount" in data.columns:
        data = data.with_columns(
            pl.col("amount").log1p().alias("amount_log")
        )
    return data


def train_xgb_model(parquet_path: str) -> None:
    """Main function to train, evaluate, and save the fraud detection model."""
    print("Starting model training process...")
    parquet = Path(parquet_path)
    X_pl, y = load_dataset(parquet)

    print("Performing feature engineering...")
    X_pl = basic_feature_engineering(X_pl)

    # Convert to pandas just before modeling, as scikit-learn/xgboost require it
    X = X_pl.to_pandas()

    # Split data, ensuring the proportion of fraud cases is the same in train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # Address class imbalance using SMOTE on the training set only
    print("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Define the XGBoost model with a standard set of hyperparameters
    model = XGBClassifier(
        tree_method="hist",
        eval_metric="aucpr",  # Area Under PR Curve is great for imbalanced data
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )

    print("Training final model on resampled data...")
    model.fit(X_train_resampled, y_train_resampled)

    # --- Evaluation on the original, unseen test set ---
    print("\n--- Model Evaluation ---")
    proba = model.predict_proba(X_test)[:, 1]
    # NOTE: The 0.5 threshold is a default and may not be optimal.
    # It should be tuned to balance precision/recall based on business needs.
    pred = (proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC (Average Precision): {ap:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, pred, digits=4))

    # --- Save artifacts (metrics, plots, model) ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    (REPORTS_DIR / "metrics.txt").write_text(
        f"ROC_AUC={roc_auc:.6f}\nPR_AUC={ap:.6f}\n"
    )

    try:
        import matplotlib.pyplot as plt
        RocCurveDisplay.from_predictions(y_test, proba)
        plt.title("ROC Curve")
        plt.savefig(REPORTS_DIR / "roc_curve.png", bbox_inches="tight")
        plt.close()

        PrecisionRecallDisplay.from_predictions(y_test, proba)
        plt.title("Precision-Recall Curve")
        plt.savefig(REPORTS_DIR / "pr_curve.png", bbox_inches="tight")
        plt.close()
        print(f"Saved ROC and PR curves to {REPORTS_DIR}")
    except Exception as e:
        print(f"Could not generate plots. Error: {e}")

    # --- Cross-validation for stability check ---
    print("\n--- Performing 5-Fold Cross-Validation ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    auc_scores = []
    # Use the same model definition for a fair comparison
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"Processing Fold {fold}/5...")
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        # Apply SMOTE *inside* the loop on the training fold only to prevent data leakage
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

        model.fit(X_tr_res, y_tr_res)
        fold_proba = model.predict_proba(X_va)[:, 1]
        auc_scores.append(roc_auc_score(y_va, fold_proba))

    print(f"CV ROC-AUC: {np.mean(auc_scores):.4f} +/- {np.std(auc_scores):.4f}")

    # --- Persist the final trained model ---
    model_path = MODELS_DIR / "xgb_fraud_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nFinal model saved to {model_path}")


if __name__ == "__main__":
    # Assumes data is in a `data/processed` directory relative to project root
    parquet_file = Path("data") / "processed" / "fraud.parquet"
    if not parquet_file.exists():
        raise SystemExit(
            f"Parquet file not found at '{parquet_file}'. "
            "Please ensure it exists or run the data preparation script."
        )
    train_xgb_model(str(parquet_file))