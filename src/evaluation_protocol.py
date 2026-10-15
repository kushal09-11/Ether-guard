"""
Evaluation Protocol & Experimental Design for Ethereum Wallet Fraud Detection (ChainSentinel).
Defines stratified train/validation/test splitting, leakage-free pipeline interfaces,
validation-set threshold optimization, calibration assessment, and multi-metric reporting.
Ensures zero data snooping between splits and forbids reporting accuracy in isolation.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    classification_report,
)
from sklearn.dummy import DummyClassifier
import config


def create_stratified_train_val_test_splits(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = config.RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Creates stratified Train / Validation / Test splits with a reproducible random seed.
    Splits:
      - Train: (1.0 - val_size - test_size) [e.g. 70%]
      - Validation: val_size [e.g. 15%]
      - Test: test_size [e.g. 15%] (Strictly held out until final evaluation)
    
    Returns:
      Dictionary containing X_train, X_val, X_test, y_train, y_val, y_test,
      along with class distribution statistics for each split.
    """
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

    # First split: Separate Test set from Train+Val
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_seed,
    )

    # Second split: Separate Validation set from Train
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=relative_val_size,
        stratify=y_train_val,
        random_state=random_seed,
    )

    def get_stats(series: pd.Series, split_name: str) -> Dict[str, Any]:
        counts = series.value_counts().to_dict()
        total = len(series)
        return {
            "split": split_name,
            "total_samples": total,
            "class_0_count": int(counts.get(0, 0)),
            "class_0_pct": round((counts.get(0, 0) / total) * 100, 2),
            "class_1_count": int(counts.get(1, 0)),
            "class_1_pct": round((counts.get(1, 0) / total) * 100, 2),
            "fraud_ratio": round(counts.get(1, 0) / total, 4),
        }

    stats = {
        "train": get_stats(y_train, "train"),
        "validation": get_stats(y_val, "validation"),
        "test": get_stats(y_test, "test"),
    }

    return {
        "X_train": X_train.reset_index(drop=True),
        "X_val": X_val.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_val": y_val.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "split_statistics": stats,
    }


def find_optimal_threshold_on_validation(
    y_val_true: np.ndarray,
    y_val_probs: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Finds the optimal decision threshold on the Validation set.
    The Test set must NEVER be used for threshold tuning.
    
    Args:
        y_val_true: Ground truth binary labels on validation set
        y_val_probs: Predicted probabilities for the positive class (fraud)
        metric: Optimization target ('f1', 'precision', or 'recall')
        
    Returns:
        (best_threshold, best_score)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_score = -1.0

    for thresh in thresholds:
        preds = (y_val_probs >= thresh).astype(int)
        if metric == "f1":
            score = f1_score(y_val_true, preds, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_val_true, preds, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_val_true, preds, zero_division=0)
        else:
            raise ValueError(f"Unsupported optimization metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return round(float(best_threshold), 4), round(float(best_score), 4)


def compute_comprehensive_evaluation_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Computes rigorous evaluation metrics for fraud detection.
    Never reports accuracy alone. Includes PR-AUC, ROC-AUC, F1, Precision,
    Recall, Brier score (calibration), and the Confusion Matrix.
    """
    y_true_arr = np.array(y_true)
    y_probs_arr = np.array(y_probs)
    y_pred = (y_probs_arr >= threshold).astype(int)

    # Confusion matrix breakdown
    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Probability-based metrics (threshold-independent)
    try:
        roc_auc = round(float(roc_auc_score(y_true_arr, y_probs_arr)), 4)
    except Exception:
        roc_auc = float("nan")

    try:
        pr_auc = round(float(average_precision_score(y_true_arr, y_probs_arr)), 4)
    except Exception:
        pr_auc = float("nan")

    # Calibration metric: Brier Score (lower is better, 0.0 is perfect calibration)
    brier = round(float(brier_score_loss(y_true_arr, y_probs_arr)), 4)

    # Class-specific metrics
    report = classification_report(y_true_arr, y_pred, output_dict=True, zero_division=0)

    return {
        "model_name": model_name,
        "decision_threshold": round(float(threshold), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "threshold_independent_metrics": {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "brier_score": brier,
        },
        "threshold_dependent_metrics": {
            "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
            "fraud_class_1": {
                "precision": round(float(report["1"]["precision"]), 4),
                "recall": round(float(report["1"]["recall"]), 4),
                "f1_score": round(float(report["1"]["f1-score"]), 4),
                "support": int(report["1"]["support"]),
            },
            "normal_class_0": {
                "precision": round(float(report["0"]["precision"]), 4),
                "recall": round(float(report["0"]["recall"]), 4),
                "f1_score": round(float(report["0"]["f1-score"]), 4),
                "support": int(report["0"]["support"]),
            },
        },
    }


def evaluate_dummy_baseline(
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Computes performance of a naive Majority-Class Baseline.
    Serves as the empirical floor: any ML model must significantly outperform this baseline.
    """
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    dummy_probs = dummy.predict_proba(np.zeros((len(y_test), 1)))[:, 1]

    return compute_comprehensive_evaluation_metrics(
        y_true=y_test.values,
        y_probs=dummy_probs,
        threshold=0.5,
        model_name="Majority Class Baseline (Always Normal)",
    )
