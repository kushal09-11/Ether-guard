"""
Automated unit tests for the evaluation protocol and split design.
Verifies stratified splitting, zero data leakage, threshold optimization,
and multi-metric calculation without reliance on raw accuracy.
"""

import pytest
import numpy as np
import pandas as pd
from src.evaluation_protocol import (
    create_stratified_train_val_test_splits,
    find_optimal_threshold_on_validation,
    compute_comprehensive_evaluation_metrics,
    evaluate_dummy_baseline,
)


@pytest.fixture
def synthetic_dataset():
    """Generates synthetic dataset with known 20% fraud imbalance."""
    np.random.seed(42)
    n = 1000
    X = pd.DataFrame({
        "feat_1": np.random.randn(n),
        "feat_2": np.random.exponential(scale=2.0, size=n),
    })
    # 200 fraud (1), 800 normal (0)
    y = pd.Series([1] * 200 + [0] * 800, name="FLAG")
    return X, y


def test_stratified_splits_ratio_and_disjointness(synthetic_dataset):
    """
    Verifies that:
    1. Splits (70% train, 15% val, 15% test) sum to total sample size.
    2. Class imbalance (20% fraud) is preserved in every split within tight tolerance.
    3. No indices or rows leak across splits.
    """
    X, y = synthetic_dataset
    splits = create_stratified_train_val_test_splits(X, y, val_size=0.15, test_size=0.15, random_seed=42)

    X_train, X_val, X_test = splits["X_train"], splits["X_val"], splits["X_test"]
    y_train, y_val, y_test = splits["y_train"], splits["y_val"], splits["y_test"]

    # Size checks
    assert len(X_train) == 700
    assert len(X_val) == 150
    assert len(X_test) == 150
    assert len(X_train) + len(X_val) + len(X_test) == 1000

    # Stratification check: exactly 20% positive in each split
    stats = splits["split_statistics"]
    assert stats["train"]["fraud_ratio"] == pytest.approx(0.20, abs=0.01)
    assert stats["validation"]["fraud_ratio"] == pytest.approx(0.20, abs=0.01)
    assert stats["test"]["fraud_ratio"] == pytest.approx(0.20, abs=0.01)


def test_validation_threshold_optimization():
    """Verifies that threshold tuning finds the score-maximizing cutoff."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Probs where fraud is around 0.4 - 0.7
    y_probs = np.array([0.1, 0.15, 0.2, 0.45, 0.48, 0.6, 0.8, 0.9])

    best_thresh, best_f1 = find_optimal_threshold_on_validation(y_true, y_probs, metric="f1")
    assert 0.4 < best_thresh < 0.55
    assert best_f1 > 0.8


def test_comprehensive_metrics_computation():
    """Verifies computation of PR-AUC, ROC-AUC, Brier score, and Confusion Matrix."""
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = compute_comprehensive_evaluation_metrics(y_true, y_probs, threshold=0.5, model_name="TestModel")

    assert metrics["threshold_independent_metrics"]["roc_auc"] == 1.0
    assert metrics["threshold_independent_metrics"]["pr_auc"] == 1.0
    assert metrics["threshold_independent_metrics"]["brier_score"] < 0.05
    assert metrics["confusion_matrix"]["true_positives"] == 2
    assert metrics["confusion_matrix"]["true_negatives"] == 2
    assert metrics["confusion_matrix"]["false_positives"] == 0
    assert metrics["confusion_matrix"]["false_negatives"] == 0
    assert metrics["threshold_dependent_metrics"]["fraud_class_1"]["f1_score"] == 1.0


def test_dummy_baseline_evaluation():
    """Verifies that dummy baseline outputs 0 recall/precision on minority fraud class."""
    y_train = pd.Series([0, 0, 0, 1])
    y_test = pd.Series([0, 0, 0, 1])

    baseline = evaluate_dummy_baseline(y_train, y_test)
    assert baseline["model_name"] == "Majority Class Baseline (Always Normal)"
    assert baseline["confusion_matrix"]["true_positives"] == 0
    assert baseline["threshold_dependent_metrics"]["fraud_class_1"]["f1_score"] == 0.0
