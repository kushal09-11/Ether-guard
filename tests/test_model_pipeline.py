"""
Unit tests for Model Pipeline construction, serialization, and evaluation.
Validates requirement 6: Unit tests for the evaluation and model-loading pipeline.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

import config
from src.model_pipeline import (
    get_model_pipeline,
    evaluate_pipeline_on_splits,
    extract_feature_importance,
    save_model_pipeline,
    load_model_pipeline,
)


def test_get_model_pipeline_valid_types():
    """Verify pipelines are correctly created for all supported model types."""
    valid_types = ["dummy", "logistic_regression", "decision_tree", "random_forest", "gradient_boosting"]
    for mtype in valid_types:
        pipeline = get_model_pipeline(mtype)
        assert isinstance(pipeline, Pipeline), f"Failed to instantiate Pipeline for {mtype}"
        assert "classifier" in pipeline.named_steps


def test_get_model_pipeline_scaler_presence():
    """Verify that linear models include a scaler while tree models do not require one."""
    lr_pipeline = get_model_pipeline("logistic_regression")
    assert "scaler" in lr_pipeline.named_steps

    rf_pipeline = get_model_pipeline("random_forest")
    assert "scaler" not in rf_pipeline.named_steps


def test_get_model_pipeline_invalid_type():
    """Verify that an unsupported model type raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported model_type"):
        get_model_pipeline("unsupported_neural_net")


def test_evaluate_pipeline_on_synthetic_splits():
    """Test full training, threshold tuning, and split evaluation on a small synthetic dataset."""
    rng = np.random.RandomState(42)
    n_features = 5
    feature_cols = [f"feat_{i}" for i in range(n_features)]

    X_train = pd.DataFrame(rng.randn(100, n_features), columns=feature_cols)
    y_train = pd.Series(rng.choice([0, 1], size=100, p=[0.8, 0.2]))

    X_val = pd.DataFrame(rng.randn(30, n_features), columns=feature_cols)
    y_val = pd.Series(rng.choice([0, 1], size=30, p=[0.8, 0.2]))

    X_test = pd.DataFrame(rng.randn(30, n_features), columns=feature_cols)
    y_test = pd.Series(rng.choice([0, 1], size=30, p=[0.8, 0.2]))

    pipeline = get_model_pipeline("logistic_regression")
    result = evaluate_pipeline_on_splits(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        model_name="TestLR",
        feature_set_name="Synthetic 5",
    )

    assert result["model_name"] == "TestLR"
    assert 0.05 <= result["selected_threshold"] <= 0.95
    assert "test_metrics" in result
    assert "validation_metrics" in result
    test_metrics = result["test_metrics"]
    assert "brier_score" in test_metrics["threshold_independent_metrics"]
    assert "pr_auc" in test_metrics["threshold_independent_metrics"]
    assert "roc_auc" in test_metrics["threshold_independent_metrics"]
    assert "fraud_class_1" in test_metrics["threshold_dependent_metrics"]
    assert "f1_score" in test_metrics["threshold_dependent_metrics"]["fraud_class_1"]


def test_save_and_load_model_pipeline(tmp_path: Path):
    """Verify serialization and deserialization consistency."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(50, 4), columns=["a", "b", "c", "d"])
    y = pd.Series(rng.choice([0, 1], size=50))

    pipeline = get_model_pipeline("logistic_regression")
    pipeline.fit(X, y)
    original_probs = pipeline.predict_proba(X)[:, 1]

    save_path = tmp_path / "test_pipeline.joblib"
    save_model_pipeline(pipeline, save_path)
    assert save_path.exists()

    loaded_pipeline = load_model_pipeline(save_path)
    loaded_probs = loaded_pipeline.predict_proba(X)[:, 1]

    np.testing.assert_allclose(original_probs, loaded_probs, atol=1e-6)


def test_load_nonexistent_model_raises_error():
    """Verify that attempting to load a non-existent model file raises FileNotFoundError."""
    missing_path = Path("models/nonexistent_dummy_model.joblib")
    with pytest.raises(FileNotFoundError):
        load_model_pipeline(missing_path)


def test_saved_model_artifacts_exist_and_loadable():
    """Verify that Phase 2 saved model artifacts exist and can be deserialized properly."""
    best_model_path = config.MODELS_DIR / "best_model_pipeline.joblib"
    rf_model_path = config.MODELS_DIR / "random_forest_canonical_24.joblib"

    assert best_model_path.exists(), f"Best model artifact missing at {best_model_path}"
    assert rf_model_path.exists(), f"Random Forest artifact missing at {rf_model_path}"

    best_pipeline = load_model_pipeline(best_model_path)
    rf_pipeline = load_model_pipeline(rf_model_path)

    assert isinstance(best_pipeline, Pipeline)
    assert isinstance(rf_pipeline, Pipeline)
    assert hasattr(best_pipeline, "predict_proba")
    assert hasattr(rf_pipeline, "predict_proba")


def test_best_model_pipeline_smoke_test_predictions():
    """
    Smoke test: Load best_model_pipeline.joblib, verify its expected feature names and order,
    and assert inference succeeds on a valid single-wallet sample with the exact 24-feature schema.
    """
    best_model_path = config.MODELS_DIR / "best_model_pipeline.joblib"
    pipeline = load_model_pipeline(best_model_path)
    classifier = pipeline.named_steps["classifier"]

    # 1. Verify expected feature names and exact ordering
    assert hasattr(classifier, "feature_names_in_")
    actual_features = list(classifier.feature_names_in_)
    assert actual_features == config.CANONICAL_24_FEATURES, (
        f"Feature schema mismatch! Expected {config.CANONICAL_24_FEATURES}, got {actual_features}"
    )

    # 2. Construct a valid single-wallet sample with realistic feature values
    valid_sample = pd.DataFrame([{
        "avg_min_between_sent_tnx": 120.5,
        "avg_min_between_received_tnx": 84.2,
        "time_diff_first_last_mins": 525600.0,  # 1 year
        "sent_tnx": 15,
        "received_tnx": 25,
        "created_contracts_count": 0,
        "unique_received_from_addresses": 18,
        "unique_sent_to_addresses": 10,
        "min_value_received_eth": 0.05,
        "max_value_received_eth": 10.0,
        "avg_value_received_eth": 1.2,
        "min_value_sent_eth": 0.01,
        "max_value_sent_eth": 5.0,
        "avg_value_sent_eth": 0.8,
        "total_transactions_count": 40,
        "total_ether_sent": 12.0,
        "total_ether_received": 30.0,
        "total_ether_balance": 18.0,
        "erc20_total_tnx": 5,
        "erc20_unique_sent_addr": 2,
        "erc20_unique_received_addr": 3,
        "erc20_unique_received_contract_addr": 3,
        "erc20_unique_sent_token_name": 1,
        "erc20_unique_received_token_name": 2,
    }], columns=config.CANONICAL_24_FEATURES)

    # 3. Predict probabilities and class
    probs = pipeline.predict_proba(valid_sample)
    assert probs.shape == (1, 2), f"Expected shape (1, 2), got {probs.shape}"
    assert np.isclose(probs.sum(), 1.0, atol=1e-5), "Probabilities must sum to 1.0"
    assert 0.0 <= probs[0, 1] <= 1.0, "Fraud probability must be in [0, 1]"

    # 4. Check thresholded decision
    tuned_threshold = 0.51
    decision = int(probs[0, 1] >= tuned_threshold)
    assert decision in (0, 1)


def test_best_model_pipeline_schema_mismatch_raises_error():
    """Verify that omitting a required feature raises an error during inference."""
    best_model_path = config.MODELS_DIR / "best_model_pipeline.joblib"
    pipeline = load_model_pipeline(best_model_path)

    # DataFrame missing 'time_diff_first_last_mins'
    incomplete_features = [f for f in config.CANONICAL_24_FEATURES if f != "time_diff_first_last_mins"]
    bad_sample = pd.DataFrame([np.zeros(len(incomplete_features))], columns=incomplete_features)

    with pytest.raises(ValueError, match="The feature names should match those that were passed during fit"):
        pipeline.predict_proba(bad_sample)


def test_confusion_matrix_sum_and_metric_agreement():
    """
    Validates Requirement 6 (Phase 2.2):
    Checks that the confusion-matrix counts sum to the test-set size,
    and verifies that precision, recall, and F1 mathematically agree with the confusion matrix.
    """
    from src.authoritative_eval import run_authoritative_evaluation

    res = run_authoritative_evaluation()

    cm = res["confusion_matrix"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]

    # 1. Check counts sum to test set size
    assert tn + fp + fn + tp == res["test_records"] == 1473
    assert tn == 1131
    assert fp == 15
    assert fn == 41
    assert tp == 286

    # 2. Mathematical agreement of precision, recall, and F1
    expected_prec = tp / (tp + fp)
    expected_rec = tp / (tp + fn)
    expected_f1 = 2 * (expected_prec * expected_rec) / (expected_prec + expected_rec)

    metrics = res["metrics"]
    assert np.isclose(metrics["fraud_precision"], round(expected_prec, 4), atol=1e-4)
    assert np.isclose(metrics["fraud_recall"], round(expected_rec, 4), atol=1e-4)
    assert np.isclose(metrics["fraud_f1"], round(expected_f1, 4), atol=1e-4)


