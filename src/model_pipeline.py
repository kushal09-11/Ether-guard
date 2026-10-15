"""
Model Pipeline & Training Engine for Ethereum Wallet Fraud Detection (ChainSentinel - Phase 2).
Constructs leak-free scikit-learn Pipelines, performs model training on training folds only,
tunes thresholds strictly on validation data, and evaluates test performance.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
import joblib

import config
from src.evaluation_protocol import (
    find_optimal_threshold_on_validation,
    compute_comprehensive_evaluation_metrics,
)


def get_model_pipeline(
    model_type: str,
    random_seed: int = config.RANDOM_SEED,
    class_weight: Optional[str] = "balanced",
) -> Pipeline:
    """
    Constructs an sklearn Pipeline ensuring all transformations
    are fitted strictly during training.
    """
    model_type_clean = model_type.lower().strip()

    if model_type_clean == "dummy":
        return Pipeline([
            ("classifier", DummyClassifier(strategy="most_frequent"))
        ])

    elif model_type_clean in ["logistic_regression", "lr"]:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=0.1,
                class_weight=class_weight,
                max_iter=2500,
                random_state=random_seed,
            )),
        ])

    elif model_type_clean in ["decision_tree", "dt"]:
        return Pipeline([
            ("classifier", DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=20,
                class_weight=class_weight,
                random_state=random_seed,
            )),
        ])

    elif model_type_clean in ["random_forest", "rf"]:
        return Pipeline([
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=10,
                class_weight=class_weight,
                random_state=random_seed,
                n_jobs=-1,
            )),
        ])

    elif model_type_clean in ["gradient_boosting", "hist_gradient_boosting", "hgb"]:
        return Pipeline([
            ("classifier", HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=6,
                min_samples_leaf=20,
                random_state=random_seed,
            )),
        ])

    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. Choose from: "
            f"['dummy', 'logistic_regression', 'decision_tree', 'random_forest', 'gradient_boosting']"
        )


def evaluate_pipeline_on_splits(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    feature_set_name: str = "Canonical 24 Features",
) -> Dict[str, Any]:
    """
    Fits the pipeline strictly on X_train, tunes decision threshold on X_val,
    and performs unbiased final evaluation on X_test.
    """
    # 1. Fit pipeline solely on training data
    pipeline.fit(X_train, y_train)

    # 2. Validation predictions and threshold selection
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    
    # Tune threshold to optimize F1 score on validation set
    best_thresh, best_val_f1 = find_optimal_threshold_on_validation(
        y_val_true=y_val.values,
        y_val_probs=val_probs,
        metric="f1",
    )

    # Validation evaluation at tuned threshold
    val_metrics = compute_comprehensive_evaluation_metrics(
        y_true=y_val.values,
        y_probs=val_probs,
        threshold=best_thresh,
        model_name=f"{model_name} (Validation @ Thresh={best_thresh})",
    )

    # 3. Test evaluation at tuned threshold (unbiased final benchmark)
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = compute_comprehensive_evaluation_metrics(
        y_true=y_test.values,
        y_probs=test_probs,
        threshold=best_thresh,
        model_name=f"{model_name} (Test @ Thresh={best_thresh})",
    )

    # Test evaluation at default 0.5 threshold for comparison
    test_metrics_default_thresh = compute_comprehensive_evaluation_metrics(
        y_true=y_test.values,
        y_probs=test_probs,
        threshold=0.5,
        model_name=f"{model_name} (Test @ Thresh=0.50)",
    )

    return {
        "model_name": model_name,
        "feature_set_name": feature_set_name,
        "feature_count": int(X_train.shape[1]),
        "features": list(X_train.columns),
        "selected_threshold": best_thresh,
        "val_f1_at_selected_threshold": best_val_f1,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_metrics_default_threshold": test_metrics_default_thresh,
        "pipeline": pipeline,
    }


def extract_feature_importance(
    pipeline: Pipeline,
    feature_names: List[str],
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Extracts feature importances or model coefficients.
    For tree models: Gini feature importances.
    For linear models: normalized absolute coefficients.
    For gradient boosting: permutation importance on validation fold.
    """
    classifier = pipeline.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        method = "Tree Gini Importance"
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
        # Normalize sum to 1.0 for comparability
        if importances.sum() > 0:
            importances = importances / importances.sum()
        method = "Normalized |Logistic Regression Coef|"
    elif X_val is not None and y_val is not None:
        perm = permutation_importance(pipeline, X_val, y_val, n_repeats=5, random_state=42)
        importances = np.maximum(0, perm.importances_mean)
        if importances.sum() > 0:
            importances = importances / importances.sum()
        method = "Validation Permutation Importance"
    else:
        importances = np.zeros(len(feature_names))
        method = "None (Dummy)"

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
        "method": method,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return fi_df


def save_model_pipeline(pipeline: Pipeline, file_path: Path) -> None:
    """Serializes a trained pipeline with joblib."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, file_path)


def load_model_pipeline(file_path: Path) -> Pipeline:
    """Loads a serialized pipeline with joblib."""
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found at: {file_path}")
    return joblib.load(file_path)
