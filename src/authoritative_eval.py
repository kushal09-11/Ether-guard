"""
Authoritative Evaluation & Discrepancy Verification Script (ChainSentinel Phase 2.2).
Loads the saved best model pipeline, reconstructs the exact stratified splits,
evaluates out-of-sample test metrics at the frozen threshold,
and verifies mathematical consistency between the confusion matrix and classification metrics.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)
import joblib

import config
from src.evaluation_protocol import create_stratified_train_val_test_splits


def run_authoritative_evaluation() -> Dict[str, Any]:
    """
    Executes authoritative, leak-free evaluation of best_model_pipeline.joblib.
    """
    # 1. Load cleaned dataset
    cleaned_path = config.PROCESSED_DATASET_PATH
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned dataset missing: {cleaned_path}")

    df = pd.read_csv(cleaned_path)
    X = df[config.CANONICAL_24_FEATURES]
    y = df["flag"]

    # 2. Reconstruct exact stratified splits
    splits = create_stratified_train_val_test_splits(
        X=X,
        y=y,
        val_size=0.15,
        test_size=0.15,
        random_seed=config.RANDOM_SEED,
    )
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    # 3. Load saved best model artifact
    model_path = config.MODELS_DIR / "best_model_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Best model artifact missing: {model_path}")
    pipeline = joblib.load(model_path)

    # 4. Probabilities
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    test_probs = pipeline.predict_proba(X_test)[:, 1]

    # Frozen threshold tuned strictly on validation
    frozen_threshold = 0.51
    test_preds = (test_probs >= frozen_threshold).astype(int)

    # 5. Confusion Matrix
    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()

    # Sum check
    assert tn + fp + fn + tp == len(y_test) == 1473, "Confusion matrix sum does not equal test set size!"

    # 6. Mathematical verification of precision, recall, and F1
    manual_precision = tp / (tp + fp)
    manual_recall = tp / (tp + fn)
    manual_f1 = 2 * (manual_precision * manual_recall) / (manual_precision + manual_recall)

    sklearn_precision = precision_score(y_test, test_preds, pos_label=1)
    sklearn_recall = recall_score(y_test, test_preds, pos_label=1)
    sklearn_f1 = f1_score(y_test, test_preds, pos_label=1)

    assert np.isclose(manual_precision, sklearn_precision, atol=1e-5), "Precision calculation mismatch!"
    assert np.isclose(manual_recall, sklearn_recall, atol=1e-5), "Recall calculation mismatch!"
    assert np.isclose(manual_f1, sklearn_f1, atol=1e-5), "F1 calculation mismatch!"

    # Normal class (0) metrics
    prec_0 = tn / (tn + fn)
    rec_0 = tn / (tn + fp)
    f1_0 = 2 * (prec_0 * rec_0) / (prec_0 + rec_0)
    manual_macro_f1 = (manual_f1 + f1_0) / 2.0
    sklearn_macro_f1 = f1_score(y_test, test_preds, average="macro")
    assert np.isclose(manual_macro_f1, sklearn_macro_f1, atol=1e-5), "Macro F1 calculation mismatch!"

    # 7. Probability-based metrics
    pr_auc = average_precision_score(y_test, test_probs)
    roc_auc = roc_auc_score(y_test, test_probs)
    brier = brier_score_loss(y_test, test_probs)

    # Discrepancy diagnosis: Evaluate validation at t=0.41
    val_preds_41 = (val_probs >= 0.41).astype(int)
    cm_val_41 = confusion_matrix(y_val, val_preds_41).ravel()
    val_f1_41 = f1_score(y_val, val_preds_41, pos_label=1)

    return {
        "dataset_records": len(df),
        "test_records": len(y_test),
        "frozen_threshold": frozen_threshold,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "metrics": {
            "fraud_precision": round(float(manual_precision), 4),
            "fraud_recall": round(float(manual_recall), 4),
            "fraud_f1": round(float(manual_f1), 4),
            "macro_f1": round(float(manual_macro_f1), 4),
            "pr_auc": round(float(pr_auc), 4),
            "roc_auc": round(float(roc_auc), 4),
            "brier_score": round(float(brier), 4),
        },
        "discrepancy_diagnosis": {
            "validation_t_0_41_cm": {
                "tn": int(cm_val_41[0]),
                "fp": int(cm_val_41[1]),
                "fn": int(cm_val_41[2]),
                "tp": int(cm_val_41[3]),
            },
            "validation_t_0_41_f1": round(float(val_f1_41), 4),
            "explanation": (
                "The original Phase 2 summary text mistakenly cited the validation split confusion matrix at t=0.41 "
                "(TN=1119, FP=27, FN=31, TP=296) because its validation F1 score (0.9108) was identical to the test F1 score at t=0.51. "
                "The authoritative held-out test evaluation at frozen threshold 0.51 is TN=1131, FP=15, FN=41, TP=286."
            ),
        },
    }


if __name__ == "__main__":
    result = run_authoritative_evaluation()
    print("=" * 65)
    print("CHAINSENTINEL: AUTHORITATIVE TEST EVALUATION RESULTS")
    print("=" * 65)
    print(f"Test Set Size: {result['test_records']} (Normal={result['confusion_matrix']['tn'] + result['confusion_matrix']['fp']}, Fraud={result['confusion_matrix']['tp'] + result['confusion_matrix']['fn']})")
    print(f"Threshold: {result['frozen_threshold']}")
    print(f"Confusion Matrix: TN={result['confusion_matrix']['tn']}, FP={result['confusion_matrix']['fp']}, FN={result['confusion_matrix']['fn']}, TP={result['confusion_matrix']['tp']}")
    print(f"Fraud Precision:  {result['metrics']['fraud_precision']:.4f}")
    print(f"Fraud Recall:     {result['metrics']['fraud_recall']:.4f}")
    print(f"Fraud F1 Score:   {result['metrics']['fraud_f1']:.4f}")
    print(f"Macro F1 Score:   {result['metrics']['macro_f1']:.4f}")
    print(f"PR-AUC:           {result['metrics']['pr_auc']:.4f}")
    print(f"ROC-AUC:          {result['metrics']['roc_auc']:.4f}")
    print(f"Brier Score:      {result['metrics']['brier_score']:.4f}")
    print("-" * 65)
    print("DISCREPANCY ROOT CAUSE:")
    print(result["discrepancy_diagnosis"]["explanation"])
    print(f"Validation @ t=0.41 CM: {result['discrepancy_diagnosis']['validation_t_0_41_cm']}")
    print(f"Validation @ t=0.41 F1: {result['discrepancy_diagnosis']['validation_t_0_41_f1']:.4f}")
    print("=" * 65)
