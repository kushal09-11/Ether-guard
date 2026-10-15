# ChainSentinel — Phase 2: Model Training and Evaluation Report
## Comparative Machine Learning Benchmark & Feature Set Ablation Study

**Date**: October 2026  
**Context**: Academic Machine Learning Case Study — Pre-production Empirical Verification  
**Random Seed**: `42` (Strictly reproducible across all splits and pipelines)  
**Selected Best Architecture**: **HistGradientBoosting** on **Canonical 24 Features**  

> [!IMPORTANT]
> **Academic Disclaimer**: This evaluation measures model performance strictly against the **dataset-provided labels** in the Kaggle Ethereum Fraud Detection Dataset. These results do not constitute proof of real-world live fraud detection accuracy on current mainnet wallets.

---
## 1. Experimental Protocol & Data Splitting

The evaluation strictly follows the stratified 70/15/15 protocol defined in Phase 1:
- **Total Cleaned Records**: 9,816 unique wallet addresses (zero duplicate addresses crossing splits).
- **Training Split (70%)**: 6,870 records (1525 fraud, 22.20%). All scalers, imputers, and pipelines were fitted exclusively on this fold.
- **Validation Split (15%)**: 1,473 records (327 fraud, 22.20%). Used exclusively for model selection and decision threshold tuning.
- **Test Split (15%)**: 1,473 records (327 fraud, 22.20%). Strictly held out until final evaluation.

---
## 2. Multi-Model Performance Comparison (Test Set Results)

Below are the unbiased test-set results evaluated at the validation-optimized decision thresholds:

| feature_set                          | model_name                         | selected_threshold | test_pr_auc | test_roc_auc | test_fraud_precision | test_fraud_recall | test_fraud_f1 | test_macro_f1 | test_brier_score |
| ------------------------------------ | ---------------------------------- | ------------------ | ----------- | ------------ | -------------------- | ----------------- | ------------- | ------------- | ---------------- |
| Canonical 24 Features                | Dummy (Majority Baseline)          | 0.01               | 0.222       | 0.5          | 0.0                  | 0.0               | 0.0           | 0.4376        | 0.222            |
| Canonical 24 Features                | Logistic Regression (L2, Balanced) | 0.64               | 0.6059      | 0.8166       | 0.5362               | 0.6789            | 0.5992        | 0.7322        | 0.2108           |
| Canonical 24 Features                | Decision Tree (Max Depth 6)        | 0.84               | 0.8365      | 0.9456       | 0.7975               | 0.7829            | 0.7901        | 0.8655        | 0.0858           |
| Canonical 24 Features                | Random Forest (100 Trees)          | 0.63               | 0.9559      | 0.9846       | 0.9148               | 0.8532            | 0.8829        | 0.9255        | 0.05             |
| Canonical 24 Features                | HistGradientBoosting               | 0.51               | 0.9736      | 0.9897       | 0.9502               | 0.8746            | 0.9108        | 0.9433        | 0.0283           |
| Reduced 21 Features (No Redundancy)  | Dummy (Majority Baseline)          | 0.01               | 0.222       | 0.5          | 0.0                  | 0.0               | 0.0           | 0.4376        | 0.222            |
| Reduced 21 Features (No Redundancy)  | Logistic Regression (L2, Balanced) | 0.64               | 0.61        | 0.8161       | 0.5314               | 0.6728            | 0.5938        | 0.7286        | 0.2103           |
| Reduced 21 Features (No Redundancy)  | Decision Tree (Max Depth 6)        | 0.84               | 0.8169      | 0.9486       | 0.8207               | 0.7278            | 0.7715        | 0.8555        | 0.0862           |
| Reduced 21 Features (No Redundancy)  | Random Forest (100 Trees)          | 0.63               | 0.9552      | 0.9842       | 0.8971               | 0.8532            | 0.8746        | 0.92          | 0.0501           |
| Reduced 21 Features (No Redundancy)  | HistGradientBoosting               | 0.44               | 0.9682      | 0.9877       | 0.932                | 0.8807            | 0.9057        | 0.9398        | 0.0315           |
| Ablation 18 Features (Base ETH Only) | Dummy (Majority Baseline)          | 0.01               | 0.222       | 0.5          | 0.0                  | 0.0               | 0.0           | 0.4376        | 0.222            |
| Ablation 18 Features (Base ETH Only) | Logistic Regression (L2, Balanced) | 0.65               | 0.5271      | 0.7824       | 0.5164               | 0.578             | 0.5455        | 0.7028        | 0.2133           |
| Ablation 18 Features (Base ETH Only) | Decision Tree (Max Depth 6)        | 0.78               | 0.8301      | 0.9437       | 0.8292               | 0.7125            | 0.7664        | 0.8529        | 0.0926           |
| Ablation 18 Features (Base ETH Only) | Random Forest (100 Trees)          | 0.64               | 0.9476      | 0.9814       | 0.8839               | 0.8379            | 0.8603        | 0.9109        | 0.0555           |
| Ablation 18 Features (Base ETH Only) | HistGradientBoosting               | 0.47               | 0.9657      | 0.9872       | 0.9161               | 0.8685            | 0.8917        | 0.9309        | 0.0338           |

---
## 3. Confusion Matrix Breakdown on Held-Out Test Set

| feature_set                          | model_name                         | selected_threshold | test_tp | test_fp | test_tn | test_fn |
| ------------------------------------ | ---------------------------------- | ------------------ | ------- | ------- | ------- | ------- |
| Canonical 24 Features                | Dummy (Majority Baseline)          | 0.01               | 0       | 0       | 1146    | 327     |
| Canonical 24 Features                | Logistic Regression (L2, Balanced) | 0.64               | 222     | 192     | 954     | 105     |
| Canonical 24 Features                | Decision Tree (Max Depth 6)        | 0.84               | 256     | 65      | 1081    | 71      |
| Canonical 24 Features                | Random Forest (100 Trees)          | 0.63               | 279     | 26      | 1120    | 48      |
| Canonical 24 Features                | HistGradientBoosting               | 0.51               | 286     | 15      | 1131    | 41      |
| Reduced 21 Features (No Redundancy)  | Dummy (Majority Baseline)          | 0.01               | 0       | 0       | 1146    | 327     |
| Reduced 21 Features (No Redundancy)  | Logistic Regression (L2, Balanced) | 0.64               | 220     | 194     | 952     | 107     |
| Reduced 21 Features (No Redundancy)  | Decision Tree (Max Depth 6)        | 0.84               | 238     | 52      | 1094    | 89      |
| Reduced 21 Features (No Redundancy)  | Random Forest (100 Trees)          | 0.63               | 279     | 32      | 1114    | 48      |
| Reduced 21 Features (No Redundancy)  | HistGradientBoosting               | 0.44               | 288     | 21      | 1125    | 39      |
| Ablation 18 Features (Base ETH Only) | Dummy (Majority Baseline)          | 0.01               | 0       | 0       | 1146    | 327     |
| Ablation 18 Features (Base ETH Only) | Logistic Regression (L2, Balanced) | 0.65               | 189     | 177     | 969     | 138     |
| Ablation 18 Features (Base ETH Only) | Decision Tree (Max Depth 6)        | 0.78               | 233     | 48      | 1098    | 94      |
| Ablation 18 Features (Base ETH Only) | Random Forest (100 Trees)          | 0.64               | 274     | 36      | 1110    | 53      |
| Ablation 18 Features (Base ETH Only) | HistGradientBoosting               | 0.47               | 284     | 26      | 1120    | 43      |

---
## 4. Key Empirical Findings & Ablation Analysis

### 4.1 Comparison Across Feature Sets (Ablation Analysis)
We evaluated three distinct feature representations:
1. **Canonical 24 Features**: The full Etherscan API V2 baseline schema.
2. **Reduced 21 Features**: Eliminates 3 algebraic redundancies (`total_transactions_count`, `total_ether_balance`, `erc20_unique_received_contract_addr`).
3. **Ablation 18 Features (Base ETH Only)**: Completely removes all 6 ERC-20 features to test sensitivity to the 829 missing ERC-20 records.

#### Random Forest Performance Across Feature Sets:
| feature_set                          | selected_threshold | test_pr_auc | test_roc_auc | test_fraud_f1 | test_fraud_recall | test_fraud_precision |
| ------------------------------------ | ------------------ | ----------- | ------------ | ------------- | ----------------- | -------------------- |
| Canonical 24 Features                | 0.63               | 0.9559      | 0.9846       | 0.8829        | 0.8532            | 0.9148               |
| Reduced 21 Features (No Redundancy)  | 0.63               | 0.9552      | 0.9842       | 0.8746        | 0.8532            | 0.8971               |
| Ablation 18 Features (Base ETH Only) | 0.64               | 0.9476      | 0.9814       | 0.8603        | 0.8379            | 0.8839               |

> [!NOTE]
> **Insight on ERC-20 Ablation**: Removing ERC-20 features entirely drops Test PR-AUC and F1 only marginally (Test F1 drops by ~2.2% in Random Forest and ~1.9% in HistGradientBoosting). This confirms that the model derives substantial predictive power from core ETH transaction dynamics (lifespan, sent/received counts, and unique addresses), proving it is NOT merely memorizing the 829 missing ERC-20 fraud records or their zero imputation.

### 4.2 Feature Importances & Dominance of `time_diff_first_last_mins`

Across all evaluated architectures, `time_diff_first_last_mins` emerges as the dominant predictive feature:
- **Logistic Regression**: Normalized $|coef| = 0.1896$ (Rank #1 feature; large negative coefficient indicating shorter lifespan increases fraud risk).
- **Decision Tree (Max Depth 6)**: Gini Importance $= 0.2263$ (Rank #2 feature, behind `total_ether_received`).
- **Random Forest (100 Trees)**: Gini Importance $= 0.1402$ (Rank #1 feature).
- **HistGradientBoosting**: Validation Permutation Importance $= 0.1229$ (Rank #1 feature).

#### Top 10 Feature Importances (Random Forest Canonical 24):
| feature                        | importance          | method               |
| ------------------------------ | ------------------- | -------------------- |
| time_diff_first_last_mins      | 0.1402              | Tree Gini Importance |
| avg_value_received_eth         | 0.0955              | Tree Gini Importance |
| total_ether_received           | 0.0889              | Tree Gini Importance |
| avg_min_between_received_tnx   | 0.0746              | Tree Gini Importance |
| unique_received_from_addresses | 0.0676              | Tree Gini Importance |
| total_ether_sent               | 0.0643              | Tree Gini Importance |
| max_value_received_eth         | 0.0578              | Tree Gini Importance |
| total_ether_balance            | 0.0555              | Tree Gini Importance |
| min_value_received_eth         | 0.0464              | Tree Gini Importance |
| sent_tnx                       | 0.0413              | Tree Gini Importance |

#### Empirical Cause of Dominance:
In the Kaggle training dataset, fraudulent wallets were predominantly disposable "burner" addresses created for specific scam or phishing campaigns, executing rapid transactions and being abandoned shortly thereafter:
- **Normal Wallets (`FLAG=0`)**: Median lifespan = **116,874.32 minutes** (~81.1 days / 2.7 months); Mean = 264,682 minutes (~183.8 days).
- **Fraud Wallets (`FLAG=1`)**: Median lifespan = **7,545.43 minutes** (~5.2 days); Mean = 55,230 minutes (~38.3 days).

#### Critical Vulnerability: Cold-Start False Positives on Legitimate New Wallets
Because models assign elevated risk scores to wallets with small `time_diff_first_last_mins`, **legitimate newly created wallets** with only a few hours of transaction history will inherently possess very small values (e.g., 0 to 120 minutes). In any live-detection setting, this creates an extreme risk of **false-positive misclassification** for brand-new, completely benign users. Live deployments must implement cold-start guardrails (e.g., aging maturity requirements or flagging confidence suppression) rather than evaluating raw model scores on fresh wallets.

---
## 5. Phase 2.1 Verification & Metric Audit

### 5.1 Independent Reproduction of Best Model Test Metrics
Re-evaluating the saved artifact `models/best_model_pipeline.joblib` on the frozen, held-out test split ($N=1,473$) at the validation-selected decision threshold of **0.51** reproduces the reported metrics exactly:
- **Confusion Matrix**: True Negatives (TN) = 1,131, False Positives (FP) = 15, False Negatives (FN) = 41, True Positives (TP) = 286.
- **Precision (Fraud Class 1)**: $\frac{286}{286 + 15} = 0.9502$ (95.02%)
- **Recall (Fraud Class 1)**: $\frac{286}{286 + 41} = 0.8746$ (87.46%)
- **F1 Score (Fraud Class 1)**: $2 \times \frac{0.9502 \times 0.8746}{0.9502 + 0.8746} = 0.9108$ (91.08%)
- **Macro F1**: $0.9433$
- **Precision-Recall AUC (PR-AUC)**: $0.9736$ (Baseline floor: 0.2220)
- **ROC-AUC**: $0.9897$
- **Brier Score**: $0.0283$

### 5.2 Verification of Split Isolation & Anti-Leakage Controls
We verified with zero uncertainty that:
1. **Split Disjointness**: The dataset was partitioned via stratified 70/15/15 sampling ($N_{\text{train}}=6,870$, $N_{\text{val}}=1,473$, $N_{\text{test}}=1,473$). The index sets satisfy $Train \cap Val = \emptyset$, $Train \cap Test = \emptyset$, and $Val \cap Test = \emptyset$.
2. **Address Uniqueness**: The cleaned dataset contains 9,816 distinct Ethereum addresses ($100\%$ uniqueness), ensuring zero cross-split wallet duplication.
3. **No Test-Set Peeking**: The test set was strictly held out. Preprocessing scalers were fitted solely on $X_{\text{train}}$. The decision threshold ($0.51$) was tuned strictly on $X_{\text{val}}$ by optimizing validation F1, and applied frozen to $X_{\text{test}}$. Hyperparameters were fixed a priori.

### 5.3 Correction Regarding Brier Score & Probability Calibration
> [!IMPORTANT]
> **Scientific Correction**:
> Brier score is **not** a ranking metric. Ranking metrics (such as ROC-AUC and PR-AUC) evaluate only the relative ordering of scores and are completely invariant to any strictly monotonic probability transformation. 
> 
> Brier score is a **strictly proper scoring rule** measuring mean squared probabilistic prediction error:
> $$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$
> Under the Murphy (1973) decomposition, $\text{Brier} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$.
> While a Brier score of **0.0283** indicates low mean squared probabilistic error on this test split, **it does not constitute proof that individual probability predictions are calibrated posterior probabilities** of real-world fraud. Proving calibration requires empirical reliability diagrams, Expected Calibration Error (ECE) testing, and post-hoc calibration curves (e.g., isotonic regression or Platt scaling). Outputs must be interpreted strictly as uncalibrated model risk scores.

### 5.4 Inspection of Saved Model Artifact (`models/best_model_pipeline.joblib`)
- **File Location**: `models/best_model_pipeline.joblib` (Size: ~365 KB).
- **Estimator**: `sklearn.ensemble.HistGradientBoostingClassifier(max_depth=6, min_samples_leaf=20, random_state=42)`.
- **Fitted Preprocessing**: Native quantile binning (255 bins) inside the estimator. No external scaling is required or applied.
- **Training Data**: Fitted strictly on $X_{\text{train}}$ (6,870 records, 24 features).
- **Expected Feature Schema**: Exactly 24 canonical features in the following strict order:
  1. `avg_min_between_sent_tnx`
  2. `avg_min_between_received_tnx`
  3. `time_diff_first_last_mins`
  4. `sent_tnx`
  5. `received_tnx`
  6. `created_contracts_count`
  7. `unique_received_from_addresses`
  8. `unique_sent_to_addresses`
  9. `min_value_received_eth`
  10. `max_value_received_eth`
  11. `avg_value_received_eth`
  12. `min_value_sent_eth`
  13. `max_value_sent_eth`
  14. `avg_value_sent_eth`
  15. `total_transactions_count`
  16. `total_ether_sent`
  17. `total_ether_received`
  18. `total_ether_balance`
  19. `erc20_total_tnx`
  20. `erc20_unique_sent_addr`
  21. `erc20_unique_received_addr`
  22. `erc20_unique_received_contract_addr`
  23. `erc20_unique_sent_token_name`
  24. `erc20_unique_received_token_name`
- **Smoke Test Verification**: Automated unit test `tests/test_model_pipeline.py::test_best_model_pipeline_smoke_test_predictions` verifies that inference on a valid sample matching this exact schema succeeds, probabilities sum to $1.0$, and any missing feature triggers a `ValueError`.

### 5.5 Production Retraining Protocol (Preserving Test Isolation)
If this model architecture is later scheduled for deployment:
1. **Model Selection is Complete**: Model family (`HistGradientBoostingClassifier`), hyperparameters (`max_depth=6`, `min_samples_leaf=20`), and operational decision threshold (`0.51`) have been finalized on Validation.
2. **Retraining Strategy**: Concatenate the Training and Validation sets ($X_{\text{train+val}} = X_{\text{train}} \cup X_{\text{val}}$, comprising 8,343 records / 85% of clean data).
3. **Fit Final Pipeline**: Fit `HistGradientBoostingClassifier(max_depth=6, min_samples_leaf=20, random_state=42)` on $X_{\text{train+val}}$ and save as `models/production_model_pipeline.joblib`.
4. **Pristine Test Set**: **The Test split ($N=1,473$, 15%) must remain completely untouched and never pooled into training data.** This guarantees that an uncompromised test set remains available for external audits, drift monitoring, and future model comparisons.

### 5.6 Phase 2.2 Evaluation Discrepancy Resolution & Audit Trail
An audit was conducted to resolve the discrepancy between test-set confusion matrix values reported in early narrative summaries ($TN=1,119, FP=27, FN=31, TP=296$) versus the CSV artifact and Phase 2.1 verification ($TN=1,131, FP=15, FN=41, TP=286$).

#### Root Cause Identification:
An exhaustive grid scan across all splits, models, and decision thresholds confirmed:
1. **The Numbers $TN=1,119, FP=27, FN=31, TP=296$**: Represent the exact confusion matrix evaluated on the **Validation Split** ($X_{\text{val}}, y_{\text{val}}$) at threshold $t=0.41$.
2. **The Numerical Coincidence**: At $t=0.41$ on Validation, the Fraud F1 score is:
   $$\text{Precision} = \frac{296}{323} = 0.9164, \quad \text{Recall} = \frac{296}{327} = 0.9052 \implies \mathbf{F1 = 0.9108}$$
   On the **held-out Test Set** at the validation-optimized threshold of $t=0.51$, the Fraud F1 score is:
   $$\text{Precision} = \frac{286}{301} = 0.9502, \quad \text{Recall} = \frac{286}{327} = 0.8746 \implies \mathbf{F1 = 0.9108}$$
   Because both evaluations yielded an identical F1 score of **0.9108**, the early conversational summary accidentally pulled the confusion matrix and precision/recall pair from the validation threshold sweep at $t=0.41$ instead of the out-of-sample test set at $t=0.51$.

#### Authoritative Test-Set Metrics:
The authoritative evaluation of [`models/best_model_pipeline.joblib`](file:///c:/Users/saiku/OneDrive/Desktop/Coding/Ether-guard/models/best_model_pipeline.joblib) on the held-out test split ($N=1,473$) at frozen threshold $t=0.51$ is:
- **Confusion Matrix**: $TN = 1,131, \quad FP = 15, \quad FN = 41, \quad TP = 286$
- **Total Samples Check**: $1,131 + 15 + 41 + 286 = 1,473$ (1,146 Normal, 327 Fraud)
- **Fraud Precision**: $0.9502$
- **Fraud Recall**: $0.8746$
- **Fraud F1 Score**: $0.9108$
- **Macro F1 Score**: $0.9433$
- **PR-AUC**: $0.9736$
- **ROC-AUC**: $0.9897$
- **Brier Score**: $0.0283$

This authoritative result is recorded in [`reports/phase2_evaluation_results.csv`](file:///c:/Users/saiku/OneDrive/Desktop/Coding/Ether-guard/reports/phase2_evaluation_results.csv) (row 6), verified executable via [`src/authoritative_eval.py`](file:///c:/Users/saiku/OneDrive/Desktop/Coding/Ether-guard/src/authoritative_eval.py), and permanently tested via [`tests/test_model_pipeline.py::test_confusion_matrix_sum_and_metric_agreement`](file:///c:/Users/saiku/OneDrive/Desktop/Coding/Ether-guard/tests/test_model_pipeline.py).

---
## 6. Major Limitations & Scientific Defense

1. **Dataset-Provided Labels**: The fraud labels originate from historical crowdsourced reports (ScamDB, Etherscan community tags). False positives and unconfirmed flags cannot be ruled out without raw cryptographic evidence.
2. **Calibration Caveat**: Model outputs are uncalibrated risk scores. While Brier score ($0.0283$) indicates low mean squared prediction error on this test set, scores should not be called true real-world calibrated probabilities without on-chain calibration curves.
3. **Cold-Start Bias**: The overwhelming reliance on transaction lifespan (`time_diff_first_last_mins`) means new benign wallets will suffer from elevated false-positive risk.
4. **Non-Detection of Modern Fraud**: Modern wash trading, flash loan exploits, cross-chain bridge laundering, and complex smart contract proxy exploits were not present in this circa-2020 dataset.