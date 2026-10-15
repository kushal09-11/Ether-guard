# Final Pre-Training Audit & Methodological Defense (ChainSentinel)

**Project**: Ethereum Wallet Fraud Detection Using Machine Learning  
**Stage**: Pre-Training Audit & Verification (Phase 1 → Phase 2 Transition)  
**Date**: October 2026  
**Auditor**: Senior Machine Learning & Blockchain Infrastructure Team  
**Status**: Completed — Ready for Phase 2 Evaluation Review  

---

## Executive Summary

This document presents the **Final Pre-Training Audit** for the ChainSentinel fraud detection system. Before committing to Phase 2 model training, we performed an uncompromising empirical review of the Kaggle Ethereum Fraud Detection Dataset, its correspondence with Etherscan API V2 endpoints, its unrecorded failure modes, and the experimental protocol.

Our investigation confirms that:
1. **Zero features can be certified as "Verified Equivalent"** without unproven assumptions, because the Kaggle dataset discarded raw transaction hashes, receipts, and error flags. 15 features are **Not Verifiable from the available dataset**, and 9 are **Approximate Equivalents**.
2. **Failed transaction handling (`isError == '1'`) is an unresolved methodological uncertainty** that cannot be settled from the CSV alone.
3. **The 829 rows with missing ERC-20 values represent an empirical scraper divergence** ($0.0$ recorded for normal wallets, `NaN` for fraud wallets). Zero-imputation is a defensible modeling assumption, but must **never** be claimed as confirmed on-chain truth without live block verification.
4. **Three algebraic redundancies and multicollinearities exist** within the 24 proposed features:
   - `total_transactions_count` is an exact collinear sum.
   - `total_ether_balance` is an exact collinear difference.
   - `erc20_unique_received_contract_addr` and `erc20_unique_received_token_name` are virtually duplicate ($r = 0.9996$).
5. **The evaluation protocol has been formalized** in `src/evaluation_protocol.py`: Stratified 70/15/15 Train/Validation/Test splitting, train-only transformation fitting, validation-only threshold tuning, and multi-metric reporting (PR-AUC, ROC-AUC, F1, Precision, Recall, Brier score, and Dummy Baseline). Accuracy is strictly barred from being reported in isolation.

---

## 1. Feature Parity Matrix (The 24 Candidate Features)

Features are evaluated against three strict tiers:
- **Verified Equivalent**: Proven 1:1 mathematical, unit, and inclusion identity. *(0 Features)*.
- **Approximate Equivalent**: Structurally compatible concept, but subject to known approximations (algebraic combination, synthetic cash flow, or zero-imputation assumptions). *(9 Features)*.
- **Not Verifiable from Available Dataset**: Identity cannot be verified because Kaggle discarded raw transaction receipts and error status flags. *(15 Features)*.

| # | Canonical Feature Name | Parity Classification | Direction & Unit | Etherscan V2 Source | Unresolved Issue & Parity Risk |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | `avg_min_between_sent_tnx` | **Not verifiable from available dataset** | Outbound (Minutes) | `txlist` | Unverifiable whether failed sent transactions (`isError==1`) were included in timestamp spans. |
| 2 | `avg_min_between_received_tnx` | **Not verifiable from available dataset** | Inbound (Minutes) | `txlist` | Unverifiable whether failed incoming transactions were included; omits internal contract incoming transfers. |
| 3 | `time_diff_first_last_mins` | **Not verifiable from available dataset** | Span (Minutes) | `txlist` | Unverifiable whether earliest/latest timestamps included reverted transactions. |
| 4 | `sent_tnx` | **Not verifiable from available dataset** | Outbound (Count) | `txlist` | Unverifiable whether failed sent transactions are included in count. |
| 5 | `received_tnx` | **Not verifiable from available dataset** | Inbound (Count) | `txlist` | Unverifiable whether failed incoming transactions are included in count. |
| 6 | `created_contracts_count` | **Not verifiable from available dataset** | Outbound (Count) | `txlist` | Unverifiable whether failed deployment transactions (`contractAddress==""`) were counted. |
| 7 | `unique_received_from_addresses` | **Not verifiable from available dataset** | Inbound (Count) | `txlist` | Unverifiable whether senders of failed incoming transactions were added to address set. |
| 8 | `unique_sent_to_addresses` | **Not verifiable from available dataset** | Outbound (Count) | `txlist` | Unverifiable whether recipients of failed outgoing transactions were added to address set. |
| 9 | `min_value_received_eth` | **Not verifiable from available dataset** | Inbound (ETH) | `txlist` | In EVM, failed transfers revert, but attempted value appears in `txlist`. Unverifiable if filtered. |
| 10 | `max_value_received_eth` | **Not verifiable from available dataset** | Inbound (ETH) | `txlist` | Unverifiable whether attempted value of failed transactions was filtered. |
| 11 | `avg_value_received_eth` | **Not verifiable from available dataset** | Inbound (ETH) | `txlist` | Unverifiable whether failed transaction values were included in denominator and numerator. |
| 12 | `min_value_sent_eth` | **Not verifiable from available dataset** | Outbound (ETH) | `txlist` | Unverifiable whether failed transaction values were included. |
| 13 | `max_value_sent_eth` | **Not verifiable from available dataset** | Outbound (ETH) | `txlist` | Unverifiable whether failed transaction values were included. |
| 14 | `avg_value_sent_eth` | **Not verifiable from available dataset** | Outbound (ETH) | `txlist` | Unverifiable whether failed transaction values were included. |
| 15 | `total_transactions_count` | **Approximate equivalent** | Total (Count) | `txlist` | Exactly equals `sent_tnx + received_tnx + created_contracts_count`. Inherits failed tx uncertainty. |
| 16 | `total_ether_sent` | **Not verifiable from available dataset** | Outbound (ETH) | `txlist` | Unverifiable whether attempted ETH in failed transactions was summed. |
| 17 | `total_ether_received` | **Approximate equivalent** | Inbound (ETH) | `txlist` | Omits internal transactions (DEX swaps, contract payouts, mining). Failed tx inclusion unverifiable. |
| 18 | `total_ether_balance` | **Approximate equivalent** | Net Flow (ETH) | `txlist` (Formula) | Synthetic formula (`received - sent`). Cannot match EVM state balance. Contains 978 negative balances. |
| 19 | `erc20_total_tnx` | **Approximate equivalent** | Total (Count) | `tokentx` | `tokentx` logs emit only on successful execution. However, 829 fraud rows were NaN in training data. |
| 20 | `erc20_unique_sent_addr` | **Approximate equivalent** | Outbound (Count) | `tokentx` | Imputation assumption on 829 training rows. |
| 21 | `erc20_unique_received_addr` | **Approximate equivalent** | Inbound (Count) | `tokentx` | Imputation assumption on 829 training rows. |
| 22 | `erc20_unique_received_contract_addr` | **Approximate equivalent** | Inbound (Count) | `tokentx` | Imputation assumption on 829 rows; highly collinear with token name count ($r = 0.9996$). |
| 23 | `erc20_unique_sent_token_name` | **Approximate equivalent** | Outbound (Count) | `tokentx` | Imputation assumption on 829 training rows. |
| 24 | `erc20_unique_received_token_name` | **Approximate equivalent** | Inbound (Count) | `tokentx` | Imputation assumption on 829 training rows. |

---

## 2. Failed Transactions: Detailed Investigation & Unresolved Status

### 2.1 The Technical Mechanism of Failed Transactions
In Ethereum execution frames:
- When a transaction reverts (e.g., `REVERT`, `INVALID`, out of gas), the consensus state reverts value transfers and contract creations.
- Etherscan's `txlist` endpoint records these failed attempts with `isError == "1"` and `txreceipt_status == "0"`. The `value` field still reflects the *attempted* transfer amount in Wei, and `timeStamp` reflects when the failed block was mined.

### 2.2 Empirical Dataset Findings
We performed an exhaustive text and column audit on `data/raw/transaction_dataset.csv`.
- **Finding**: There is no column recording transaction receipt status, error flags, gas used, or failed counts.
- **Finding**: Raw transaction hashes (`hash`) were discarded when the tabular summary was compiled.
- **Unresolved Determination**: **It is impossible to establish from the Kaggle dataset whether the original author filtered out failed transactions (`isError == '0'`) or included all transactions indiscriminately.**

### 2.3 Recommended Mitigation for Phase 3
In our reference calculation engine (`src/feature_calculator.py`), we explicitly set:
```python
successful_normal_txs = [
    tx for tx in normal_transactions
    if str(tx.get("isError", "0")).strip() == "0"
]
```
To prevent silent assumption bias, the live extractor in Phase 3 must expose a configuration parameter `filter_failed_transactions: bool = True` and support ablation testing.

---

## 3. Reassessment of ERC-20 Missingness

### 3.1 Empirical Evidence of Scraper Divergence
We audited the relationship between target class and ERC-20 missingness:
- **Class 0 (Normal Wallets, 7,662 rows)**:
  - Wallets with `Total ERC20 tnxs == 0.0`: **4,399 (57.41%)**
  - Wallets with `Total ERC20 tnxs > 0`: **3,263 (42.59%)**
  - Wallets with missing ERC-20: **0 (0.0%)**
- **Class 1 (Fraud Wallets, 2,179 rows)**:
  - Wallets with `Total ERC20 tnxs == 0.0`: **0 (0.0%)**
  - Wallets with `Total ERC20 tnxs > 0`: **1,350 (61.95%)**
  - Wallets with missing ERC-20: **829 (38.05%)**

### 3.2 Methodological Evaluation
1. **Scraper Asymmetry**: The numerical identity $1,350 + 829 = 2,179$ demonstrates that the scraper used two distinct parsing routines: when `tokentx` returned an empty response (`[]`), it assigned `0.0` for normal accounts and `NaN` for fraud accounts.
2. **Lack of Live Ground Truth**: We **do not** claim these 829 wallets are verified on-chain to have zero token transfers. It is plausible that some fraud queries failed due to Etherscan rate limits (HTTP 429), timeouts, or API quota exhaustion.
3. **Defense of Zero Imputation**: Zero-imputation (`fillna(0.0)`) is a **pragmatic modeling assumption** that aligns fraud records with the 4,399 normal wallets that conducted zero token transactions. It prevents the model from exploiting the scraper's collection bug.
4. **Strict Anti-Leakage Rule**: **An `is_missing_erc20` indicator feature is strictly forbidden.** Adding such a feature would leak the target label with 100% precision on 829 records.

---

## 4. Redundancy & Multicollinearity Audit

Our correlation and algebraic dependency audit revealed three key structural issues within the 24 proposed features:

```
+---------------------------------------------------------------------------------------------------+
|                              STRUCTURAL REDUNDANCIES IDENTIFIED                                   |
+------------------------------------+-----------------------------------+--------------------------+
| Dependent Feature                  | Deterministic Formula / Counterpart| Pearson Correlation (r)  |
+------------------------------------+-----------------------------------+--------------------------+
| total_transactions_count           | sent_tnx + received_tnx + created | r = 1.0000 (Exact sum)   |
| total_ether_balance                | total_ether_received - total_sent | r = 1.0000 (Exact diff)  |
| erc20_unique_received_contract_addr| erc20_unique_received_token_name  | r = 0.9996 (Near-dup)    |
+------------------------------------+-----------------------------------+--------------------------+
```

### Modeling Implications for Phase 2:
- **Tree-based Models (Random Forest, XGBoost, LightGBM)**: Relatively robust to collinearity, but split importance will be split arbitrarily between collinear pairs.
- **Linear Models (Logistic Regression)**: Collinearity will cause matrix instability. ElasticNet or L2 regularization (`C=1.0`) is required, or one feature from each redundant pair must be dropped during feature selection.
- **Target Independence**: The target `FLAG` and identifiers (`Address`, `Index`, `Unnamed: 0`) are strictly quarantined from the feature matrix $X$.

---

## 5. Formal Evaluation Protocol (ChainSentinel)

To avoid optimistic bias, data snooping, and misleading accuracy metrics, we established `src/evaluation_protocol.py`:

```
[Full Dataset: 9,816 Clean Records]
       │
       ▼ (Stratified Split, random_state=42)
       ├── 70% Train Set (6,871 records; 22.20% Fraud) ──► Fit Scalers & Train Models
       ├── 15% Validation Set (1,472 records; 22.20% Fraud) ──► Hyperparameter Tuning & Threshold Optimization
       └── 15% Test Set (1,473 records; 22.20% Fraud) ──► STRICTLY HELD OUT until final evaluation
```

### Strict Evaluation Rules:
1. **Fit on Train Only**: All scalers (`RobustScaler`, `StandardScaler`) and imputers must be fitted solely on `X_train`. Calling `fit_transform` across the combined dataset is strictly prohibited.
2. **Threshold Tuning on Validation Only**: The decision threshold will be tuned on the validation set to maximize PR-AUC / F1-score. The tuned threshold is then applied once to the test set.
3. **Multi-Metric Suite (No Accuracy Alone)**:
   - **Threshold-Independent**: PR-AUC (Average Precision), ROC-AUC, Brier Score (Calibration).
   - **Threshold-Dependent**: Precision, Recall, F1-Score for Class 1 (Fraud), Class 0 (Normal), and Macro Average.
   - **Confusion Matrix**: Full breakdown of TP, FP, TN, FN.
4. **Empirical Baseline**: Every model must be benchmarked against a `DummyClassifier(strategy='most_frequent')` baseline.

---

## 6. Documented Academic Limitations

1. **Unverified Real-World Provenance of Labels**:
   - The labels (`FLAG = 0` and `FLAG = 1`) are **dataset-provided annotations** derived from early community reports (ScamDB, Etherscan public tags).
   - Their real-world cryptographic provenance and forensic reliability have not been independently audited against raw block traces.
2. **Model Scores Are NOT Calibrated Probabilities**:
   - Raw output scores from classifiers (e.g. `model.predict_proba`) represent ranking scores, **not calibrated Bayesian fraud probabilities**.
   - Model outputs must not be presented as true risk probabilities unless evaluated using Brier score and calibrated via Platt scaling or isotonic regression on the validation set.
3. **Temporal Drift**:
   - The dataset represents Ethereum transactions up to circa 2020. Modern Layer-2 rollups, MEV bot transactions, and complex DeFi interactions are unrepresented.

---

## 7. Pre-Training Readiness Verdict

### Summary of Parity & Readiness:
- **Parity Tier Summary**: 0 Verified Equivalent, 9 Approximate Equivalent, 15 Not Verifiable from Available Dataset.
- **Test Suite Status**: **27 passed, 0 failed** across all edge cases, manual calculations, and evaluation split integrity.
- **Unresolved Blockers**: The inclusion of failed transactions remains an inherent limitation of the Kaggle tabular dataset, mitigated by explicit documentation and configurable inference semantics.

**Verdict**: The dataset, preprocessing pipeline, and 24-feature schema are **methodologically audited and ready for Phase 2 experimental model training**, subject to the strict evaluation protocol defined herein.
