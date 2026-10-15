# Dataset Inspection & Data Quality Audit Report
## Ethereum Wallet Fraud Detection Machine Learning Study - Phase 1

**Inspection Target**: `transaction_dataset.csv` (2.75 MB)  
**Dimensions**: 9,841 rows × 51 columns  
**Unique Wallet Addresses**: 9,816  

---
## 1. Verified Target Label & Class Imbalance Analysis

The primary target column was identified and verified as **`FLAG`**.
Verification against Kaggle documentation and the published literature confirms:
- **Class `0`**: Non-fraudulent / standard Ethereum wallet.
- **Class `1`**: Fraudulent / illicit wallet (e.g., phishing, scamming, hack receiver).

| Target Label (FLAG) | Class Meaning | Record Count | Percentage |
| :--- | :--- | :--- | :--- |
| `0` | Non-Fraudulent / Normal Wallet | 7,662 | 77.86% |
| `1` | Fraudulent / Illicit Wallet | 2,179 | 22.14% |

**Class Imbalance Ratio**: Non-Fraud to Fraud is approximately **3.52:1** (moderate imbalance). In Phase 1, no artificial resampling or synthetic rows (e.g., SMOTE) are applied to preserve empirical integrity.

---
## 2. Duplicate Records & Address Collisions

- **Exact Row Duplicates (across all 51 columns)**: 0
- **Feature Duplicates (ignoring Unnamed: 0 and Index)**: 18 rows
- **Address Collisions**: 25 duplicate address rows (representing 25 unique addresses appearing twice).
- **Conflicting Labels on Colliding Addresses**: **0**.
> [!NOTE]
> None of the 25 duplicate addresses possess conflicting `FLAG` values. In 18 cases, all 49 transaction features are 100% identical. In the remaining 7 address pairs (14 rows), all numerical features are identical, and only the arbitrary tie-breaking choice for most frequent token string differs. Deduplication by wallet address preserves one canonical record per wallet without discarding conflicting labels.

---
## 3. Missing Value Audit & Critical Target Leakage Finding

All base Ethereum transaction features (columns 0 through 25) have **0 missing values**.
However, missingness occurs exclusively within ERC-20 token features:
- **Numerical ERC-20 columns**: 829 rows (8.42%) are null.
- **Categorical token columns**: `ERC20 most sent token type` is null for 2,697 rows (27.41%); `ERC20_most_rec_token_type` is null for 871 rows (8.85%).

### Critical Finding: Systematic Target Leakage via Missingness
- **Column**: ` Total ERC20 tnxs`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20 total Ether received`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20 total ether sent`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20 total Ether sent contract`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20 uniq sent addr`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20 uniq rec addr`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20 uniq sent addr.1`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20 uniq rec contract addr`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20 avg time between sent tnx`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20 avg time between rec tnx`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20 avg time between rec 2 tnx`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20 avg time between contract tnx`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20 min val rec`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20 max val rec`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20 avg val rec`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20 min val sent`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20 max val sent`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20 avg val sent`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20 min val sent contract`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20 max val sent contract`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20 avg val sent contract`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20 uniq sent token name`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20 uniq rec token name`
  - **Missing Count**: 829 rows
  - **Fraud Rate in Null Subset**: **100.0%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

- **Column**: ` ERC20_most_rec_token_type`
  - **Missing Count**: 871 rows
  - **Fraud Rate in Null Subset**: **95.18%** (vs. 22.14% in full population).
  - **Diagnostic**: Extreme Target Leakage Risk: Missingness in this column is heavily biased toward fraudulent records (100% fraud in null rows).

> [!WARNING]
> Every single one of the 829 rows with null ERC-20 values belongs to `FLAG = 1` (Fraud). If an ML pipeline naively injects a missingness indicator (e.g. `is_missing_erc20`), the model will achieve deceptive 100% precision on those records solely by exploiting collection artifact leakage. In preprocessing, these nulls must be legitimately imputed to 0.0 (representing 0 token transactions) without adding a leakage indicator.

---
## 4. Anomaly Detection & Invalid Numerical Values

- **Infinite Values (`np.isinf`)**: 0 across all columns.
- **Negative Values**: Detected exclusively in `total ether balance`.
  - Occurrences: **978 rows** (9.94%)
  - Minimum Value: **-15,605,352.04 ETH**
  - **Root Cause Analysis**: An Ethereum address can never have a negative balance in EVM world state. In this dataset, `total ether balance` was computed as `total ether received - total Ether sent`. Because the data collector only queried normal transactions (`txlist`) and omitted internal transactions (`txlistinternal`), mining rewards, and contract withdrawal events, funds received via DEX swaps or smart contracts were never counted, yet subsequent outgoing transfers were counted. This created phantom negative balances of up to -15.6M ETH.

---
## 5. Constant & Redundant Feature Audit

### Zero-Variance (Constant) Columns
The following columns contain exactly one unique value across all valid entries and contribute zero predictive signal:
- ` ERC20 avg time between sent tnx`: constant value = `0.0`
- ` ERC20 avg time between rec tnx`: constant value = `0.0`
- ` ERC20 avg time between rec 2 tnx`: constant value = `0.0`
- ` ERC20 avg time between contract tnx`: constant value = `0.0`
- ` ERC20 min val sent contract`: constant value = `0.0`
- ` ERC20 max val sent contract`: constant value = `0.0`
- ` ERC20 avg val sent contract`: constant value = `0.0`

### Near-Zero-Variance Columns (>99.9% Zeros)

### Redundant Sums
- `total_transactions_sum`: `Sent tnx + Received Tnx + Number of Created Contracts` exactly matches reported total (Max difference: 0.0).

---
## 6. Complete Column Schema & Missingness Summary

| Column Name (Raw) | Standardized Name | Data Type | Null Count | Null % | Unique Values |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Unnamed: 0` | `unnamed:_0` | `int64` | 0 | 0.0% | 9,841 |
| `Index` | `index` | `int64` | 0 | 0.0% | 4,729 |
| `Address` | `address` | `str` | 0 | 0.0% | 9,816 |
| `FLAG` | `flag` | `int64` | 0 | 0.0% | 2 |
| `Avg min between sent tnx` | `avg_min_between_sent_tnx` | `float64` | 0 | 0.0% | 5,013 |
| `Avg min between received tnx` | `avg_min_between_received_tnx` | `float64` | 0 | 0.0% | 6,223 |
| `Time Diff between first and last (Mins)` | `time_diff_between_first_and_last_mins` | `float64` | 0 | 0.0% | 7,810 |
| `Sent tnx` | `sent_tnx` | `int64` | 0 | 0.0% | 641 |
| `Received Tnx` | `received_tnx` | `int64` | 0 | 0.0% | 727 |
| `Number of Created Contracts` | `number_of_created_contracts` | `int64` | 0 | 0.0% | 20 |
| `Unique Received From Addresses` | `unique_received_from_addresses` | `int64` | 0 | 0.0% | 256 |
| `Unique Sent To Addresses` | `unique_sent_to_addresses` | `int64` | 0 | 0.0% | 258 |
| `min value received` | `min_value_received` | `float64` | 0 | 0.0% | 4,589 |
| `max value received ` | `max_value_received` | `float64` | 0 | 0.0% | 6,302 |
| `avg val received` | `avg_val_received` | `float64` | 0 | 0.0% | 6,767 |
| `min val sent` | `min_val_sent` | `float64` | 0 | 0.0% | 4,719 |
| `max val sent` | `max_val_sent` | `float64` | 0 | 0.0% | 6,647 |
| `avg val sent` | `avg_val_sent` | `float64` | 0 | 0.0% | 5,854 |
| `min value sent to contract` | `min_value_sent_to_contract` | `float64` | 0 | 0.0% | 3 |
| `max val sent to contract` | `max_val_sent_to_contract` | `float64` | 0 | 0.0% | 4 |
| `avg value sent to contract` | `avg_value_sent_to_contract` | `float64` | 0 | 0.0% | 4 |
| `total transactions (including tnx to create contract` | `total_transactions_including_tnx_to_create_contract` | `int64` | 0 | 0.0% | 897 |
| `total Ether sent` | `total_ether_sent` | `float64` | 0 | 0.0% | 5,868 |
| `total ether received` | `total_ether_received` | `float64` | 0 | 0.0% | 6,728 |
| `total ether sent contracts` | `total_ether_sent_contracts` | `float64` | 0 | 0.0% | 4 |
| `total ether balance` | `total_ether_balance` | `float64` | 0 | 0.0% | 5,717 |
| ` Total ERC20 tnxs` | `total_erc20_tnxs` | `float64` | 829 | 8.42% | 301 |
| ` ERC20 total Ether received` | `erc20_total_ether_received` | `float64` | 829 | 8.42% | 3,461 |
| ` ERC20 total ether sent` | `erc20_total_ether_sent` | `float64` | 829 | 8.42% | 1,416 |
| ` ERC20 total Ether sent contract` | `erc20_total_ether_sent_contract` | `float64` | 829 | 8.42% | 30 |
| ` ERC20 uniq sent addr` | `erc20_uniq_sent_addr` | `float64` | 829 | 8.42% | 108 |
| ` ERC20 uniq rec addr` | `erc20_uniq_rec_addr` | `float64` | 829 | 8.42% | 148 |
| ` ERC20 uniq sent addr.1` | `erc20_uniq_sent_addr_1` | `float64` | 829 | 8.42% | 5 |
| ` ERC20 uniq rec contract addr` | `erc20_uniq_rec_contract_addr` | `float64` | 829 | 8.42% | 124 |
| ` ERC20 avg time between sent tnx` | `erc20_avg_time_between_sent_tnx` | `float64` | 829 | 8.42% | 2 |
| ` ERC20 avg time between rec tnx` | `erc20_avg_time_between_rec_tnx` | `float64` | 829 | 8.42% | 2 |
| ` ERC20 avg time between rec 2 tnx` | `erc20_avg_time_between_rec_2_tnx` | `float64` | 829 | 8.42% | 2 |
| ` ERC20 avg time between contract tnx` | `erc20_avg_time_between_contract_tnx` | `float64` | 829 | 8.42% | 2 |
| ` ERC20 min val rec` | `erc20_min_val_rec` | `float64` | 829 | 8.42% | 1,277 |
| ` ERC20 max val rec` | `erc20_max_val_rec` | `float64` | 829 | 8.42% | 2,648 |
| ` ERC20 avg val rec` | `erc20_avg_val_rec` | `float64` | 829 | 8.42% | 3,381 |
| ` ERC20 min val sent` | `erc20_min_val_sent` | `float64` | 829 | 8.42% | 477 |
| ` ERC20 max val sent` | `erc20_max_val_sent` | `float64` | 829 | 8.42% | 1,131 |
| ` ERC20 avg val sent` | `erc20_avg_val_sent` | `float64` | 829 | 8.42% | 1,310 |
| ` ERC20 min val sent contract` | `erc20_min_val_sent_contract` | `float64` | 829 | 8.42% | 2 |
| ` ERC20 max val sent contract` | `erc20_max_val_sent_contract` | `float64` | 829 | 8.42% | 2 |
| ` ERC20 avg val sent contract` | `erc20_avg_val_sent_contract` | `float64` | 829 | 8.42% | 2 |
| ` ERC20 uniq sent token name` | `erc20_uniq_sent_token_name` | `float64` | 829 | 8.42% | 71 |
| ` ERC20 uniq rec token name` | `erc20_uniq_rec_token_name` | `float64` | 829 | 8.42% | 122 |
| ` ERC20 most sent token type` | `erc20_most_sent_token_type` | `str` | 2,697 | 27.41% | 305 |
| ` ERC20_most_rec_token_type` | `erc20_most_rec_token_type` | `str` | 871 | 8.85% | 467 |

---

## 7. Sample Records (Masked Identifiers)

Below are sample records demonstrating feature representations while protecting address privacy:

| Index | Address       | FLAG | Sent tnx | Received Tnx | total Ether sent | total ether received | total ether balance |
| ----- | ------------- | ---- | -------- | ------------ | ---------------- | -------------------- | ------------------- |
| 1     | 0x0000...05e8 | 0    | 721      | 89           | 865.6910932      | 586.4666748          | -279.2244185        |
| 2     | 0x0002...6fed | 0    | 94       | 8            | 3.08729702       | 3.085478209          | -0.001818811        |
| 3     | 0x0002...a244 | 0    | 2        | 10           | 3.58861565       | 3.58905665           | 0.000441            |
| 2177  | 0xff8e...bccc | 1    | 67       | 43           | 61.78599493      | 53.07025157          | -8.715743362000001  |
| 2178  | 0xffde...8218 | 1    | 0        | 1            | 0.0              | 0.5                  | 0.5                 |
| 2179  | 0xd624...9a3c | 1    | 18       | 3            | 11599.7          | 18999.79523          | 7400.095233         |