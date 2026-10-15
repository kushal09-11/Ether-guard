# Etherscan API V2 Feature Compatibility Audit (Pre-Training Verification)
## Ethereum Wallet Fraud Detection Machine Learning Study (ChainSentinel) - Phase 1

This audit systematically examines every single one of the 51 columns in the Kaggle Ethereum Fraud Detection dataset.
It determines whether and how each feature can be reconstructed for a live Ethereum wallet using Etherscan API V2 endpoints (`txlist`, `tokentx`, `txlistinternal`, `balance`).

### Rigorous Parity Classification Framework
In strict compliance with academic auditing requirements, features are classified into three parity tiers:
- **Verified equivalent**: Mathematical definition, unit, directional logic, and edge-case behavior are proven identical between training and live calculation. *(Result: 0 features qualify, because raw transaction logs were discarded by the dataset compiler)*.
- **Approximate equivalent**: Feature is close in concept and units, but differences in historical vs live calculation exist (algebraic redundancy, missing internal transactions, or token zero-imputation assumptions). *(Result: 9 features)*.
- **Not verifiable from the available dataset**: Parity cannot be proven because the Kaggle tabular summary discarded transaction-level error status (`isError`). *(Result: 15 features)*.

---
## Full 51-Feature Compatibility Matrix

| Column in CSV | Role | Unit | Parity Classification | Failed Tx Inclusion | Additional Endpoint | In Schema? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Unnamed: 0` | Metadata / Identifier | Integer index | **Excluded from Schema** | N/A (Excluded) | `None` | ❌ Excluded |
| `Index` | Metadata / Identifier | Integer index | **Excluded from Schema** | N/A (Excluded) | `None` | ❌ Excluded |
| `Address` | Identifier | Hexadecimal string | **Excluded from Schema** | N/A (Excluded) | `None (input query parameter).` | ❌ Excluded |
| `FLAG` | Target Label | Binary flag (0 or 1) | **Excluded from Schema** | N/A (Excluded) | `Etherscan Label Cloud / CryptScamDB / Chainalysis for benchmark verification.` | ❌ Excluded |
| `Avg min between sent tnx` | Input Feature | Minutes | **Not verifiable from the available dataset** | Unresolved (raw transaction logs and isError flags discarded in CSV) | `None (normal `txlist` endpoint).` | ✅ Yes |
| `Avg min between received tnx` | Input Feature | Minutes | **Not verifiable from the available dataset** | Unresolved (raw transaction logs and isError flags discarded in CSV) | `None for normal txs. Internal contract incoming transfers are omitted to match dataset.` | ✅ Yes |
| `Time Diff between first and last (Mins)` | Input Feature | Minutes | **Not verifiable from the available dataset** | Unresolved (raw transaction logs and isError flags discarded in CSV) | `None (`txlist`).` | ✅ Yes |
| `Sent tnx` | Input Feature | Count | **Not verifiable from the available dataset** | Unresolved (raw transaction logs and isError flags discarded in CSV) | `None (`txlist`).` | ✅ Yes |
| `Received Tnx` | Input Feature | Count | **Not verifiable from the available dataset** | Unresolved (raw transaction logs and isError flags discarded in CSV) | `None (`txlist`).` | ✅ Yes |
| `Number of Created Contracts` | Input Feature | Count | **Not verifiable from the available dataset** | Unresolved (raw transaction logs and isError flags discarded in CSV) | `None (`txlist`).` | ✅ Yes |
| `Unique Received From Addresses` | Input Feature | Count | **Not verifiable from the available dataset** | Unresolved (raw transaction logs and isError flags discarded in CSV) | `None (`txlist`).` | ✅ Yes |
| `Unique Sent To Addresses` | Input Feature | Count | **Not verifiable from the available dataset** | Unresolved (raw transaction logs and isError flags discarded in CSV) | `None (`txlist`).` | ✅ Yes |
| `min value received` | Input Feature | Ether (ETH) | **Not verifiable from the available dataset** | Unresolved (raw transaction logs and isError flags discarded in CSV) | `None (`txlist`).` | ✅ Yes |
| `max value received ` | Input Feature | Ether (ETH) | **Not verifiable from the available dataset** | Unresolved (raw transaction logs and isError flags discarded in CSV) | `None (`txlist`).` | ✅ Yes |
| `avg val received` | Input Feature | Ether (ETH) | **Not verifiable from the available dataset** | Unresolved (raw transaction logs and isError flags discarded in CSV) | `None (`txlist`).` | ✅ Yes |
| `min val sent` | Input Feature | Ether (ETH) | **Not verifiable from the available dataset** | Unresolved (raw transaction logs and isError flags discarded in CSV) | `None (`txlist`).` | ✅ Yes |
| `max val sent` | Input Feature | Ether (ETH) | **Not verifiable from the available dataset** | Unresolved (raw transaction logs and isError flags discarded in CSV) | `None (`txlist`).` | ✅ Yes |
| `avg val sent` | Input Feature | Ether (ETH) | **Not verifiable from the available dataset** | Unresolved (raw transaction logs and isError flags discarded in CSV) | `None (`txlist`).` | ✅ Yes |
| `min value sent to contract` | Input Feature | Ether (ETH) | **Excluded from Schema** | N/A (Excluded) | `None (`txlist`).` | ❌ Excluded |
| `max val sent to contract` | Input Feature | Ether (ETH) | **Excluded from Schema** | N/A (Excluded) | `None (`txlist`).` | ❌ Excluded |
| `avg value sent to contract` | Input Feature | Ether (ETH) | **Excluded from Schema** | N/A (Excluded) | `None (`txlist`).` | ❌ Excluded |
| `total transactions (including tnx to create contract` | Input Feature (Redundant) | Count | **Approximate equivalent** | Unresolved (inherits underlying sent/received failed tx uncertainty) | `None (`txlist`).` | ✅ Yes |
| `total Ether sent` | Input Feature | Ether (ETH) | **Not verifiable from the available dataset** | Unresolved (raw transaction logs and isError flags discarded in CSV) | `None (`txlist`).` | ✅ Yes |
| `total ether received` | Input Feature | Ether (ETH) | **Approximate equivalent** | Unresolved (cannot verify if failed incoming txs were counted) | `None (`txlist`).` | ✅ Yes |
| `total ether sent contracts` | Input Feature | Ether (ETH) | **Excluded from Schema** | N/A (Excluded) | `None (`txlist`).` | ❌ Excluded |
| `total ether balance` | Input Feature (Defective/Synthetic) | Ether (ETH) | **Approximate equivalent** | Unresolved (inherits underlying sent/received failed tx uncertainty) | `Etherscan `module=account&action=balance` returns true EVM balance.` | ✅ Yes |
| ` Total ERC20 tnxs` | Input Feature | Count | **Approximate equivalent** | Excluded by EVM semantics (tokentx event logs emit only on successful execution) | `Etherscan `module=account&action=tokentx&address={address}`.` | ✅ Yes |
| ` ERC20 total Ether received` | Input Feature (Mislabeled Unit) | Crude token unit (scaled by 1e18) | **Excluded from Schema** | N/A (Excluded) | `Etherscan `module=account&action=tokentx`.` | ❌ Excluded |
| ` ERC20 total ether sent` | Input Feature (Mislabeled Unit) | Crude token unit (scaled by 1e18) | **Excluded from Schema** | N/A (Excluded) | `Etherscan `module=account&action=tokentx`.` | ❌ Excluded |
| ` ERC20 total Ether sent contract` | Input Feature | Crude token unit (scaled by 1e18) | **Excluded from Schema** | N/A (Excluded) | `Etherscan `tokentx` + smart contract detection (`eth_getCode`).` | ❌ Excluded |
| ` ERC20 uniq sent addr` | Input Feature | Count | **Approximate equivalent** | Excluded by EVM semantics (tokentx event logs emit only on successful execution) | `Etherscan `module=account&action=tokentx`.` | ✅ Yes |
| ` ERC20 uniq rec addr` | Input Feature | Count | **Approximate equivalent** | Excluded by EVM semantics (tokentx event logs emit only on successful execution) | `Etherscan `module=account&action=tokentx`.` | ✅ Yes |
| ` ERC20 uniq sent addr.1` | Input Feature | Count | **Excluded from Schema** | N/A (Excluded) | `Etherscan `tokentx` + contract verification.` | ❌ Excluded |
| ` ERC20 uniq rec contract addr` | Input Feature | Count | **Approximate equivalent** | Excluded by EVM semantics (tokentx event logs emit only on successful execution) | `Etherscan `module=account&action=tokentx`.` | ✅ Yes |
| ` ERC20 avg time between sent tnx` | Constant Feature | Minutes | **Excluded from Schema** | N/A (Excluded) | `None` | ❌ Excluded |
| ` ERC20 avg time between rec tnx` | Constant Feature | Minutes | **Excluded from Schema** | N/A (Excluded) | `None` | ❌ Excluded |
| ` ERC20 avg time between rec 2 tnx` | Constant Feature | Minutes | **Excluded from Schema** | N/A (Excluded) | `None` | ❌ Excluded |
| ` ERC20 avg time between contract tnx` | Constant Feature | Minutes | **Excluded from Schema** | N/A (Excluded) | `None` | ❌ Excluded |
| ` ERC20 min val rec` | Input Feature | Crude token unit (scaled by 1e18) | **Excluded from Schema** | N/A (Excluded) | `Etherscan `module=account&action=tokentx`.` | ❌ Excluded |
| ` ERC20 max val rec` | Input Feature | Crude token unit (scaled by 1e18) | **Excluded from Schema** | N/A (Excluded) | `Etherscan `module=account&action=tokentx`.` | ❌ Excluded |
| ` ERC20 avg val rec` | Input Feature | Crude token unit (scaled by 1e18) | **Excluded from Schema** | N/A (Excluded) | `Etherscan `module=account&action=tokentx`.` | ❌ Excluded |
| ` ERC20 min val sent` | Input Feature | Crude token unit (scaled by 1e18) | **Excluded from Schema** | N/A (Excluded) | `Etherscan `module=account&action=tokentx`.` | ❌ Excluded |
| ` ERC20 max val sent` | Input Feature | Crude token unit (scaled by 1e18) | **Excluded from Schema** | N/A (Excluded) | `Etherscan `module=account&action=tokentx`.` | ❌ Excluded |
| ` ERC20 avg val sent` | Input Feature | Crude token unit (scaled by 1e18) | **Excluded from Schema** | N/A (Excluded) | `Etherscan `module=account&action=tokentx`.` | ❌ Excluded |
| ` ERC20 min val sent contract` | Constant Feature | Crude token unit | **Excluded from Schema** | N/A (Excluded) | `None` | ❌ Excluded |
| ` ERC20 max val sent contract` | Constant Feature | Crude token unit | **Excluded from Schema** | N/A (Excluded) | `None` | ❌ Excluded |
| ` ERC20 avg val sent contract` | Constant Feature | Crude token unit | **Excluded from Schema** | N/A (Excluded) | `None` | ❌ Excluded |
| ` ERC20 uniq sent token name` | Input Feature | Count | **Approximate equivalent** | Excluded by EVM semantics (tokentx event logs emit only on successful execution) | `Etherscan `module=account&action=tokentx`.` | ✅ Yes |
| ` ERC20 uniq rec token name` | Input Feature | Count | **Approximate equivalent** | Excluded by EVM semantics (tokentx event logs emit only on successful execution) | `Etherscan `module=account&action=tokentx`.` | ✅ Yes |
| ` ERC20 most sent token type` | Categorical Feature | Token symbol/name string | **Excluded from Schema** | N/A (Excluded) | `Etherscan `module=account&action=tokentx`.` | ❌ Excluded |
| ` ERC20_most_rec_token_type` | Categorical Feature | Token symbol/name string | **Excluded from Schema** | N/A (Excluded) | `Etherscan `module=account&action=tokentx`.` | ❌ Excluded |

---

## Detailed Technical Audit by Feature Group

### Base ETH Transaction Features (Exact Reproducibility via `txlist`)

#### `Avg min between sent tnx` (Standardized: `avg_min_between_sent_tnx`)
- **Role**: Input Feature
- **Unit**: Minutes
- **Calculation Required**: Filter normal transactions where `from == wallet_address` and `isError == '0'`. Sort ascending by `timeStamp`. If count <= 1, return 0.0. Else: (max(timeStamp) - min(timeStamp)) / (60.0 * (count - 1)).
- **Endpoint Dependency**: `None (normal `txlist` endpoint).`
- **Reproducibility Rating**: **Exact**
- **Assumptions and Risks**: Etherscan returns max 10,000 transactions per query without pagination. Truncation on high-volume wallets would underestimate or overestimate interval.

#### `Avg min between received tnx` (Standardized: `avg_min_between_received_tnx`)
- **Role**: Input Feature
- **Unit**: Minutes
- **Calculation Required**: Filter normal transactions where `to == wallet_address` and `isError == '0'`. Sort ascending by `timeStamp`. If count <= 1, return 0.0. Else: (max(timeStamp) - min(timeStamp)) / (60.0 * (count - 1)).
- **Endpoint Dependency**: `None for normal txs. Internal contract incoming transfers are omitted to match dataset.`
- **Reproducibility Rating**: **Exact**
- **Assumptions and Risks**: Captures only normal incoming transfers. Inflows from smart contracts (DEX, mining) are omitted unless `txlistinternal` is used, matching training dataset generation.

#### `Time Diff between first and last (Mins)` (Standardized: `time_diff_first_last_mins`)
- **Role**: Input Feature
- **Unit**: Minutes
- **Calculation Required**: Across all normal transactions in `txlist`: if len(txlist) <= 1, return 0.0. Else: (max(timeStamp) - min(timeStamp)) / 60.0.
- **Endpoint Dependency**: `None (`txlist`).`
- **Reproducibility Rating**: **Exact**
- **Assumptions and Risks**: Reflects wallet longevity. If a wallet has only 1 transaction, active window is 0. Pagination required for wallets with >10,000 transactions to capture true first transaction.

#### `Sent tnx` (Standardized: `sent_tnx`)
- **Role**: Input Feature
- **Unit**: Count
- **Calculation Required**: Count of records in `txlist` where `from == wallet_address` and `isError == '0'`.
- **Endpoint Dependency**: `None (`txlist`).`
- **Reproducibility Rating**: **Exact**
- **Assumptions and Risks**: Exact match with normal external transaction history.

#### `Received Tnx` (Standardized: `received_tnx`)
- **Role**: Input Feature
- **Unit**: Count
- **Calculation Required**: Count of records in `txlist` where `to == wallet_address` and `isError == '0'`.
- **Endpoint Dependency**: `None (`txlist`).`
- **Reproducibility Rating**: **Exact**
- **Assumptions and Risks**: Matches training dataset's normal transaction incoming count.

#### `Number of Created Contracts` (Standardized: `created_contracts_count`)
- **Role**: Input Feature
- **Unit**: Count
- **Calculation Required**: Count records in `txlist` where `from == wallet_address` and (`to == ''` or `to is None` or `to == '0x'` or `contractAddress != ''`).
- **Endpoint Dependency**: `None (`txlist`).`
- **Reproducibility Rating**: **Exact**
- **Assumptions and Risks**: Standard contracts deployed by wallet. Essential indicator for scam factory detection.

#### `Unique Received From Addresses` (Standardized: `unique_received_from_addresses`)
- **Role**: Input Feature
- **Unit**: Count
- **Calculation Required**: len(set(tx['from'].lower() for tx in txlist if tx['to'].lower() == wallet_address.lower()))
- **Endpoint Dependency**: `None (`txlist`).`
- **Reproducibility Rating**: **Exact**
- **Assumptions and Risks**: Measures sender diversity and fan-in structure.

#### `Unique Sent To Addresses` (Standardized: `unique_sent_to_addresses`)
- **Role**: Input Feature
- **Unit**: Count
- **Calculation Required**: len(set(tx['to'].lower() for tx in txlist if tx['from'].lower() == wallet_address.lower() and tx.get('to')))
- **Endpoint Dependency**: `None (`txlist`).`
- **Reproducibility Rating**: **Exact**
- **Assumptions and Risks**: Measures recipient diversity and fan-out structure.

#### `min value received` (Standardized: `min_value_received_eth`)
- **Role**: Input Feature
- **Unit**: Ether (ETH)
- **Calculation Required**: min([float(tx['value'])/1e18 for tx in incoming_txlist]) if incoming_txlist else 0.0
- **Endpoint Dependency**: `None (`txlist`).`
- **Reproducibility Rating**: **Exact**
- **Assumptions and Risks**: Requires converting Wei to Ether by dividing by 10^18.

#### `max value received ` (Standardized: `max_value_received_eth`)
- **Role**: Input Feature
- **Unit**: Ether (ETH)
- **Calculation Required**: max([float(tx['value'])/1e18 for tx in incoming_txlist]) if incoming_txlist else 0.0
- **Endpoint Dependency**: `None (`txlist`).`
- **Reproducibility Rating**: **Exact**
- **Assumptions and Risks**: Requires dividing Wei by 10^18. Raw column name contains trailing whitespace in CSV.

#### `avg val received` (Standardized: `avg_value_received_eth`)
- **Role**: Input Feature
- **Unit**: Ether (ETH)
- **Calculation Required**: sum([float(tx['value'])/1e18 for tx in incoming_txlist]) / len(incoming_txlist) if incoming_txlist else 0.0
- **Endpoint Dependency**: `None (`txlist`).`
- **Reproducibility Rating**: **Exact**
- **Assumptions and Risks**: Requires dividing Wei by 10^18.

#### `min val sent` (Standardized: `min_value_sent_eth`)
- **Role**: Input Feature
- **Unit**: Ether (ETH)
- **Calculation Required**: min([float(tx['value'])/1e18 for tx in outgoing_txlist]) if outgoing_txlist else 0.0
- **Endpoint Dependency**: `None (`txlist`).`
- **Reproducibility Rating**: **Exact**
- **Assumptions and Risks**: Zero-value contract interaction transactions return 0.0.

#### `max val sent` (Standardized: `max_value_sent_eth`)
- **Role**: Input Feature
- **Unit**: Ether (ETH)
- **Calculation Required**: max([float(tx['value'])/1e18 for tx in outgoing_txlist]) if outgoing_txlist else 0.0
- **Endpoint Dependency**: `None (`txlist`).`
- **Reproducibility Rating**: **Exact**
- **Assumptions and Risks**: Requires dividing Wei by 10^18.

#### `avg val sent` (Standardized: `avg_value_sent_eth`)
- **Role**: Input Feature
- **Unit**: Ether (ETH)
- **Calculation Required**: sum([float(tx['value'])/1e18 for tx in outgoing_txlist]) / len(outgoing_txlist) if outgoing_txlist else 0.0
- **Endpoint Dependency**: `None (`txlist`).`
- **Reproducibility Rating**: **Exact**
- **Assumptions and Risks**: Requires dividing Wei by 10^18.

#### `total transactions (including tnx to create contract` (Standardized: `total_transactions_count`)
- **Role**: Input Feature (Redundant)
- **Unit**: Count
- **Calculation Required**: len(txlist) or (sent_tnx + received_tnx + created_contracts_count)
- **Endpoint Dependency**: `None (`txlist`).`
- **Reproducibility Rating**: **Exact**
- **Assumptions and Risks**: Exact linear combination of sent, received, and contract creation counts.

#### `total Ether sent` (Standardized: `total_ether_sent`)
- **Role**: Input Feature
- **Unit**: Ether (ETH)
- **Calculation Required**: sum([float(tx['value'])/1e18 for tx in outgoing_txlist])
- **Endpoint Dependency**: `None (`txlist`).`
- **Reproducibility Rating**: **Exact**
- **Assumptions and Risks**: Does not include gas fees paid.

#### `total ether received` (Standardized: `total_ether_received`)
- **Role**: Input Feature
- **Unit**: Ether (ETH)
- **Calculation Required**: sum([float(tx['value'])/1e18 for tx in incoming_txlist])
- **Endpoint Dependency**: `None (`txlist`).`
- **Reproducibility Rating**: **Exact**
- **Assumptions and Risks**: Captures only normal transactions. Inflows from smart contracts (DEX, mining) are omitted to strictly reproduce training dataset distribution.

### Anomalous / Defective Balance Feature

#### `total ether balance` (Standardized: `total_ether_balance`)
- **Role**: Input Feature (Defective/Synthetic)
- **Unit**: Ether (ETH)
- **Calculation Required**: Formula: `total_ether_received - total_ether_sent`. DO NOT use Etherscan `balance` endpoint directly, as live balance is >= 0 while dataset contains negative balances down to -15.6M ETH due to missing internal txs.
- **Endpoint Dependency**: `Etherscan `module=account&action=balance` returns true EVM balance.`
- **Reproducibility Rating**: **Approximate / Semantic Conflict**
- **Assumptions and Risks**: CRITICAL TRAIN-INFERENCE RISK: The true EVM balance is never negative. Using live balance creates severe covariate shift. Live pipeline MUST compute the formula `total_ether_received - total_ether_sent` to match training semantics.

### Contract Creation Value Features (Near-Zero Variance)

#### `min value sent to contract` (Standardized: `min_value_sent_to_contract_eth`)
- **Role**: Input Feature
- **Unit**: Ether (ETH)
- **Calculation Required**: min([float(tx['value'])/1e18 for tx in contract_deploy_txs]) if contract_deploy_txs else 0.0
- **Endpoint Dependency**: `None (`txlist`).`
- **Reproducibility Rating**: **Exact**
- **Assumptions and Risks**: Near-zero variance (>99.97% zero in dataset). High sparsity.

#### `max val sent to contract` (Standardized: `max_value_sent_to_contract_eth`)
- **Role**: Input Feature
- **Unit**: Ether (ETH)
- **Calculation Required**: max([float(tx['value'])/1e18 for tx in contract_deploy_txs]) if contract_deploy_txs else 0.0
- **Endpoint Dependency**: `None (`txlist`).`
- **Reproducibility Rating**: **Exact**
- **Assumptions and Risks**: Near-zero variance (>99.97% zero in dataset). High sparsity.

#### `avg value sent to contract` (Standardized: `avg_value_sent_to_contract_eth`)
- **Role**: Input Feature
- **Unit**: Ether (ETH)
- **Calculation Required**: sum([float(tx['value'])/1e18 for tx in contract_deploy_txs]) / len(contract_deploy_txs) if contract_deploy_txs else 0.0
- **Endpoint Dependency**: `None (`txlist`).`
- **Reproducibility Rating**: **Exact**
- **Assumptions and Risks**: Near-zero variance (>99.97% zero in dataset). High sparsity.

#### `total ether sent contracts` (Standardized: `total_ether_sent_contracts`)
- **Role**: Input Feature
- **Unit**: Ether (ETH)
- **Calculation Required**: sum([float(tx['value'])/1e18 for tx in contract_deploy_txs])
- **Endpoint Dependency**: `None (`txlist`).`
- **Reproducibility Rating**: **Exact**
- **Assumptions and Risks**: Near-zero variance (>99.97% zero in dataset).

### ERC-20 Asset & Frequency Features (Requires `tokentx`)

#### ` Total ERC20 tnxs` (Standardized: `erc20_total_tnx`)
- **Role**: Input Feature
- **Unit**: Count
- **Calculation Required**: len(tokentx_list)
- **Endpoint Dependency**: `Etherscan `module=account&action=tokentx&address={address}`.`
- **Reproducibility Rating**: **Exact (via ERC20 endpoint)**
- **Assumptions and Risks**: Requires calling a dedicated secondary endpoint. For wallets with 0 token transfers, value must be 0.0 (imputed), avoiding target leakage.

#### ` ERC20 uniq sent addr` (Standardized: `erc20_unique_sent_addr`)
- **Role**: Input Feature
- **Unit**: Count
- **Calculation Required**: len(set(tx['to'].lower() for tx in tokentx if tx['from'].lower() == wallet.lower() and tx.get('to')))
- **Endpoint Dependency**: `Etherscan `module=account&action=tokentx`.`
- **Reproducibility Rating**: **Exact (via ERC20 endpoint)**
- **Assumptions and Risks**: Measures token transfer recipient diversity.

#### ` ERC20 uniq rec addr` (Standardized: `erc20_unique_received_addr`)
- **Role**: Input Feature
- **Unit**: Count
- **Calculation Required**: len(set(tx['from'].lower() for tx in tokentx if tx['to'].lower() == wallet.lower()))
- **Endpoint Dependency**: `Etherscan `module=account&action=tokentx`.`
- **Reproducibility Rating**: **Exact (via ERC20 endpoint)**
- **Assumptions and Risks**: Measures token transfer sender diversity.

#### ` ERC20 uniq rec contract addr` (Standardized: `erc20_unique_received_contract_addr`)
- **Role**: Input Feature
- **Unit**: Count
- **Calculation Required**: len(set(tx['contractAddress'].lower() for tx in tokentx if tx['to'].lower() == wallet.lower()))
- **Endpoint Dependency**: `Etherscan `module=account&action=tokentx`.`
- **Reproducibility Rating**: **Exact (via ERC20 endpoint)**
- **Assumptions and Risks**: Measures token portfolio breadth and airdrop exposure.

#### ` ERC20 uniq sent token name` (Standardized: `erc20_unique_sent_token_name`)
- **Role**: Input Feature
- **Unit**: Count
- **Calculation Required**: len(set(tx['tokenSymbol'] for tx in tokentx if tx['from'].lower() == wallet.lower()))
- **Endpoint Dependency**: `Etherscan `module=account&action=tokentx`.`
- **Reproducibility Rating**: **Exact (via ERC20 endpoint)**
- **Assumptions and Risks**: Count of distinct outbound token symbols.

#### ` ERC20 uniq rec token name` (Standardized: `erc20_unique_received_token_name`)
- **Role**: Input Feature
- **Unit**: Count
- **Calculation Required**: len(set(tx['tokenSymbol'] for tx in tokentx if tx['to'].lower() == wallet.lower()))
- **Endpoint Dependency**: `Etherscan `module=account&action=tokentx`.`
- **Reproducibility Rating**: **Exact (via ERC20 endpoint)**
- **Assumptions and Risks**: Count of distinct inbound token symbols (airdrop spam, tokens).

### Flawed / Heterogeneous Token Value Sums (Unit Distortion Risk)

#### ` ERC20 total Ether received` (Standardized: `erc20_total_received_amount`)
- **Role**: Input Feature (Mislabeled Unit)
- **Unit**: Crude token unit (scaled by 1e18)
- **Calculation Required**: sum([float(tx['value'])/1e18 for tx in tokentx if tx['to'].lower() == wallet.lower()])
- **Endpoint Dependency**: `Etherscan `module=account&action=tokentx`.`
- **Reproducibility Rating**: **Approximate (Methodological Flaw)**
- **Assumptions and Risks**: Methodological flaw in Kaggle dataset: sums heterogeneous tokens without normalising for decimals (USDT=6, WBTC=8, DAI=18). Live inference must emulate this exact crude calculation.

#### ` ERC20 total ether sent` (Standardized: `erc20_total_sent_amount`)
- **Role**: Input Feature (Mislabeled Unit)
- **Unit**: Crude token unit (scaled by 1e18)
- **Calculation Required**: sum([float(tx['value'])/1e18 for tx in tokentx if tx['from'].lower() == wallet.lower()])
- **Endpoint Dependency**: `Etherscan `module=account&action=tokentx`.`
- **Reproducibility Rating**: **Approximate (Methodological Flaw)**
- **Assumptions and Risks**: Methodological flaw in dataset: heterogeneous token values summed without decimal normalization.

#### ` ERC20 total Ether sent contract` (Standardized: `erc20_total_sent_contract_amount`)
- **Role**: Input Feature
- **Unit**: Crude token unit (scaled by 1e18)
- **Calculation Required**: Sum of token values sent to contract addresses divided by 1e18.
- **Endpoint Dependency**: `Etherscan `tokentx` + smart contract detection (`eth_getCode`).`
- **Reproducibility Rating**: **Approximate**
- **Assumptions and Risks**: Requires checking contract bytecode for recipient addresses.

#### ` ERC20 min val rec` (Standardized: `erc20_min_val_received`)
- **Role**: Input Feature
- **Unit**: Crude token unit (scaled by 1e18)
- **Calculation Required**: min([float(tx['value'])/1e18 for tx in tokentx_in]) if tokentx_in else 0.0
- **Endpoint Dependency**: `Etherscan `module=account&action=tokentx`.`
- **Reproducibility Rating**: **Approximate**
- **Assumptions and Risks**: Token decimal heterogeneity distorts values across different assets.

#### ` ERC20 max val rec` (Standardized: `erc20_max_val_received`)
- **Role**: Input Feature
- **Unit**: Crude token unit (scaled by 1e18)
- **Calculation Required**: max([float(tx['value'])/1e18 for tx in tokentx_in]) if tokentx_in else 0.0
- **Endpoint Dependency**: `Etherscan `module=account&action=tokentx`.`
- **Reproducibility Rating**: **Approximate**
- **Assumptions and Risks**: Distorted by token decimal differences.

#### ` ERC20 avg val rec` (Standardized: `erc20_avg_val_received`)
- **Role**: Input Feature
- **Unit**: Crude token unit (scaled by 1e18)
- **Calculation Required**: Average of token values divided by 1e18.
- **Endpoint Dependency**: `Etherscan `module=account&action=tokentx`.`
- **Reproducibility Rating**: **Approximate**
- **Assumptions and Risks**: Distorted by token decimal differences.

#### ` ERC20 min val sent` (Standardized: `erc20_min_val_sent`)
- **Role**: Input Feature
- **Unit**: Crude token unit (scaled by 1e18)
- **Calculation Required**: min([float(tx['value'])/1e18 for tx in tokentx_out]) if tokentx_out else 0.0
- **Endpoint Dependency**: `Etherscan `module=account&action=tokentx`.`
- **Reproducibility Rating**: **Approximate**
- **Assumptions and Risks**: Distorted by token decimal differences.

#### ` ERC20 max val sent` (Standardized: `erc20_max_val_sent`)
- **Role**: Input Feature
- **Unit**: Crude token unit (scaled by 1e18)
- **Calculation Required**: max([float(tx['value'])/1e18 for tx in tokentx_out]) if tokentx_out else 0.0
- **Endpoint Dependency**: `Etherscan `module=account&action=tokentx`.`
- **Reproducibility Rating**: **Approximate**
- **Assumptions and Risks**: Distorted by token decimal differences.

#### ` ERC20 avg val sent` (Standardized: `erc20_avg_val_sent`)
- **Role**: Input Feature
- **Unit**: Crude token unit (scaled by 1e18)
- **Calculation Required**: Average of sent token values divided by 1e18.
- **Endpoint Dependency**: `Etherscan `module=account&action=tokentx`.`
- **Reproducibility Rating**: **Approximate**
- **Assumptions and Risks**: Distorted by token decimal differences.

#### ` ERC20 uniq sent addr.1` (Standardized: `erc20_unique_sent_contract_addr`)
- **Role**: Input Feature
- **Unit**: Count
- **Calculation Required**: Count distinct contract recipients for token transfers.
- **Endpoint Dependency**: `Etherscan `tokentx` + contract verification.`
- **Reproducibility Rating**: **Approximate**
- **Assumptions and Risks**: Header collision in raw CSV; 91.3% zeros. Near-zero variance.

### Zero-Variance Features (100% Constant in Training Dataset)

#### ` ERC20 avg time between sent tnx` (Standardized: `erc20_avg_time_between_sent_tnx`)
- **Role**: Constant Feature
- **Unit**: Minutes
- **Calculation Required**: None (100% constant zeros in dataset).
- **Endpoint Dependency**: `None`
- **Reproducibility Rating**: **Not Reproducible / Zero-Variance**
- **Assumptions and Risks**: Constant 0.0 across all records. Excluded.

#### ` ERC20 avg time between rec tnx` (Standardized: `erc20_avg_time_between_rec_tnx`)
- **Role**: Constant Feature
- **Unit**: Minutes
- **Calculation Required**: None (100% constant zeros in dataset).
- **Endpoint Dependency**: `None`
- **Reproducibility Rating**: **Not Reproducible / Zero-Variance**
- **Assumptions and Risks**: Constant 0.0 across all records. Excluded.

#### ` ERC20 avg time between rec 2 tnx` (Standardized: `erc20_avg_time_between_rec_2_tnx`)
- **Role**: Constant Feature
- **Unit**: Minutes
- **Calculation Required**: None (100% constant zeros in dataset).
- **Endpoint Dependency**: `None`
- **Reproducibility Rating**: **Not Reproducible / Zero-Variance**
- **Assumptions and Risks**: Constant 0.0 across all records. Excluded.

#### ` ERC20 avg time between contract tnx` (Standardized: `erc20_avg_time_between_contract_tnx`)
- **Role**: Constant Feature
- **Unit**: Minutes
- **Calculation Required**: None (100% constant zeros in dataset).
- **Endpoint Dependency**: `None`
- **Reproducibility Rating**: **Not Reproducible / Zero-Variance**
- **Assumptions and Risks**: Constant 0.0 across all records. Excluded.

#### ` ERC20 min val sent contract` (Standardized: `erc20_min_val_sent_contract`)
- **Role**: Constant Feature
- **Unit**: Crude token unit
- **Calculation Required**: None (100% constant zeros in dataset).
- **Endpoint Dependency**: `None`
- **Reproducibility Rating**: **Not Reproducible / Zero-Variance**
- **Assumptions and Risks**: Constant 0.0 across all records. Excluded.

#### ` ERC20 max val sent contract` (Standardized: `erc20_max_val_sent_contract`)
- **Role**: Constant Feature
- **Unit**: Crude token unit
- **Calculation Required**: None (100% constant zeros in dataset).
- **Endpoint Dependency**: `None`
- **Reproducibility Rating**: **Not Reproducible / Zero-Variance**
- **Assumptions and Risks**: Constant 0.0 across all records. Excluded.

#### ` ERC20 avg val sent contract` (Standardized: `erc20_avg_val_sent_contract`)
- **Role**: Constant Feature
- **Unit**: Crude token unit
- **Calculation Required**: None (100% constant zeros in dataset).
- **Endpoint Dependency**: `None`
- **Reproducibility Rating**: **Not Reproducible / Zero-Variance**
- **Assumptions and Risks**: Constant 0.0 across all records. Excluded.

### Categorical Token Identifiers (High Cardinality / Temporal Drift)

#### ` ERC20 most sent token type` (Standardized: `erc20_most_sent_token_type`)
- **Role**: Categorical Feature
- **Unit**: Token symbol/name string
- **Calculation Required**: Mode of `tokenSymbol` for outgoing `tokentx` events.
- **Endpoint Dependency**: `Etherscan `module=account&action=tokentx`.`
- **Reproducibility Rating**: **Approximate / Temporal Drift**
- **Assumptions and Risks**: Extremely high cardinality (304 unique names), heavy temporal drift, and spam token vulnerability. Not recommended for core feature schema.

#### ` ERC20_most_rec_token_type` (Standardized: `erc20_most_received_token_type`)
- **Role**: Categorical Feature
- **Unit**: Token symbol/name string
- **Calculation Required**: Mode of `tokenSymbol` for incoming `tokentx` events.
- **Endpoint Dependency**: `Etherscan `module=account&action=tokentx`.`
- **Reproducibility Rating**: **Approximate / Temporal Drift**
- **Assumptions and Risks**: High cardinality (466 unique names), heavily contaminated by promotional airdrop tokens (e.g., Livepeer, OmiseGO, INS Promo). Not recommended for core schema.
