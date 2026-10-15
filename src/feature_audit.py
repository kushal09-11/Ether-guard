"""
Feature Compatibility Audit Module for Ethereum Wallet Fraud Detection (Phase 1).
Audits all candidate dataset features against Etherscan API V2 endpoints,
documenting feasibility, calculations, endpoints, units, and train-inference risks.
Generates CSV and Markdown audit tables and the canonical common feature schema.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

import config
from src.utils import find_raw_dataset, save_json, clean_column_name


# Detailed feature audit dictionary for all 51 columns in transaction_dataset.csv
FEATURE_AUDIT_DATA: List[Dict[str, Any]] = [
    {
        "csv_column": "Unnamed: 0",
        "clean_name": "raw_row_id",
        "role": "Metadata / Identifier",
        "unit": "Integer index",
        "meaning": "Pandas dataframe export row index.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Not Reproducible",
        "calculation_required": "None (internal pandas artifact).",
        "additional_endpoint": "None",
        "assumptions_and_risks": "Arbitrary sequential row counter. Must be strictly excluded from model features.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": "Index",
        "clean_name": "dataset_index",
        "role": "Metadata / Identifier",
        "unit": "Integer index",
        "meaning": "Source collection index within fraud (1-2179) and normal (1-4729) subsets.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Not Reproducible",
        "calculation_required": "None (collection metadata).",
        "additional_endpoint": "None",
        "assumptions_and_risks": "Artificial collection identifier with target leakage if used. Must be strictly excluded.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": "Address",
        "clean_name": "address",
        "role": "Identifier",
        "unit": "Hexadecimal string",
        "meaning": "Ethereum wallet public address (0x...).",
        "normal_etherscan_tx": "Query Parameter",
        "reproducibility": "Exact",
        "calculation_required": "Supplied directly by user as target wallet for risk evaluation.",
        "additional_endpoint": "None (input query parameter).",
        "assumptions_and_risks": "Primary key for API lookups. Must be excluded from predictive features to prevent memorization.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": "FLAG",
        "clean_name": "flag",
        "role": "Target Label",
        "unit": "Binary flag (0 or 1)",
        "meaning": "Ground truth fraud label (0 = Normal/Non-fraudulent, 1 = Fraudulent/Illicit).",
        "normal_etherscan_tx": "No",
        "reproducibility": "Not Reproducible Dynamically",
        "calculation_required": "Supervised training target; live inference requires model prediction or external threat intel.",
        "additional_endpoint": "Etherscan Label Cloud / CryptScamDB / Chainalysis for benchmark verification.",
        "assumptions_and_risks": "Verified against Kaggle documentation. Imbalance ratio is 3.52:1.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": "Avg min between sent tnx",
        "clean_name": "avg_min_between_sent_tnx",
        "role": "Input Feature",
        "unit": "Minutes",
        "meaning": "Average time interval in minutes between consecutive outgoing (sent) transactions.",
        "normal_etherscan_tx": "Yes",
        "reproducibility": "Exact",
        "calculation_required": (
            "Filter normal transactions where `from == wallet_address` and `isError == '0'`. "
            "Sort ascending by `timeStamp`. If count <= 1, return 0.0. "
            "Else: (max(timeStamp) - min(timeStamp)) / (60.0 * (count - 1))."
        ),
        "additional_endpoint": "None (normal `txlist` endpoint).",
        "assumptions_and_risks": (
            "Etherscan returns max 10,000 transactions per query without pagination. "
            "Truncation on high-volume wallets would underestimate or overestimate interval."
        ),
        "recommended_for_schema": True,
    },
    {
        "csv_column": "Avg min between received tnx",
        "clean_name": "avg_min_between_received_tnx",
        "role": "Input Feature",
        "unit": "Minutes",
        "meaning": "Average time interval in minutes between consecutive incoming (received) transactions.",
        "normal_etherscan_tx": "Yes",
        "reproducibility": "Exact",
        "calculation_required": (
            "Filter normal transactions where `to == wallet_address` and `isError == '0'`. "
            "Sort ascending by `timeStamp`. If count <= 1, return 0.0. "
            "Else: (max(timeStamp) - min(timeStamp)) / (60.0 * (count - 1))."
        ),
        "additional_endpoint": "None for normal txs. Internal contract incoming transfers are omitted to match dataset.",
        "assumptions_and_risks": (
            "Captures only normal incoming transfers. Inflows from smart contracts (DEX, mining) "
            "are omitted unless `txlistinternal` is used, matching training dataset generation."
        ),
        "recommended_for_schema": True,
    },
    {
        "csv_column": "Time Diff between first and last (Mins)",
        "clean_name": "time_diff_first_last_mins",
        "role": "Input Feature",
        "unit": "Minutes",
        "meaning": "Total active observation window: elapsed time between first and most recent normal transaction.",
        "normal_etherscan_tx": "Yes",
        "reproducibility": "Exact",
        "calculation_required": (
            "Across all normal transactions in `txlist`: "
            "if len(txlist) <= 1, return 0.0. Else: (max(timeStamp) - min(timeStamp)) / 60.0."
        ),
        "additional_endpoint": "None (`txlist`).",
        "assumptions_and_risks": (
            "Reflects wallet longevity. If a wallet has only 1 transaction, active window is 0. "
            "Pagination required for wallets with >10,000 transactions to capture true first transaction."
        ),
        "recommended_for_schema": True,
    },
    {
        "csv_column": "Sent tnx",
        "clean_name": "sent_tnx",
        "role": "Input Feature",
        "unit": "Count",
        "meaning": "Number of successful outgoing normal transactions initiated by the wallet.",
        "normal_etherscan_tx": "Yes",
        "reproducibility": "Exact",
        "calculation_required": "Count of records in `txlist` where `from == wallet_address` and `isError == '0'`.",
        "additional_endpoint": "None (`txlist`).",
        "assumptions_and_risks": "Exact match with normal external transaction history.",
        "recommended_for_schema": True,
    },
    {
        "csv_column": "Received Tnx",
        "clean_name": "received_tnx",
        "role": "Input Feature",
        "unit": "Count",
        "meaning": "Number of successful incoming normal transactions received by the wallet.",
        "normal_etherscan_tx": "Yes",
        "reproducibility": "Exact",
        "calculation_required": "Count of records in `txlist` where `to == wallet_address` and `isError == '0'`.",
        "additional_endpoint": "None (`txlist`).",
        "assumptions_and_risks": "Matches training dataset's normal transaction incoming count.",
        "recommended_for_schema": True,
    },
    {
        "csv_column": "Number of Created Contracts",
        "clean_name": "created_contracts_count",
        "role": "Input Feature",
        "unit": "Count",
        "meaning": "Number of smart contract deployment transactions initiated by the wallet.",
        "normal_etherscan_tx": "Yes",
        "reproducibility": "Exact",
        "calculation_required": (
            "Count records in `txlist` where `from == wallet_address` and "
            "(`to == ''` or `to is None` or `to == '0x'` or `contractAddress != ''`)."
        ),
        "additional_endpoint": "None (`txlist`).",
        "assumptions_and_risks": "Standard contracts deployed by wallet. Essential indicator for scam factory detection.",
        "recommended_for_schema": True,
    },
    {
        "csv_column": "Unique Received From Addresses",
        "clean_name": "unique_received_from_addresses",
        "role": "Input Feature",
        "unit": "Count",
        "meaning": "Count of unique sender addresses that have sent normal ETH transactions to this wallet.",
        "normal_etherscan_tx": "Yes",
        "reproducibility": "Exact",
        "calculation_required": "len(set(tx['from'].lower() for tx in txlist if tx['to'].lower() == wallet_address.lower()))",
        "additional_endpoint": "None (`txlist`).",
        "assumptions_and_risks": "Measures sender diversity and fan-in structure.",
        "recommended_for_schema": True,
    },
    {
        "csv_column": "Unique Sent To Addresses",
        "clean_name": "unique_sent_to_addresses",
        "role": "Input Feature",
        "unit": "Count",
        "meaning": "Count of unique recipient addresses that have received normal ETH transactions from this wallet.",
        "normal_etherscan_tx": "Yes",
        "reproducibility": "Exact",
        "calculation_required": "len(set(tx['to'].lower() for tx in txlist if tx['from'].lower() == wallet_address.lower() and tx.get('to')))",
        "additional_endpoint": "None (`txlist`).",
        "assumptions_and_risks": "Measures recipient diversity and fan-out structure.",
        "recommended_for_schema": True,
    },
    {
        "csv_column": "min value received",
        "clean_name": "min_value_received_eth",
        "role": "Input Feature",
        "unit": "Ether (ETH)",
        "meaning": "Minimum ETH value received in a single normal incoming transaction.",
        "normal_etherscan_tx": "Yes",
        "reproducibility": "Exact",
        "calculation_required": "min([float(tx['value'])/1e18 for tx in incoming_txlist]) if incoming_txlist else 0.0",
        "additional_endpoint": "None (`txlist`).",
        "assumptions_and_risks": "Requires converting Wei to Ether by dividing by 10^18.",
        "recommended_for_schema": True,
    },
    {
        "csv_column": "max value received ",
        "clean_name": "max_value_received_eth",
        "role": "Input Feature",
        "unit": "Ether (ETH)",
        "meaning": "Maximum ETH value received in a single normal incoming transaction.",
        "normal_etherscan_tx": "Yes",
        "reproducibility": "Exact",
        "calculation_required": "max([float(tx['value'])/1e18 for tx in incoming_txlist]) if incoming_txlist else 0.0",
        "additional_endpoint": "None (`txlist`).",
        "assumptions_and_risks": "Requires dividing Wei by 10^18. Raw column name contains trailing whitespace in CSV.",
        "recommended_for_schema": True,
    },
    {
        "csv_column": "avg val received",
        "clean_name": "avg_value_received_eth",
        "role": "Input Feature",
        "unit": "Ether (ETH)",
        "meaning": "Arithmetic mean of ETH value received across incoming normal transactions.",
        "normal_etherscan_tx": "Yes",
        "reproducibility": "Exact",
        "calculation_required": "sum([float(tx['value'])/1e18 for tx in incoming_txlist]) / len(incoming_txlist) if incoming_txlist else 0.0",
        "additional_endpoint": "None (`txlist`).",
        "assumptions_and_risks": "Requires dividing Wei by 10^18.",
        "recommended_for_schema": True,
    },
    {
        "csv_column": "min val sent",
        "clean_name": "min_value_sent_eth",
        "role": "Input Feature",
        "unit": "Ether (ETH)",
        "meaning": "Minimum ETH value sent in a single normal outgoing transaction.",
        "normal_etherscan_tx": "Yes",
        "reproducibility": "Exact",
        "calculation_required": "min([float(tx['value'])/1e18 for tx in outgoing_txlist]) if outgoing_txlist else 0.0",
        "additional_endpoint": "None (`txlist`).",
        "assumptions_and_risks": "Zero-value contract interaction transactions return 0.0.",
        "recommended_for_schema": True,
    },
    {
        "csv_column": "max val sent",
        "clean_name": "max_value_sent_eth",
        "role": "Input Feature",
        "unit": "Ether (ETH)",
        "meaning": "Maximum ETH value sent in a single normal outgoing transaction.",
        "normal_etherscan_tx": "Yes",
        "reproducibility": "Exact",
        "calculation_required": "max([float(tx['value'])/1e18 for tx in outgoing_txlist]) if outgoing_txlist else 0.0",
        "additional_endpoint": "None (`txlist`).",
        "assumptions_and_risks": "Requires dividing Wei by 10^18.",
        "recommended_for_schema": True,
    },
    {
        "csv_column": "avg val sent",
        "clean_name": "avg_value_sent_eth",
        "role": "Input Feature",
        "unit": "Ether (ETH)",
        "meaning": "Arithmetic mean of ETH value sent across outgoing normal transactions.",
        "normal_etherscan_tx": "Yes",
        "reproducibility": "Exact",
        "calculation_required": "sum([float(tx['value'])/1e18 for tx in outgoing_txlist]) / len(outgoing_txlist) if outgoing_txlist else 0.0",
        "additional_endpoint": "None (`txlist`).",
        "assumptions_and_risks": "Requires dividing Wei by 10^18.",
        "recommended_for_schema": True,
    },
    {
        "csv_column": "min value sent to contract",
        "clean_name": "min_value_sent_to_contract_eth",
        "role": "Input Feature",
        "unit": "Ether (ETH)",
        "meaning": "Minimum ETH value attached to contract creation transactions.",
        "normal_etherscan_tx": "Yes",
        "reproducibility": "Exact",
        "calculation_required": "min([float(tx['value'])/1e18 for tx in contract_deploy_txs]) if contract_deploy_txs else 0.0",
        "additional_endpoint": "None (`txlist`).",
        "assumptions_and_risks": "Near-zero variance (>99.97% zero in dataset). High sparsity.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": "max val sent to contract",
        "clean_name": "max_value_sent_to_contract_eth",
        "role": "Input Feature",
        "unit": "Ether (ETH)",
        "meaning": "Maximum ETH value attached to contract creation transactions.",
        "normal_etherscan_tx": "Yes",
        "reproducibility": "Exact",
        "calculation_required": "max([float(tx['value'])/1e18 for tx in contract_deploy_txs]) if contract_deploy_txs else 0.0",
        "additional_endpoint": "None (`txlist`).",
        "assumptions_and_risks": "Near-zero variance (>99.97% zero in dataset). High sparsity.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": "avg value sent to contract",
        "clean_name": "avg_value_sent_to_contract_eth",
        "role": "Input Feature",
        "unit": "Ether (ETH)",
        "meaning": "Average ETH value attached to contract creation transactions.",
        "normal_etherscan_tx": "Yes",
        "reproducibility": "Exact",
        "calculation_required": "sum([float(tx['value'])/1e18 for tx in contract_deploy_txs]) / len(contract_deploy_txs) if contract_deploy_txs else 0.0",
        "additional_endpoint": "None (`txlist`).",
        "assumptions_and_risks": "Near-zero variance (>99.97% zero in dataset). High sparsity.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": "total transactions (including tnx to create contract",
        "clean_name": "total_transactions_count",
        "role": "Input Feature (Redundant)",
        "unit": "Count",
        "meaning": "Total count of normal transactions (sent + received + contract creations).",
        "normal_etherscan_tx": "Yes",
        "reproducibility": "Exact",
        "calculation_required": "len(txlist) or (sent_tnx + received_tnx + created_contracts_count)",
        "additional_endpoint": "None (`txlist`).",
        "assumptions_and_risks": "Exact linear combination of sent, received, and contract creation counts.",
        "recommended_for_schema": True,
    },
    {
        "csv_column": "total Ether sent",
        "clean_name": "total_ether_sent",
        "role": "Input Feature",
        "unit": "Ether (ETH)",
        "meaning": "Cumulative total of all ETH value sent across outgoing normal transactions.",
        "normal_etherscan_tx": "Yes",
        "reproducibility": "Exact",
        "calculation_required": "sum([float(tx['value'])/1e18 for tx in outgoing_txlist])",
        "additional_endpoint": "None (`txlist`).",
        "assumptions_and_risks": "Does not include gas fees paid.",
        "recommended_for_schema": True,
    },
    {
        "csv_column": "total ether received",
        "clean_name": "total_ether_received",
        "role": "Input Feature",
        "unit": "Ether (ETH)",
        "meaning": "Cumulative total of all ETH value received across incoming normal transactions.",
        "normal_etherscan_tx": "Yes",
        "reproducibility": "Exact",
        "calculation_required": "sum([float(tx['value'])/1e18 for tx in incoming_txlist])",
        "additional_endpoint": "None (`txlist`).",
        "assumptions_and_risks": (
            "Captures only normal transactions. Inflows from smart contracts (DEX, mining) "
            "are omitted to strictly reproduce training dataset distribution."
        ),
        "recommended_for_schema": True,
    },
    {
        "csv_column": "total ether sent contracts",
        "clean_name": "total_ether_sent_contracts",
        "role": "Input Feature",
        "unit": "Ether (ETH)",
        "meaning": "Cumulative total ETH sent during contract creation transactions.",
        "normal_etherscan_tx": "Yes",
        "reproducibility": "Exact",
        "calculation_required": "sum([float(tx['value'])/1e18 for tx in contract_deploy_txs])",
        "additional_endpoint": "None (`txlist`).",
        "assumptions_and_risks": "Near-zero variance (>99.97% zero in dataset).",
        "recommended_for_schema": False,
    },
    {
        "csv_column": "total ether balance",
        "clean_name": "total_ether_balance",
        "role": "Input Feature (Defective/Synthetic)",
        "unit": "Ether (ETH)",
        "meaning": "Dataset-calculated net balance: total ether received - total Ether sent.",
        "normal_etherscan_tx": "Yes (as formula)",
        "reproducibility": "Approximate / Semantic Conflict",
        "calculation_required": (
            "Formula: `total_ether_received - total_ether_sent`. "
            "DO NOT use Etherscan `balance` endpoint directly, as live balance is >= 0 while "
            "dataset contains negative balances down to -15.6M ETH due to missing internal txs."
        ),
        "additional_endpoint": "Etherscan `module=account&action=balance` returns true EVM balance.",
        "assumptions_and_risks": (
            "CRITICAL TRAIN-INFERENCE RISK: The true EVM balance is never negative. "
            "Using live balance creates severe covariate shift. Live pipeline MUST compute "
            "the formula `total_ether_received - total_ether_sent` to match training semantics."
        ),
        "recommended_for_schema": True,
    },
    {
        "csv_column": " Total ERC20 tnxs",
        "clean_name": "erc20_total_tnx",
        "role": "Input Feature",
        "unit": "Count",
        "meaning": "Total count of ERC-20 token transfer events involving the wallet.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Exact (via ERC20 endpoint)",
        "calculation_required": "len(tokentx_list)",
        "additional_endpoint": "Etherscan `module=account&action=tokentx&address={address}`.",
        "assumptions_and_risks": (
            "Requires calling a dedicated secondary endpoint. For wallets with 0 token transfers, "
            "value must be 0.0 (imputed), avoiding target leakage."
        ),
        "recommended_for_schema": True,
    },
    {
        "csv_column": " ERC20 total Ether received",
        "clean_name": "erc20_total_received_amount",
        "role": "Input Feature (Mislabeled Unit)",
        "unit": "Crude token unit (scaled by 1e18)",
        "meaning": "Mislabeled: Sum of raw token transfer values divided by 10^18 regardless of token decimals.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Approximate (Methodological Flaw)",
        "calculation_required": "sum([float(tx['value'])/1e18 for tx in tokentx if tx['to'].lower() == wallet.lower()])",
        "additional_endpoint": "Etherscan `module=account&action=tokentx`.",
        "assumptions_and_risks": (
            "Methodological flaw in Kaggle dataset: sums heterogeneous tokens without normalising "
            "for decimals (USDT=6, WBTC=8, DAI=18). Live inference must emulate this exact crude calculation."
        ),
        "recommended_for_schema": False,
    },
    {
        "csv_column": " ERC20 total ether sent",
        "clean_name": "erc20_total_sent_amount",
        "role": "Input Feature (Mislabeled Unit)",
        "unit": "Crude token unit (scaled by 1e18)",
        "meaning": "Mislabeled: Sum of raw outgoing token transfer values divided by 10^18.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Approximate (Methodological Flaw)",
        "calculation_required": "sum([float(tx['value'])/1e18 for tx in tokentx if tx['from'].lower() == wallet.lower()])",
        "additional_endpoint": "Etherscan `module=account&action=tokentx`.",
        "assumptions_and_risks": "Methodological flaw in dataset: heterogeneous token values summed without decimal normalization.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": " ERC20 total Ether sent contract",
        "clean_name": "erc20_total_sent_contract_amount",
        "role": "Input Feature",
        "unit": "Crude token unit (scaled by 1e18)",
        "meaning": "Sum of raw token transfer values sent to smart contracts divided by 10^18.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Approximate",
        "calculation_required": "Sum of token values sent to contract addresses divided by 1e18.",
        "additional_endpoint": "Etherscan `tokentx` + smart contract detection (`eth_getCode`).",
        "assumptions_and_risks": "Requires checking contract bytecode for recipient addresses.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": " ERC20 uniq sent addr",
        "clean_name": "erc20_unique_sent_addr",
        "role": "Input Feature",
        "unit": "Count",
        "meaning": "Count of unique recipient addresses for outgoing ERC-20 transfers.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Exact (via ERC20 endpoint)",
        "calculation_required": "len(set(tx['to'].lower() for tx in tokentx if tx['from'].lower() == wallet.lower() and tx.get('to')))",
        "additional_endpoint": "Etherscan `module=account&action=tokentx`.",
        "assumptions_and_risks": "Measures token transfer recipient diversity.",
        "recommended_for_schema": True,
    },
    {
        "csv_column": " ERC20 uniq rec addr",
        "clean_name": "erc20_unique_received_addr",
        "role": "Input Feature",
        "unit": "Count",
        "meaning": "Count of unique sender addresses for incoming ERC-20 transfers.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Exact (via ERC20 endpoint)",
        "calculation_required": "len(set(tx['from'].lower() for tx in tokentx if tx['to'].lower() == wallet.lower()))",
        "additional_endpoint": "Etherscan `module=account&action=tokentx`.",
        "assumptions_and_risks": "Measures token transfer sender diversity.",
        "recommended_for_schema": True,
    },
    {
        "csv_column": " ERC20 uniq sent addr.1",
        "clean_name": "erc20_unique_sent_contract_addr",
        "role": "Input Feature",
        "unit": "Count",
        "meaning": "Count of unique smart contract addresses to which tokens were transferred.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Approximate",
        "calculation_required": "Count distinct contract recipients for token transfers.",
        "additional_endpoint": "Etherscan `tokentx` + contract verification.",
        "assumptions_and_risks": "Header collision in raw CSV; 91.3% zeros. Near-zero variance.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": " ERC20 uniq rec contract addr",
        "clean_name": "erc20_unique_received_contract_addr",
        "role": "Input Feature",
        "unit": "Count",
        "meaning": "Count of unique token contract addresses (distinct ERC-20 assets) received by the wallet.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Exact (via ERC20 endpoint)",
        "calculation_required": "len(set(tx['contractAddress'].lower() for tx in tokentx if tx['to'].lower() == wallet.lower()))",
        "additional_endpoint": "Etherscan `module=account&action=tokentx`.",
        "assumptions_and_risks": "Measures token portfolio breadth and airdrop exposure.",
        "recommended_for_schema": True,
    },
    {
        "csv_column": " ERC20 avg time between sent tnx",
        "clean_name": "erc20_avg_time_between_sent_tnx",
        "role": "Constant Feature",
        "unit": "Minutes",
        "meaning": "Average time between sent token transactions.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Not Reproducible / Zero-Variance",
        "calculation_required": "None (100% constant zeros in dataset).",
        "additional_endpoint": "None",
        "assumptions_and_risks": "Constant 0.0 across all records. Excluded.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": " ERC20 avg time between rec tnx",
        "clean_name": "erc20_avg_time_between_rec_tnx",
        "role": "Constant Feature",
        "unit": "Minutes",
        "meaning": "Average time between received token transactions.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Not Reproducible / Zero-Variance",
        "calculation_required": "None (100% constant zeros in dataset).",
        "additional_endpoint": "None",
        "assumptions_and_risks": "Constant 0.0 across all records. Excluded.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": " ERC20 avg time between rec 2 tnx",
        "clean_name": "erc20_avg_time_between_rec_2_tnx",
        "role": "Constant Feature",
        "unit": "Minutes",
        "meaning": "Average time between received 2 transactions.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Not Reproducible / Zero-Variance",
        "calculation_required": "None (100% constant zeros in dataset).",
        "additional_endpoint": "None",
        "assumptions_and_risks": "Constant 0.0 across all records. Excluded.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": " ERC20 avg time between contract tnx",
        "clean_name": "erc20_avg_time_between_contract_tnx",
        "role": "Constant Feature",
        "unit": "Minutes",
        "meaning": "Average time between contract token transactions.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Not Reproducible / Zero-Variance",
        "calculation_required": "None (100% constant zeros in dataset).",
        "additional_endpoint": "None",
        "assumptions_and_risks": "Constant 0.0 across all records. Excluded.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": " ERC20 min val rec",
        "clean_name": "erc20_min_val_received",
        "role": "Input Feature",
        "unit": "Crude token unit (scaled by 1e18)",
        "meaning": "Minimum token value received divided by 10^18.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Approximate",
        "calculation_required": "min([float(tx['value'])/1e18 for tx in tokentx_in]) if tokentx_in else 0.0",
        "additional_endpoint": "Etherscan `module=account&action=tokentx`.",
        "assumptions_and_risks": "Token decimal heterogeneity distorts values across different assets.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": " ERC20 max val rec",
        "clean_name": "erc20_max_val_received",
        "role": "Input Feature",
        "unit": "Crude token unit (scaled by 1e18)",
        "meaning": "Maximum token value received divided by 10^18.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Approximate",
        "calculation_required": "max([float(tx['value'])/1e18 for tx in tokentx_in]) if tokentx_in else 0.0",
        "additional_endpoint": "Etherscan `module=account&action=tokentx`.",
        "assumptions_and_risks": "Distorted by token decimal differences.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": " ERC20 avg val rec",
        "clean_name": "erc20_avg_val_received",
        "role": "Input Feature",
        "unit": "Crude token unit (scaled by 1e18)",
        "meaning": "Average token value received divided by 10^18.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Approximate",
        "calculation_required": "Average of token values divided by 1e18.",
        "additional_endpoint": "Etherscan `module=account&action=tokentx`.",
        "assumptions_and_risks": "Distorted by token decimal differences.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": " ERC20 min val sent",
        "clean_name": "erc20_min_val_sent",
        "role": "Input Feature",
        "unit": "Crude token unit (scaled by 1e18)",
        "meaning": "Minimum token value sent divided by 10^18.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Approximate",
        "calculation_required": "min([float(tx['value'])/1e18 for tx in tokentx_out]) if tokentx_out else 0.0",
        "additional_endpoint": "Etherscan `module=account&action=tokentx`.",
        "assumptions_and_risks": "Distorted by token decimal differences.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": " ERC20 max val sent",
        "clean_name": "erc20_max_val_sent",
        "role": "Input Feature",
        "unit": "Crude token unit (scaled by 1e18)",
        "meaning": "Maximum token value sent divided by 10^18.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Approximate",
        "calculation_required": "max([float(tx['value'])/1e18 for tx in tokentx_out]) if tokentx_out else 0.0",
        "additional_endpoint": "Etherscan `module=account&action=tokentx`.",
        "assumptions_and_risks": "Distorted by token decimal differences.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": " ERC20 avg val sent",
        "clean_name": "erc20_avg_val_sent",
        "role": "Input Feature",
        "unit": "Crude token unit (scaled by 1e18)",
        "meaning": "Average token value sent divided by 10^18.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Approximate",
        "calculation_required": "Average of sent token values divided by 1e18.",
        "additional_endpoint": "Etherscan `module=account&action=tokentx`.",
        "assumptions_and_risks": "Distorted by token decimal differences.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": " ERC20 min val sent contract",
        "clean_name": "erc20_min_val_sent_contract",
        "role": "Constant Feature",
        "unit": "Crude token unit",
        "meaning": "Minimum token value sent to contract.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Not Reproducible / Zero-Variance",
        "calculation_required": "None (100% constant zeros in dataset).",
        "additional_endpoint": "None",
        "assumptions_and_risks": "Constant 0.0 across all records. Excluded.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": " ERC20 max val sent contract",
        "clean_name": "erc20_max_val_sent_contract",
        "role": "Constant Feature",
        "unit": "Crude token unit",
        "meaning": "Maximum token value sent to contract.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Not Reproducible / Zero-Variance",
        "calculation_required": "None (100% constant zeros in dataset).",
        "additional_endpoint": "None",
        "assumptions_and_risks": "Constant 0.0 across all records. Excluded.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": " ERC20 avg val sent contract",
        "clean_name": "erc20_avg_val_sent_contract",
        "role": "Constant Feature",
        "unit": "Crude token unit",
        "meaning": "Average token value sent to contract.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Not Reproducible / Zero-Variance",
        "calculation_required": "None (100% constant zeros in dataset).",
        "additional_endpoint": "None",
        "assumptions_and_risks": "Constant 0.0 across all records. Excluded.",
        "recommended_for_schema": False,
    },
    {
        "csv_column": " ERC20 uniq sent token name",
        "clean_name": "erc20_unique_sent_token_name",
        "role": "Input Feature",
        "unit": "Count",
        "meaning": "Number of unique token symbols sent by the wallet.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Exact (via ERC20 endpoint)",
        "calculation_required": "len(set(tx['tokenSymbol'] for tx in tokentx if tx['from'].lower() == wallet.lower()))",
        "additional_endpoint": "Etherscan `module=account&action=tokentx`.",
        "assumptions_and_risks": "Count of distinct outbound token symbols.",
        "recommended_for_schema": True,
    },
    {
        "csv_column": " ERC20 uniq rec token name",
        "clean_name": "erc20_unique_received_token_name",
        "role": "Input Feature",
        "unit": "Count",
        "meaning": "Number of unique token symbols received by the wallet.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Exact (via ERC20 endpoint)",
        "calculation_required": "len(set(tx['tokenSymbol'] for tx in tokentx if tx['to'].lower() == wallet.lower()))",
        "additional_endpoint": "Etherscan `module=account&action=tokentx`.",
        "assumptions_and_risks": "Count of distinct inbound token symbols (airdrop spam, tokens).",
        "recommended_for_schema": True,
    },
    {
        "csv_column": " ERC20 most sent token type",
        "clean_name": "erc20_most_sent_token_type",
        "role": "Categorical Feature",
        "unit": "Token symbol/name string",
        "meaning": "Name/symbol of the token sent most frequently.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Approximate / Temporal Drift",
        "calculation_required": "Mode of `tokenSymbol` for outgoing `tokentx` events.",
        "additional_endpoint": "Etherscan `module=account&action=tokentx`.",
        "assumptions_and_risks": (
            "Extremely high cardinality (304 unique names), heavy temporal drift, "
            "and spam token vulnerability. Not recommended for core feature schema."
        ),
        "recommended_for_schema": False,
    },
    {
        "csv_column": " ERC20_most_rec_token_type",
        "clean_name": "erc20_most_received_token_type",
        "role": "Categorical Feature",
        "unit": "Token symbol/name string",
        "meaning": "Name/symbol of the token received most frequently.",
        "normal_etherscan_tx": "No",
        "reproducibility": "Approximate / Temporal Drift",
        "calculation_required": "Mode of `tokenSymbol` for incoming `tokentx` events.",
        "additional_endpoint": "Etherscan `module=account&action=tokentx`.",
        "assumptions_and_risks": (
            "High cardinality (466 unique names), heavily contaminated by promotional airdrop tokens "
            "(e.g., Livepeer, OmiseGO, INS Promo). Not recommended for core schema."
        ),
        "recommended_for_schema": False,
    },
]


def generate_feature_audit_reports(
    csv_out_path: Optional[Path] = None,
    md_out_path: Optional[Path] = None,
    schema_out_path: Optional[Path] = None,
) -> Tuple[Path, Path, Path]:
    """
    Generates:
      1. reports/feature_compatibility_audit.csv
      2. reports/feature_compatibility_audit.md
      3. reports/common_feature_schema.json
    """
    csv_path = csv_out_path if csv_out_path is not None else config.FEATURE_AUDIT_CSV_PATH
    md_path = md_out_path if md_out_path is not None else config.FEATURE_AUDIT_MD_PATH
    schema_path = schema_out_path if schema_out_path is not None else config.COMMON_FEATURE_SCHEMA_PATH

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Enrich FEATURE_AUDIT_DATA with strict Parity Tiers and Unresolved Issue documentation
    for item in FEATURE_AUDIT_DATA:
        clean = item["clean_name"]
        if not item["recommended_for_schema"]:
            item["parity_tier"] = "Excluded from Schema"
            item["failed_tx_handling"] = "N/A (Excluded)"
            item["unresolved_issues"] = "Excluded due to zero variance, identifier status, unit flaw, or high cardinality."
        elif clean in ["total_transactions_count"]:
            item["parity_tier"] = "Approximate equivalent"
            item["failed_tx_handling"] = "Unresolved (inherits underlying sent/received failed tx uncertainty)"
            item["unresolved_issues"] = (
                "Algebraically redundant: exactly equals sent_tnx + received_tnx + created_contracts_count. "
                "Multicollinear in linear models."
            )
        elif clean in ["total_ether_balance"]:
            item["parity_tier"] = "Approximate equivalent"
            item["failed_tx_handling"] = "Unresolved (inherits underlying sent/received failed tx uncertainty)"
            item["unresolved_issues"] = (
                "Synthetic flow formula (received - sent). Cannot match true on-chain EVM balance. "
                "Omits internal transactions, yielding 978 negative balances."
            )
        elif clean in ["total_ether_received"]:
            item["parity_tier"] = "Approximate equivalent"
            item["failed_tx_handling"] = "Unresolved (cannot verify if failed incoming txs were counted)"
            item["unresolved_issues"] = (
                "Omits incoming ETH from internal transactions (DEX swaps, contract withdrawals, mining). "
                "Inclusion of attempted values from failed transactions cannot be established."
            )
        elif clean.startswith("erc20_"):
            item["parity_tier"] = "Approximate equivalent"
            item["failed_tx_handling"] = "Excluded by EVM semantics (tokentx event logs emit only on successful execution)"
            item["unresolved_issues"] = (
                "829 fraud rows have NaN in training data. Zero-imputation is a pragmatic modeling assumption, "
                "not confirmed on-chain ground truth. erc20_unique_received_contract_addr is highly collinear "
                "with erc20_unique_received_token_name (r = 0.9996)."
            )
        else:
            # All remaining base ETH features from txlist
            item["parity_tier"] = "Not verifiable from the available dataset"
            item["failed_tx_handling"] = "Unresolved (raw transaction logs and isError flags discarded in CSV)"
            item["unresolved_issues"] = (
                "The Kaggle dataset does not document whether failed transactions (isError == '1') were excluded "
                "or included in counts, intervals, and value statistics. Cannot be determined from tabular data alone."
            )

    # 1. Save CSV audit table
    audit_df = pd.DataFrame(FEATURE_AUDIT_DATA)
    audit_df.to_csv(csv_path, index=False)

    # 2. Build Markdown audit report
    md_lines = [
        "# Etherscan API V2 Feature Compatibility Audit (Pre-Training Verification)",
        "## Ethereum Wallet Fraud Detection Machine Learning Study (ChainSentinel) - Phase 1\n",
        "This audit systematically examines every single one of the 51 columns in the Kaggle Ethereum Fraud Detection dataset.",
        "It determines whether and how each feature can be reconstructed for a live Ethereum wallet using Etherscan API V2 endpoints (`txlist`, `tokentx`, `txlistinternal`, `balance`).\n",
        "### Rigorous Parity Classification Framework",
        "In strict compliance with academic auditing requirements, features are classified into three parity tiers:",
        "- **Verified equivalent**: Mathematical definition, unit, directional logic, and edge-case behavior are proven identical between training and live calculation. *(Result: 0 features qualify, because raw transaction logs were discarded by the dataset compiler)*.",
        "- **Approximate equivalent**: Feature is close in concept and units, but differences in historical vs live calculation exist (algebraic redundancy, missing internal transactions, or token zero-imputation assumptions). *(Result: 9 features)*.",
        "- **Not verifiable from the available dataset**: Parity cannot be proven because the Kaggle tabular summary discarded transaction-level error status (`isError`). *(Result: 15 features)*.",
        "\n---",
        "## Full 51-Feature Compatibility Matrix\n",
    ]

    table_cols = [
        "csv_column",
        "role",
        "unit",
        "parity_tier",
        "failed_tx_handling",
        "additional_endpoint",
        "recommended_for_schema",
    ]
    md_table_headers = "| Column in CSV | Role | Unit | Parity Classification | Failed Tx Inclusion | Additional Endpoint | In Schema? |"
    md_table_sep = "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    md_lines.append(md_table_headers)
    md_lines.append(md_table_sep)

    for item in FEATURE_AUDIT_DATA:
        in_schema_badge = "✅ Yes" if item["recommended_for_schema"] else "❌ Excluded"
        md_lines.append(
            f"| `{item['csv_column']}` | {item['role']} | {item['unit']} | "
            f"**{item['parity_tier']}** | {item['failed_tx_handling']} | "
            f"`{item['additional_endpoint']}` | {in_schema_badge} |"
        )

    md_lines.append("\n---\n")
    md_lines.append("## Detailed Technical Audit by Feature Group\n")

    groups = {
        "Base ETH Transaction Features (Exact Reproducibility via `txlist`)": [
            "Avg min between sent tnx", "Avg min between received tnx",
            "Time Diff between first and last (Mins)", "Sent tnx", "Received Tnx",
            "Number of Created Contracts", "Unique Received From Addresses",
            "Unique Sent To Addresses", "min value received", "max value received ",
            "avg val received", "min val sent", "max val sent", "avg val sent",
            "total transactions (including tnx to create contract",
            "total Ether sent", "total ether received"
        ],
        "Anomalous / Defective Balance Feature": [
            "total ether balance"
        ],
        "Contract Creation Value Features (Near-Zero Variance)": [
            "min value sent to contract", "max val sent to contract",
            "avg value sent to contract", "total ether sent contracts"
        ],
        "ERC-20 Asset & Frequency Features (Requires `tokentx`)": [
            " Total ERC20 tnxs", " ERC20 uniq sent addr", " ERC20 uniq rec addr",
            " ERC20 uniq rec contract addr", " ERC20 uniq sent token name",
            " ERC20 uniq rec token name"
        ],
        "Flawed / Heterogeneous Token Value Sums (Unit Distortion Risk)": [
            " ERC20 total Ether received", " ERC20 total ether sent",
            " ERC20 total Ether sent contract", " ERC20 min val rec",
            " ERC20 max val rec", " ERC20 avg val rec", " ERC20 min val sent",
            " ERC20 max val sent", " ERC20 avg val sent", " ERC20 uniq sent addr.1"
        ],
        "Zero-Variance Features (100% Constant in Training Dataset)": [
            " ERC20 avg time between sent tnx", " ERC20 avg time between rec tnx",
            " ERC20 avg time between rec 2 tnx", " ERC20 avg time between contract tnx",
            " ERC20 min val sent contract", " ERC20 max val sent contract",
            " ERC20 avg val sent contract"
        ],
        "Categorical Token Identifiers (High Cardinality / Temporal Drift)": [
            " ERC20 most sent token type", " ERC20_most_rec_token_type"
        ],
    }

    for group_title, col_names in groups.items():
        md_lines.append(f"### {group_title}\n")
        for col_name in col_names:
            match = next((item for item in FEATURE_AUDIT_DATA if item["csv_column"] == col_name), None)
            if match:
                md_lines.append(f"#### `{match['csv_column']}` (Standardized: `{match['clean_name']}`)")
                md_lines.append(f"- **Role**: {match['role']}")
                md_lines.append(f"- **Unit**: {match['unit']}")
                md_lines.append(f"- **Calculation Required**: {match['calculation_required']}")
                md_lines.append(f"- **Endpoint Dependency**: `{match['additional_endpoint']}`")
                md_lines.append(f"- **Reproducibility Rating**: **{match['reproducibility']}**")
                md_lines.append(f"- **Assumptions and Risks**: {match['assumptions_and_risks']}\n")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # 3. Build Canonical Common Feature Schema (JSON)
    # Recommended features that can be reproducibly calculated from Etherscan and exist in training
    recommended_items = [item for item in FEATURE_AUDIT_DATA if item["recommended_for_schema"]]
    
    # Precise field and endpoint mappings for the 24 canonical features
    FEATURE_SOURCE_CONFIG: Dict[str, Dict[str, Any]] = {
        # Base ETH features from txlist
        "avg_min_between_sent_tnx": {
            "endpoint": "module=account&action=txlist&address={address}",
            "fields": ["from", "to", "timeStamp", "isError", "contractAddress"],
        },
        "avg_min_between_received_tnx": {
            "endpoint": "module=account&action=txlist&address={address}",
            "fields": ["to", "timeStamp", "isError"],
        },
        "time_diff_first_last_mins": {
            "endpoint": "module=account&action=txlist&address={address}",
            "fields": ["timeStamp", "isError"],
        },
        "sent_tnx": {
            "endpoint": "module=account&action=txlist&address={address}",
            "fields": ["from", "to", "isError", "contractAddress"],
        },
        "received_tnx": {
            "endpoint": "module=account&action=txlist&address={address}",
            "fields": ["to", "isError"],
        },
        "created_contracts_count": {
            "endpoint": "module=account&action=txlist&address={address}",
            "fields": ["from", "to", "contractAddress", "isError"],
        },
        "unique_received_from_addresses": {
            "endpoint": "module=account&action=txlist&address={address}",
            "fields": ["from", "to", "isError"],
        },
        "unique_sent_to_addresses": {
            "endpoint": "module=account&action=txlist&address={address}",
            "fields": ["from", "to", "isError", "contractAddress"],
        },
        "min_value_received_eth": {
            "endpoint": "module=account&action=txlist&address={address}",
            "fields": ["to", "value", "isError"],
        },
        "max_value_received_eth": {
            "endpoint": "module=account&action=txlist&address={address}",
            "fields": ["to", "value", "isError"],
        },
        "avg_value_received_eth": {
            "endpoint": "module=account&action=txlist&address={address}",
            "fields": ["to", "value", "isError"],
        },
        "min_value_sent_eth": {
            "endpoint": "module=account&action=txlist&address={address}",
            "fields": ["from", "to", "value", "isError", "contractAddress"],
        },
        "max_value_sent_eth": {
            "endpoint": "module=account&action=txlist&address={address}",
            "fields": ["from", "to", "value", "isError", "contractAddress"],
        },
        "avg_value_sent_eth": {
            "endpoint": "module=account&action=txlist&address={address}",
            "fields": ["from", "to", "value", "isError", "contractAddress"],
        },
        "total_transactions_count": {
            "endpoint": "module=account&action=txlist&address={address}",
            "fields": ["from", "to", "isError", "contractAddress"],
        },
        "total_ether_sent": {
            "endpoint": "module=account&action=txlist&address={address}",
            "fields": ["from", "to", "value", "isError", "contractAddress"],
        },
        "total_ether_received": {
            "endpoint": "module=account&action=txlist&address={address}",
            "fields": ["to", "value", "isError"],
        },
        "total_ether_balance": {
            "endpoint": "module=account&action=txlist&address={address} (reconstructed net flow)",
            "fields": ["from", "to", "value", "isError"],
        },
        # ERC-20 features from tokentx
        "erc20_total_tnx": {
            "endpoint": "module=account&action=tokentx&address={address}",
            "fields": ["from", "to", "contractAddress"],
        },
        "erc20_unique_sent_addr": {
            "endpoint": "module=account&action=tokentx&address={address}",
            "fields": ["from", "to"],
        },
        "erc20_unique_received_addr": {
            "endpoint": "module=account&action=tokentx&address={address}",
            "fields": ["from", "to"],
        },
        "erc20_unique_received_contract_addr": {
            "endpoint": "module=account&action=tokentx&address={address}",
            "fields": ["to", "contractAddress"],
        },
        "erc20_unique_sent_token_name": {
            "endpoint": "module=account&action=tokentx&address={address}",
            "fields": ["from", "tokenSymbol"],
        },
        "erc20_unique_received_token_name": {
            "endpoint": "module=account&action=tokentx&address={address}",
            "fields": ["to", "tokenSymbol"],
        },
    }

    schema_definitions = []
    for item in recommended_items:
        conf = FEATURE_SOURCE_CONFIG.get(item["clean_name"], {
            "endpoint": "module=account&action=txlist&address={address}",
            "fields": ["from", "to", "value", "timeStamp", "isError"],
        })
        source_endpoint = conf["endpoint"]
        source_fields = conf["fields"]

        schema_definitions.append({
            "canonical_feature_name": item["clean_name"],
            "data_type": "float64" if "eth" in item["clean_name"] or "mins" in item["clean_name"] or "balance" in item["clean_name"] or "avg" in item["clean_name"] else "int64",
            "unit": item["unit"],
            "parity_tier": item["parity_tier"],
            "failed_transaction_handling": item["failed_tx_handling"],
            "calculation_formula": item["calculation_required"],
            "required_source_fields": source_fields,
            "source_endpoint": source_endpoint,
            "missing_value_policy": "Default to 0.0 if no qualifying transactions exist.",
            "observation_period": "Full wallet transaction history (from genesis transaction to latest block).",
            "training_dataset_mapping": item["csv_column"],
            "unresolved_issues_and_risks": item["unresolved_issues"],
            "etherscan_data_mapping": {
                "endpoint": source_endpoint,
                "reproducibility": item["reproducibility"],
                "notes": item["assumptions_and_risks"],
            },
        })

    schema_metadata = {
        "schema_version": "1.0.0",
        "description": (
            "Canonical Common Feature Schema bridging the Kaggle Ethereum Fraud Dataset "
            "and the live Etherscan API V2 pipeline for Phase 2 model training and Phase 3 inference."
        ),
        "total_recommended_features": len(schema_definitions),
        "feature_selection_rationale": (
            f"Selected {len(schema_definitions)} high-integrity features that are mathematically reproducible from Etherscan "
            "API V2 (`txlist` and `tokentx`), free from zero-variance degradation, and free from "
            "crude heterogeneous token value summing flaws. Excluded pure identifiers, zero-variance "
            "constants, high-cardinality transient token strings, and unscaled token amounts."
        ),
        "etherscan_api_version": "API V2 (https://docs.etherscan.io/)",
        "features": schema_definitions,
        "recommendations_and_limitations": {
            "negative_balance_resolution": (
                "The training dataset computed balance as `total_ether_received - total_ether_sent` "
                "without accounting for internal contract transactions. Live inference MUST compute "
                "this exact same net flow formula rather than using Etherscan's `balance` endpoint."
            ),
            "erc20_zero_imputation": (
                "Wallets with zero ERC-20 activity must have their token features imputed to 0.0. "
                "A missingness indicator must never be added to avoid target leakage."
            ),
            "pagination_policy": (
                "For wallets exceeding 10,000 transactions, Etherscan API V2 block pagination "
                "(`startblock` and `endblock`) is required to capture the entire activity lifespan."
            ),
        },
    }

    save_json(schema_metadata, schema_path)

    return csv_path, md_path, schema_path


def run_feature_audit() -> None:
    """CLI runner for feature compatibility audit."""
    print("[*] Starting Etherscan API V2 Feature Compatibility Audit...")
    csv_file, md_file, schema_file = generate_feature_audit_reports()
    print(f"[+] Feature audit complete.")
    print(f"    - Audit CSV: {csv_file}")
    print(f"    - Audit Markdown Report: {md_file}")
    print(f"    - Common Feature Schema: {schema_file}")


if __name__ == "__main__":
    run_feature_audit()
