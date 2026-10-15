"""
Data Preprocessing Pipeline for Ethereum Wallet Fraud Detection (Phase 1).
Implements reproducible cleaning, deduplication, conflict checking,
identifier exclusion, and leakage-safe missing value imputation.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np

import config
from src.utils import find_raw_dataset, clean_column_name, save_json


# Canonical mapping for problematic or duplicate raw column names
RAW_TO_CANONICAL_COLUMNS: Dict[str, str] = {
    "Unnamed: 0": "raw_row_id",
    "Index": "dataset_index",
    "Address": "address",
    "FLAG": "flag",
    "Avg min between sent tnx": "avg_min_between_sent_tnx",
    "Avg min between received tnx": "avg_min_between_received_tnx",
    "Time Diff between first and last (Mins)": "time_diff_first_last_mins",
    "Sent tnx": "sent_tnx",
    "Received Tnx": "received_tnx",
    "Number of Created Contracts": "created_contracts_count",
    "Unique Received From Addresses": "unique_received_from_addresses",
    "Unique Sent To Addresses": "unique_sent_to_addresses",
    "min value received": "min_value_received_eth",
    "max value received ": "max_value_received_eth",
    "avg val received": "avg_value_received_eth",
    "min val sent": "min_value_sent_eth",
    "max val sent": "max_value_sent_eth",
    "avg val sent": "avg_value_sent_eth",
    "min value sent to contract": "min_value_sent_to_contract_eth",
    "max val sent to contract": "max_value_sent_to_contract_eth",
    "avg value sent to contract": "avg_value_sent_to_contract_eth",
    "total transactions (including tnx to create contract": "total_transactions_count",
    "total Ether sent": "total_ether_sent",
    "total ether received": "total_ether_received",
    "total ether sent contracts": "total_ether_sent_contracts",
    "total ether balance": "total_ether_balance",
    " Total ERC20 tnxs": "erc20_total_tnx",
    " ERC20 total Ether received": "erc20_total_received_amount",
    " ERC20 total ether sent": "erc20_total_sent_amount",
    " ERC20 total Ether sent contract": "erc20_total_sent_contract_amount",
    " ERC20 uniq sent addr": "erc20_unique_sent_addr",
    " ERC20 uniq rec addr": "erc20_unique_received_addr",
    " ERC20 uniq sent addr.1": "erc20_unique_sent_contract_addr",  # Raw header duplicate
    " ERC20 uniq rec contract addr": "erc20_unique_received_contract_addr",
    " ERC20 avg time between sent tnx": "erc20_avg_time_between_sent_tnx",
    " ERC20 avg time between rec tnx": "erc20_avg_time_between_rec_tnx",
    " ERC20 avg time between rec 2 tnx": "erc20_avg_time_between_rec_2_tnx",
    " ERC20 avg time between contract tnx": "erc20_avg_time_between_contract_tnx",
    " ERC20 min val rec": "erc20_min_val_received",
    " ERC20 max val rec": "erc20_max_val_received",
    " ERC20 avg val rec": "erc20_avg_val_received",
    " ERC20 min val sent": "erc20_min_val_sent",
    " ERC20 max val sent": "erc20_max_val_sent",
    " ERC20 avg val sent": "erc20_avg_val_sent",
    " ERC20 min val sent contract": "erc20_min_val_sent_contract",
    " ERC20 max val sent contract": "erc20_max_val_sent_contract",
    " ERC20 avg val sent contract": "erc20_avg_val_sent_contract",
    " ERC20 uniq sent token name": "erc20_unique_sent_token_name",
    " ERC20 uniq rec token name": "erc20_unique_received_token_name",
    " ERC20 most sent token type": "erc20_most_sent_token_type",
    " ERC20_most_rec_token_type": "erc20_most_received_token_type",
}


def validate_raw_dataframe(df: pd.DataFrame, target_col: str = config.TARGET_COLUMN) -> None:
    """
    Validates structural requirements and constraints of the raw dataframe.
    Raises ValueError on empty data, missing target column, or unexpected labels.
    """
    if df.empty:
        raise ValueError("Input dataframe is completely empty (0 records).")

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataframe columns: {list(df.columns)}"
        )

    # Check target values
    unique_targets = set(df[target_col].dropna().unique())
    expected_targets = {0, 1}
    if not unique_targets.issubset(expected_targets):
        raise ValueError(
            f"Unexpected target labels detected in '{target_col}': {unique_targets}. "
            f"Expected binary labels {expected_targets}."
        )


def check_conflicting_labels(df: pd.DataFrame, address_col: str = "Address", target_col: str = config.TARGET_COLUMN) -> None:
    """
    Checks whether any duplicate wallet address has conflicting target labels.
    Raises ValueError if conflicting labels are discovered.
    """
    if address_col not in df.columns or target_col not in df.columns:
        return

    grouped = df.groupby(address_col)[target_col].nunique()
    conflicts = grouped[grouped > 1]
    if not conflicts.empty:
        raise ValueError(
            f"Data integrity error: Discovered {len(conflicts)} wallet addresses with conflicting "
            f"target labels! Conflicting addresses: {list(conflicts.index)[:5]}"
        )


def preprocess_dataset(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Executes the Phase 1 reproducible preprocessing pipeline.
    
    Returns:
        cleaned_df: Full cleaned dataset including canonical metadata.
        X: Feature matrix strictly excluding identifiers and constant columns.
        y: Target series (FLAG).
        metadata: Preprocessing summary and schema documentation.
    """
    raw_path = find_raw_dataset(input_path)
    save_path = output_path if output_path is not None else config.PROCESSED_DATASET_PATH
    meta_path = metadata_path if metadata_path is not None else config.FEATURE_METADATA_PATH

    # Read raw dataset in read-only mode
    raw_df = pd.read_csv(raw_path)
    initial_rows, initial_cols = raw_df.shape

    # 1. Validation and integrity checks
    validate_raw_dataframe(raw_df, target_col=config.TARGET_COLUMN)
    check_conflicting_labels(raw_df, address_col="Address", target_col=config.TARGET_COLUMN)

    # 2. Standardize column names using canonical mapping
    col_mapping = {
        col: RAW_TO_CANONICAL_COLUMNS.get(col, clean_column_name(col))
        for col in raw_df.columns
    }
    df = raw_df.rename(columns=col_mapping).copy()

    # 3. Deduplication: deduplicate on wallet address keeping first occurrence
    # (Verified that no conflicting labels exist across duplicates)
    pre_dedup_rows = len(df)
    df = df.drop_duplicates(subset=["address"], keep="first").reset_index(drop=True)
    dedup_removed = pre_dedup_rows - len(df)

    # 4. Handle Missing Values
    # Numerical ERC-20 columns: The 829 null rows correspond to wallets where no ERC20
    # transactions were found during scraping. We impute with 0.0 (logically 0 transactions).
    # We explicitly DO NOT add an is_missing indicator to avoid 100% target leakage.
    num_erc20_cols = [
        c for c in df.columns
        if c.startswith("erc20_") and c not in ["erc20_most_sent_token_type", "erc20_most_received_token_type"]
    ]
    imputed_num_erc20_counts = int(df[num_erc20_cols].isnull().sum().max())
    df[num_erc20_cols] = df[num_erc20_cols].fillna(0.0)

    # Categorical ERC-20 columns: Clean whitespace, '0', and NaN to 'None'
    cat_erc20_cols = ["erc20_most_sent_token_type", "erc20_most_received_token_type"]
    for c in cat_erc20_cols:
        if c in df.columns:
            df[c] = df[c].fillna("None").astype(str).str.strip()
            df[c] = df[c].replace({"0": "None", "": "None"})

    # 5. Handle Constant / Zero-Variance Columns
    canonical_constant_cols = [
        RAW_TO_CANONICAL_COLUMNS.get(c, clean_column_name(c))
        for c in config.CONSTANT_COLUMNS
    ]
    constant_cols_to_drop = [c for c in canonical_constant_cols if c in df.columns]

    # 6. Separate Identifiers, Features X, and Target y
    canonical_id_cols = ["raw_row_id", "dataset_index", "address"]
    target_col = "flag"

    # Identify candidate feature columns
    excluded_from_features = set(canonical_id_cols + [target_col] + constant_cols_to_drop)
    feature_cols = [c for c in df.columns if c not in excluded_from_features]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # 7. Document Class Imbalance
    target_distribution = y.value_counts().to_dict()
    total_samples = len(y)
    class_imbalance_report = {
        "class_0_count": int(target_distribution.get(0, 0)),
        "class_0_pct": round(float((target_distribution.get(0, 0) / total_samples) * 100), 2),
        "class_1_count": int(target_distribution.get(1, 0)),
        "class_1_pct": round(float((target_distribution.get(1, 0) / total_samples) * 100), 2),
        "imbalance_ratio": round(float(target_distribution.get(0, 0) / max(1, target_distribution.get(1, 0))), 2),
    }

    # 8. Save cleaned dataset and metadata report
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)

    metadata = {
        "raw_dataset_source": str(raw_path.name),
        "initial_dimensions": {"rows": initial_rows, "cols": initial_cols},
        "cleaned_dimensions": {"rows": len(df), "cols": len(df.columns)},
        "feature_matrix_dimensions": {"rows": len(X), "features": len(X.columns)},
        "deduplication": {
            "strategy": "drop_duplicates by wallet address (keep='first')",
            "removed_count": dedup_removed,
            "conflicts_detected": 0,
        },
        "missing_value_policy": {
            "numerical_erc20": "Imputed with 0.0 (representing zero token transfers; avoids missingness target leakage).",
            "categorical_erc20": "Sanitized whitespace, '0', and NaN to 'None'.",
            "imputed_records_count": imputed_num_erc20_counts,
        },
        "excluded_columns": {
            "identifiers": canonical_id_cols,
            "constant_columns": constant_cols_to_drop,
        },
        "target_column": target_col,
        "class_imbalance": class_imbalance_report,
        "feature_list": feature_cols,
        "column_name_mapping": col_mapping,
        "pipeline_rules": [
            "Raw data in data/raw/ is strictly read-only and preserved without modification.",
            "Pure identifiers (Address, Index, Unnamed: 0) are strictly excluded from predictive features.",
            "Zero-variance constant columns are dropped.",
            "No synthetic oversampling (e.g. SMOTE) or undersampling is applied in Phase 1.",
            "No dataset-wide normalization/scaling is applied in Phase 1 to prevent test set data snooping.",
        ],
    }
    save_json(metadata, meta_path)

    return df, X, y, metadata


def run_preprocessing() -> None:
    """CLI runner for the preprocessing pipeline."""
    print("[*] Running Phase 1 data preprocessing pipeline...")
    df, X, y, meta = preprocess_dataset()
    print(f"[+] Preprocessing complete.")
    print(f"    - Cleaned rows: {len(df):,} (removed {meta['deduplication']['removed_count']} duplicates)")
    print(f"    - Feature count in X: {X.shape[1]}")
    print(f"    - Target distribution: {meta['class_imbalance']}")
    print(f"    - Cleaned dataset saved to: {config.PROCESSED_DATASET_PATH}")
    print(f"    - Feature metadata saved to: {config.FEATURE_METADATA_PATH}")


if __name__ == "__main__":
    run_preprocessing()
