"""
Automated unit tests for data preprocessing and validation logic.
Tests edge cases, missing files, duplicates, conflicts, invalid targets, and identifier exclusions.
Uses small artificial records only within tests.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

import config
from src.utils import find_raw_dataset
from src.preprocessing import (
    preprocess_dataset,
    validate_raw_dataframe,
    check_conflicting_labels,
    RAW_TO_CANONICAL_COLUMNS,
)


@pytest.fixture
def sample_synthetic_raw_df() -> pd.DataFrame:
    """Creates a small artificial raw dataframe for testing preprocessing logic."""
    data = {
        "Unnamed: 0": [0, 1, 2, 3],
        "Index": [1, 2, 3, 4],
        "Address": [
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
            "0x3333333333333333333333333333333333333333",
            "0x1111111111111111111111111111111111111111",  # Duplicate address with same label
        ],
        "FLAG": [0, 1, 0, 0],
        "Avg min between sent tnx": [10.5, 0.0, 55.2, 10.5],
        "Avg min between received tnx": [5.2, 0.0, 12.1, 5.2],
        "Time Diff between first and last (Mins)": [100.0, 0.0, 500.0, 100.0],
        "Sent tnx": [5, 0, 10, 5],
        "Received Tnx": [10, 1, 20, 10],
        "Number of Created Contracts": [0, 0, 1, 0],
        "Unique Received From Addresses": [3, 1, 5, 3],
        "Unique Sent To Addresses": [4, 0, 8, 4],
        "min value received": [0.1, 1.0, 0.05, 0.1],
        "max value received ": [5.0, 1.0, 10.0, 5.0],
        "avg val received": [1.2, 1.0, 2.5, 1.2],
        "min val sent": [0.1, 0.0, 0.2, 0.1],
        "max val sent": [2.0, 0.0, 5.0, 2.0],
        "avg val sent": [0.8, 0.0, 1.5, 0.8],
        "min value sent to contract": [0.0, 0.0, 0.0, 0.0],
        "max val sent to contract": [0.0, 0.0, 0.0, 0.0],
        "avg value sent to contract": [0.0, 0.0, 0.0, 0.0],
        "total transactions (including tnx to create contract": [15, 1, 31, 15],
        "total Ether sent": [4.0, 0.0, 15.0, 4.0],
        "total ether received": [12.0, 1.0, 50.0, 12.0],
        "total ether sent contracts": [0.0, 0.0, 0.0, 0.0],
        "total ether balance": [8.0, 1.0, 35.0, 8.0],
        " Total ERC20 tnxs": [2.0, np.nan, 5.0, 2.0],
        " ERC20 total Ether received": [100.0, np.nan, 250.0, 100.0],
        " ERC20 total ether sent": [50.0, np.nan, 100.0, 50.0],
        " ERC20 total Ether sent contract": [0.0, np.nan, 0.0, 0.0],
        " ERC20 uniq sent addr": [1.0, np.nan, 3.0, 1.0],
        " ERC20 uniq rec addr": [2.0, np.nan, 4.0, 2.0],
        " ERC20 uniq sent addr.1": [0.0, np.nan, 0.0, 0.0],
        " ERC20 uniq rec contract addr": [2.0, np.nan, 4.0, 2.0],
        " ERC20 avg time between sent tnx": [0.0, np.nan, 0.0, 0.0],
        " ERC20 avg time between rec tnx": [0.0, np.nan, 0.0, 0.0],
        " ERC20 avg time between rec 2 tnx": [0.0, np.nan, 0.0, 0.0],
        " ERC20 avg time between contract tnx": [0.0, np.nan, 0.0, 0.0],
        " ERC20 min val rec": [10.0, np.nan, 5.0, 10.0],
        " ERC20 max val rec": [90.0, np.nan, 200.0, 90.0],
        " ERC20 avg val rec": [50.0, np.nan, 50.0, 50.0],
        " ERC20 min val sent": [50.0, np.nan, 100.0, 50.0],
        " ERC20 max val sent": [50.0, np.nan, 100.0, 50.0],
        " ERC20 avg val sent": [50.0, np.nan, 100.0, 50.0],
        " ERC20 min val sent contract": [0.0, np.nan, 0.0, 0.0],
        " ERC20 max val sent contract": [0.0, np.nan, 0.0, 0.0],
        " ERC20 avg val sent contract": [0.0, np.nan, 0.0, 0.0],
        " ERC20 uniq sent token name": [1.0, np.nan, 2.0, 1.0],
        " ERC20 uniq rec token name": [2.0, np.nan, 3.0, 2.0],
        " ERC20 most sent token type": ["USDT", np.nan, "DAI", "USDT"],
        " ERC20_most_rec_token_type": ["USDT", np.nan, "LINK", "USDT"],
    }
    return pd.DataFrame(data)


def test_missing_csv_file(tmp_path: Path):
    """Verifies that attempting to load a non-existent CSV raises a helpful FileNotFoundError."""
    non_existent = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError) as exc_info:
        find_raw_dataset(non_existent)
    assert "DATASET MISSING ERROR" in str(exc_info.value)
    assert "https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset" in str(exc_info.value)


def test_empty_and_malformed_input_data():
    """Verifies that empty dataframes and data missing target column raise ValueError."""
    # Test completely empty dataframe
    with pytest.raises(ValueError, match="completely empty"):
        validate_raw_dataframe(pd.DataFrame())

    # Test missing target column
    bad_df = pd.DataFrame({"Address": ["0x123"], "some_col": [1]})
    with pytest.raises(ValueError, match="Target column 'FLAG' not found"):
        validate_raw_dataframe(bad_df, target_col="FLAG")


def test_unexpected_target_labels():
    """Verifies that non-binary target labels trigger an informative ValueError."""
    invalid_target_df = pd.DataFrame({
        "Address": ["0x1", "0x2", "0x3"],
        "FLAG": [0, 1, 2],  # '2' is invalid
    })
    with pytest.raises(ValueError, match="Unexpected target labels detected in 'FLAG'"):
        validate_raw_dataframe(invalid_target_df, target_col="FLAG")


def test_duplicate_records_and_conflicts():
    """Verifies conflict detection when duplicate addresses have conflicting labels."""
    conflicting_df = pd.DataFrame({
        "Address": [
            "0x1111111111111111111111111111111111111111",
            "0x1111111111111111111111111111111111111111",  # Same address, conflicting labels
        ],
        "FLAG": [0, 1],
    })
    with pytest.raises(ValueError, match="(?i)conflicting target labels"):
        check_conflicting_labels(conflicting_df)


def test_preprocessing_pipeline_execution(tmp_path: Path, sample_synthetic_raw_df: pd.DataFrame):
    """
    Tests full execution of preprocessing pipeline:
    - deduplication of identical addresses
    - exclusion of identifiers from feature matrix X
    - imputation of ERC20 nulls to 0.0
    - preservation of target vector y
    - exclusion of zero-variance columns
    """
    raw_csv = tmp_path / "synthetic_raw.csv"
    sample_synthetic_raw_df.to_csv(raw_csv, index=False)

    out_csv = tmp_path / "cleaned.csv"
    meta_json = tmp_path / "metadata.json"

    cleaned_df, X, y, meta = preprocess_dataset(
        input_path=raw_csv,
        output_path=out_csv,
        metadata_path=meta_json,
    )

    # 1. Deduplication check: 4 raw rows -> 3 unique addresses
    assert len(cleaned_df) == 3
    assert len(X) == 3
    assert len(y) == 3
    assert meta["deduplication"]["removed_count"] == 1

    # 2. Identifier exclusion check: Unnamed: 0, Index, Address must NOT be in X
    assert "raw_row_id" not in X.columns
    assert "dataset_index" not in X.columns
    assert "address" not in X.columns
    assert "Address" not in X.columns
    assert "Index" not in X.columns
    assert "flag" not in X.columns

    # 3. Target vector preservation
    assert list(y) == [0, 1, 0]

    # 4. Missing value imputation check: row 1 (which had NaNs) must now be 0.0 for numericals
    assert X["erc20_total_tnx"].isnull().sum() == 0
    assert X.loc[1, "erc20_total_tnx"] == 0.0
    assert X.loc[1, "erc20_total_received_amount"] == 0.0

    # Categorical token types must be 'None' instead of NaN
    assert cleaned_df.loc[1, "erc20_most_sent_token_type"] == "None"

    # 5. Constant columns dropped from X
    constant_dropped = [
        "erc20_avg_time_between_sent_tnx",
        "erc20_avg_time_between_rec_tnx",
        "erc20_min_val_sent_contract",
    ]
    for c in constant_dropped:
        assert c not in X.columns


def test_non_numeric_values_in_numerical_columns():
    """Verifies that non-numeric values in numerical columns can be detected."""
    bad_num_df = pd.DataFrame({
        "Address": ["0x1", "0x2"],
        "FLAG": [0, 1],
        "Sent tnx": ["5", "invalid_string"],
    })
    # Attempting to convert or coerce
    coerced = pd.to_numeric(bad_num_df["Sent tnx"], errors="coerce")
    assert coerced.isnull().sum() == 1
    assert coerced.iloc[0] == 5.0
