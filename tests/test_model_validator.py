"""
Unit tests for Model Input Validation Module (ChainSentinel - Phase 3).
Validates Requirement 6: Rejection of missing, extra, non-numeric, NaN, or non-finite features.
Prevents unscoreable data fabrication.
"""

import pytest
import numpy as np
import pandas as pd

import config
from src.model_validator import validate_model_input, ModelInputValidationError


@pytest.fixture
def valid_features_dict():
    """Returns a valid, complete 24-feature canonical dictionary."""
    return {
        "avg_min_between_sent_tnx": 10.5,
        "avg_min_between_received_tnx": 20.0,
        "time_diff_first_last_mins": 50000.0,
        "sent_tnx": 5,
        "received_tnx": 10,
        "created_contracts_count": 0,
        "unique_received_from_addresses": 8,
        "unique_sent_to_addresses": 4,
        "min_value_received_eth": 0.1,
        "max_value_received_eth": 5.0,
        "avg_value_received_eth": 1.5,
        "min_value_sent_eth": 0.05,
        "max_value_sent_eth": 2.0,
        "avg_value_sent_eth": 0.8,
        "total_transactions_count": 15,
        "total_ether_sent": 4.0,
        "total_ether_received": 15.0,
        "total_ether_balance": 11.0,
        "erc20_total_tnx": 4,
        "erc20_unique_sent_addr": 1,
        "erc20_unique_received_addr": 3,
        "erc20_unique_received_contract_addr": 2,
        "erc20_unique_sent_token_name": 1,
        "erc20_unique_received_token_name": 2,
    }


def test_valid_input_passes_validation(valid_features_dict):
    """Verify that a valid canonical dictionary and DataFrame pass validation."""
    df_from_dict = validate_model_input(valid_features_dict)
    assert isinstance(df_from_dict, pd.DataFrame)
    assert list(df_from_dict.columns) == config.CANONICAL_24_FEATURES
    assert df_from_dict.shape == (1, 24)

    # Test passing DataFrame directly
    df_from_df = validate_model_input(df_from_dict)
    assert df_from_df.shape == (1, 24)


def test_missing_features_rejected(valid_features_dict):
    """Verify that missing one or more required features raises ModelInputValidationError."""
    bad_dict = valid_features_dict.copy()
    del bad_dict["time_diff_first_last_mins"]

    with pytest.raises(ModelInputValidationError, match="Missing 1 required feature"):
        validate_model_input(bad_dict)


def test_extra_unexpected_features_rejected(valid_features_dict):
    """Verify that unexpected extra features raise ModelInputValidationError."""
    bad_dict = valid_features_dict.copy()
    bad_dict["fraud_indicator_flag"] = 1.0  # extra/leaked column

    with pytest.raises(ModelInputValidationError, match="Unexpected extra feature"):
        validate_model_input(bad_dict)


def test_non_numeric_values_rejected(valid_features_dict):
    """Verify that string or non-numeric values raise ModelInputValidationError."""
    bad_dict = valid_features_dict.copy()
    bad_dict["sent_tnx"] = "five_transactions"

    with pytest.raises(ModelInputValidationError, match="contains non-numeric values"):
        validate_model_input(bad_dict)


def test_nan_values_rejected(valid_features_dict):
    """Verify that NaN or null values raise ModelInputValidationError."""
    bad_dict = valid_features_dict.copy()
    bad_dict["avg_value_received_eth"] = np.nan

    with pytest.raises(ModelInputValidationError, match="contains missing"):
        validate_model_input(bad_dict)


def test_infinite_values_rejected(valid_features_dict):
    """Verify that positive or negative infinity raises ModelInputValidationError."""
    bad_dict_pos = valid_features_dict.copy()
    bad_dict_pos["total_ether_balance"] = np.inf

    with pytest.raises(ModelInputValidationError, match="contains non-finite"):
        validate_model_input(bad_dict_pos)

    bad_dict_neg = valid_features_dict.copy()
    bad_dict_neg["time_diff_first_last_mins"] = -np.inf

    with pytest.raises(ModelInputValidationError, match="contains non-finite"):
        validate_model_input(bad_dict_neg)


def test_empty_dataframe_rejected():
    """Verify that empty DataFrame raises ModelInputValidationError."""
    with pytest.raises(ModelInputValidationError, match="DataFrame is empty"):
        validate_model_input(pd.DataFrame())
