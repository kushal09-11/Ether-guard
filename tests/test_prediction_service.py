"""
Unit tests for WalletPredictionService (ChainSentinel - Phase 3).
Validates Requirement 7: Thin prediction service loading best_model_pipeline.joblib,
risk score generation, cold-start warnings, zero-activity rejection, and disclaimers.
Runs entirely offline without requiring a live Etherscan API key.
"""

from unittest.mock import Mock
import pytest

import config
from src.prediction_service import WalletPredictionService
from src.etherscan_client import EtherscanClient
from src.feature_extractor import LiveFeatureExtractor

TARGET_WALLET = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"


@pytest.fixture
def prediction_service():
    """Initializes WalletPredictionService using existing saved best_model_pipeline.joblib."""
    return WalletPredictionService(model_path=config.BEST_MODEL_PATH)


@pytest.fixture
def mature_wallet_features():
    """Features for an active, long-lived wallet (over 1 year active, 30 txs)."""
    return {
        "avg_min_between_sent_tnx": 100.0,
        "avg_min_between_received_tnx": 80.0,
        "time_diff_first_last_mins": 525600.0,  # 1 year
        "sent_tnx": 10,
        "received_tnx": 20,
        "created_contracts_count": 0,
        "unique_received_from_addresses": 15,
        "unique_sent_to_addresses": 8,
        "min_value_received_eth": 0.05,
        "max_value_received_eth": 10.0,
        "avg_value_received_eth": 1.2,
        "min_value_sent_eth": 0.01,
        "max_value_sent_eth": 5.0,
        "avg_value_sent_eth": 0.8,
        "total_transactions_count": 30,
        "total_ether_sent": 8.0,
        "total_ether_received": 24.0,
        "total_ether_balance": 16.0,
        "erc20_total_tnx": 5,
        "erc20_unique_sent_addr": 2,
        "erc20_unique_received_addr": 3,
        "erc20_unique_received_contract_addr": 3,
        "erc20_unique_sent_token_name": 1,
        "erc20_unique_received_token_name": 2,
    }


@pytest.fixture
def cold_start_wallet_features():
    """Features for a brand-new wallet active for only 45 minutes."""
    return {
        "avg_min_between_sent_tnx": 10.0,
        "avg_min_between_received_tnx": 0.0,
        "time_diff_first_last_mins": 45.0,  # 45 minutes lifespan (< 24 hours)
        "sent_tnx": 2,
        "received_tnx": 1,
        "created_contracts_count": 0,
        "unique_received_from_addresses": 1,
        "unique_sent_to_addresses": 1,
        "min_value_received_eth": 1.0,
        "max_value_received_eth": 1.0,
        "avg_value_received_eth": 1.0,
        "min_value_sent_eth": 0.4,
        "max_value_sent_eth": 0.5,
        "avg_value_sent_eth": 0.45,
        "total_transactions_count": 3,
        "total_ether_sent": 0.9,
        "total_ether_received": 1.0,
        "total_ether_balance": 0.1,
        "erc20_total_tnx": 0,
        "erc20_unique_sent_addr": 0,
        "erc20_unique_received_addr": 0,
        "erc20_unique_received_contract_addr": 0,
        "erc20_unique_sent_token_name": 0,
        "erc20_unique_received_token_name": 0,
    }


def test_assess_mature_wallet_risk(prediction_service, mature_wallet_features):
    """Verify inference on mature wallet produces valid risk score and NO cold-start warning."""
    result = prediction_service.assess_from_features(mature_wallet_features, address=TARGET_WALLET)

    assert result["is_scoreable"] is True
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["decision"] in ("FLAGGED", "NORMAL")
    assert result["decision_threshold"] == 0.51
    assert result["has_cold_start"] is False
    assert len(result["disclaimers"]) > 0


def test_assess_cold_start_wallet_triggers_warning(prediction_service, cold_start_wallet_features):
    """Verify that a wallet with short lifespan (<24h) triggers a cold-start warning."""
    result = prediction_service.assess_from_features(cold_start_wallet_features, address=TARGET_WALLET)

    assert result["is_scoreable"] is True
    assert result["has_cold_start"] is True
    assert any("COLD-START WARNING" in w for w in result["warnings"])


def test_assess_zero_activity_wallet_is_unscoreable(prediction_service):
    """Verify that a wallet with zero activity is rejected from scoring."""
    zero_features = {col: 0.0 for col in config.CANONICAL_24_FEATURES}
    result = prediction_service.assess_from_features(zero_features, address=TARGET_WALLET)

    assert result["is_scoreable"] is False
    assert result["decision"] == "INSUFFICIENT_ACTIVITY"
    assert result["risk_score"] is None
    assert any("zero recorded transactions" in w for w in result["warnings"])


def test_end_to_end_assess_wallet_with_mocked_client(prediction_service):
    """
    Verify complete end-to-end assess_wallet execution using mocked EtherscanClient.
    Does not require real API key or internet access.
    """
    mock_client = Mock(spec=EtherscanClient)
    mock_client.fetch_wallet_history.return_value = {
        "address": TARGET_WALLET.lower(),
        "normal_transactions": [
            {"timeStamp": "1000", "from": "0xsender1", "to": TARGET_WALLET, "value": "2000000000000000000", "isError": "0"},
            {"timeStamp": "2000", "from": TARGET_WALLET, "to": "0xrecip1", "value": "1000000000000000000", "isError": "0"},
            {"timeStamp": "90000", "from": TARGET_WALLET, "to": "0xrecip2", "value": "500000000000000000", "isError": "0"},
        ],
        "erc20_transfers": [],
        "is_history_complete": True,
        "total_normal_fetched": 3,
        "total_erc20_fetched": 0,
    }

    service = WalletPredictionService(
        model_path=config.BEST_MODEL_PATH,
        etherscan_client=mock_client,
    )

    assessment = service.assess_wallet(TARGET_WALLET)

    assert assessment["is_scoreable"] is True
    assert 0.0 <= assessment["risk_score"] <= 1.0
    assert assessment["decision"] in ("FLAGGED", "NORMAL")
    assert assessment["address"] == TARGET_WALLET.lower()
    assert "features" in assessment
    assert len(assessment["features"]) == 24


def test_api_errors_never_silently_produce_risk_score(prediction_service):
    """
    Requirement 4 (Phase 3.1):
    Ensure an API error (rate limit, API error, timeout) can NEVER silently produce a valid-looking risk score.
    """
    from src.etherscan_client import EtherscanRateLimitError, EtherscanAPIError, EtherscanTimeoutError

    # 1. Rate Limit Error
    mock_client_rl = Mock(spec=EtherscanClient)
    mock_client_rl.fetch_wallet_history.side_effect = EtherscanRateLimitError("Rate limit exceeded")
    service_rl = WalletPredictionService(model_path=config.BEST_MODEL_PATH, etherscan_client=mock_client_rl)

    with pytest.raises(EtherscanRateLimitError, match="Rate limit exceeded"):
        service_rl.assess_wallet(TARGET_WALLET)

    # 2. General API Error (NOTOK, Invalid Key)
    mock_client_api = Mock(spec=EtherscanClient)
    mock_client_api.fetch_wallet_history.side_effect = EtherscanAPIError("Invalid API Key")
    service_api = WalletPredictionService(model_path=config.BEST_MODEL_PATH, etherscan_client=mock_client_api)

    with pytest.raises(EtherscanAPIError, match="Invalid API Key"):
        service_api.assess_wallet(TARGET_WALLET)

    # 3. Timeout Error
    mock_client_to = Mock(spec=EtherscanClient)
    mock_client_to.fetch_wallet_history.side_effect = EtherscanTimeoutError("Request timed out")
    service_to = WalletPredictionService(model_path=config.BEST_MODEL_PATH, etherscan_client=mock_client_to)

    with pytest.raises(EtherscanTimeoutError, match="Request timed out"):
        service_to.assess_wallet(TARGET_WALLET)


def test_incomplete_history_triggers_prominent_warning(prediction_service):
    """Verify that truncated/incomplete transaction history surfaces an explicit warning."""
    mock_client = Mock(spec=EtherscanClient)
    mock_client.fetch_wallet_history.return_value = {
        "address": TARGET_WALLET.lower(),
        "normal_transactions": [
            {"timeStamp": str(i * 1000), "from": TARGET_WALLET, "to": "0xrecip", "value": "100000000000000000", "isError": "0"}
            for i in range(1, 10)
        ],
        "erc20_transfers": [],
        "is_history_complete": False,  # Truncated by pagination
        "total_normal_fetched": 9,
        "total_erc20_fetched": 0,
    }

    service = WalletPredictionService(model_path=config.BEST_MODEL_PATH, etherscan_client=mock_client)
    res = service.assess_wallet(TARGET_WALLET)

    assert res["is_scoreable"] is True
    assert any("truncated by pagination limits" in w for w in res["warnings"])


def test_feature_schema_and_order_exact_match_model(prediction_service):
    """
    Requirement 5 (Phase 3.1):
    Verify that the extracted feature names and order exactly match the saved model's expected input schema.
    """
    classifier = prediction_service.pipeline.named_steps["classifier"]
    assert hasattr(classifier, "feature_names_in_")
    model_expected_features = list(classifier.feature_names_in_)

    assert model_expected_features == config.CANONICAL_24_FEATURES, (
        "Model expected features do not match config.CANONICAL_24_FEATURES"
    )

    # Also verify that LiveFeatureExtractor produces keys matching this exact list
    extractor = LiveFeatureExtractor()
    extracted_res = extractor.extract_features(TARGET_WALLET, [], [])
    assert list(extracted_res.features.keys()) == config.CANONICAL_24_FEATURES

