"""
Integration tests for ChainSentinel Phase 4 Frontend and FastAPI Application.
Validates all API endpoints, address validation, error mappings, demo presets,
cold-start warnings, zero-activity rejection, and static frontend serving.
Runs 100% offline without requiring an Etherscan API key or internet connection.
"""

from unittest.mock import patch, Mock
import pytest
from fastapi.testclient import TestClient

import config
from src.app import app
from src.etherscan_client import (
    EtherscanRateLimitError,
    EtherscanQuotaExceededError,
    EtherscanTimeoutError,
    EtherscanNetworkError,
    EtherscanAPIError,
)

TARGET_WALLET = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"


@pytest.fixture
def client():
    """FastAPI test client fixture."""
    return TestClient(app)


# -------------------------------------------------------------
# 1. Health & Introspection Tests
# -------------------------------------------------------------

def test_api_health(client):
    """Verify /api/health returns healthy status and model configuration."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ChainSentinel"
    assert data["model_loaded"] is True
    assert data["decision_threshold"] == 0.51
    assert data["canonical_features_count"] == 24


def test_api_model_info(client):
    """Verify /api/model-info provides complete architecture and held-out test metrics."""
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "HistGradientBoostingClassifier"

    metrics = data["held_out_test_metrics"]
    cm = metrics["confusion_matrix"]
    assert cm["true_negatives"] == 1131
    assert cm["false_positives"] == 15
    assert cm["false_negatives"] == 41
    assert cm["true_positives"] == 286
    assert cm["total"] == 1473

    assert metrics["precision"] == 0.9502
    assert metrics["recall"] == 0.8746
    assert metrics["fraud_f1"] == 0.9108
    assert metrics["brier_score"] == 0.0283
    assert len(data["canonical_features"]) == 24


def test_api_presets_list(client):
    """Verify /api/presets lists all required demonstration scenarios."""
    response = client.get("/api/presets")
    assert response.status_code == 200
    presets = response.json()["presets"]
    preset_ids = [p["id"] for p in presets]

    assert "normal" in preset_ids
    assert "flagged" in preset_ids
    assert "cold_start" in preset_ids
    assert "zero_activity" in preset_ids
    assert "live_vitalik" in preset_ids


# -------------------------------------------------------------
# 2. Demonstration Presets Execution Tests
# -------------------------------------------------------------

def test_assess_preset_normal(client):
    """Verify mature normal preset evaluates to NORMAL with low score and no cold-start warning."""
    response = client.post("/api/assess-preset/normal")
    assert response.status_code == 200
    data = response.json()

    assert data["is_scoreable"] is True
    assert data["decision"] == "NORMAL"
    assert data["analysis_mode"] == "OFFLINE_PRESET"
    assert 0.0 <= data["risk_score"] < 0.51
    assert data["has_cold_start"] is False
    assert data["risk_level"] == "LOW"
    assert "features" in data
    assert len(data["features"]) == 24


def test_assess_preset_flagged(client):
    """Verify flagged preset evaluates to FLAGGED with risk score >= 0.51."""
    response = client.post("/api/assess-preset/flagged")
    assert response.status_code == 200
    data = response.json()

    assert data["is_scoreable"] is True
    assert data["decision"] == "FLAGGED"
    assert data["risk_score"] >= 0.51
    assert data["risk_level"] in ("MEDIUM", "HIGH")


def test_assess_preset_cold_start(client):
    """Verify cold-start preset triggers prominent warning due to lifespan < 24h."""
    response = client.post("/api/assess-preset/cold_start")
    assert response.status_code == 200
    data = response.json()

    assert data["is_scoreable"] is True
    assert data["has_cold_start"] is True
    assert any("COLD-START WARNING" in w for w in data["warnings"])


def test_assess_preset_zero_activity(client):
    """Verify zero-activity preset rejects scoring and returns INSUFFICIENT_ACTIVITY with no risk score."""
    response = client.post("/api/assess-preset/zero_activity")
    assert response.status_code == 200
    data = response.json()

    assert data["is_scoreable"] is False
    assert data["decision"] == "INSUFFICIENT_ACTIVITY"
    assert data["risk_score"] is None
    assert any("zero recorded transactions" in w for w in data["warnings"])


def test_assess_preset_unknown(client):
    """Verify request for non-existent preset returns 404."""
    response = client.post("/api/assess-preset/unknown_preset_123")
    assert response.status_code == 404


# -------------------------------------------------------------
# 3. Live Assessment Validation & Error Mapping Tests
# -------------------------------------------------------------

def test_assess_wallet_invalid_address_format(client):
    """Verify invalid address formats return HTTP 400 Bad Request."""
    bad_addresses = [
        "not-an-address",
        "0x123",
        "0xZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ",
        "742d35Cc6634C0532925a3b844Bc454e4438f44e",  # missing 0x prefix
    ]
    for bad in bad_addresses:
        resp = client.post("/api/assess-wallet", json={"address": bad})
        assert resp.status_code == 400
        assert "Invalid Ethereum address format" in resp.json()["detail"]


def test_assess_wallet_missing_api_key_returns_503(client):
    """
    Verify when no ETHERSCAN_API_KEY is configured in the environment,
    live wallet query returns HTTP 503 Service Unavailable with setup instructions.
    """
    with patch.dict("os.environ", {"ETHERSCAN_API_KEY": ""}, clear=False):
        resp = client.post("/api/assess-wallet", json={"address": TARGET_WALLET})
        assert resp.status_code == 503
        detail = resp.json()["detail"]
        assert detail["error_type"] == "EtherscanConfigError"
        assert "ETHERSCAN_API_KEY" in detail["help"]


def test_assess_wallet_mocked_success(client):
    """Verify successful live wallet assessment when Etherscan returns data."""
    mock_history = {
        "address": TARGET_WALLET.lower(),
        "normal_transactions": [
            {"timeStamp": "1000", "from": "0xsender", "to": TARGET_WALLET, "value": "2000000000000000000", "isError": "0"},
            {"timeStamp": "3000", "from": TARGET_WALLET, "to": "0xrecip", "value": "1000000000000000000", "isError": "0"},
        ],
        "erc20_transfers": [],
        "is_history_complete": True,
        "total_normal_fetched": 2,
        "total_erc20_fetched": 0,
    }

    with patch.dict("os.environ", {"ETHERSCAN_API_KEY": "test_mock_key"}):
        with patch("src.etherscan_client.EtherscanClient.fetch_wallet_history", return_value=mock_history):
            resp = client.post("/api/assess-wallet", json={"address": TARGET_WALLET})
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_scoreable"] is True
            assert 0.0 <= data["risk_score"] <= 1.0
            assert data["address"] == TARGET_WALLET.lower()


def test_assess_wallet_rate_limit_never_produces_zero_activity(client):
    """
    Requirement 4 & 7:
    Ensure Etherscan rate limit (HTTP 429) returns 429 error and NEVER silently returns zero activity.
    """
    with patch.dict("os.environ", {"ETHERSCAN_API_KEY": "test_mock_key"}):
        with patch("src.etherscan_client.EtherscanClient.fetch_wallet_history", side_effect=EtherscanRateLimitError("Rate limit reached")):
            resp = client.post("/api/assess-wallet", json={"address": TARGET_WALLET})
            assert resp.status_code == 429
            data = resp.json()["detail"]
            assert data["error_type"] == "EtherscanRateLimitError"
            assert "rate limit" in data["message"].lower()


def test_assess_wallet_daily_quota_exceeded_returns_503(client):
    """Verify exhausted 24h daily quota returns 503 Service Unavailable with daily quota details."""
    with patch.dict("os.environ", {"ETHERSCAN_API_KEY": "test_mock_key"}):
        with patch("src.etherscan_client.EtherscanClient.fetch_wallet_history", side_effect=EtherscanQuotaExceededError("Daily limit reached: 100,000 calls per day")):
            resp = client.post("/api/assess-wallet", json={"address": TARGET_WALLET})
            assert resp.status_code == 503
            data = resp.json()["detail"]
            assert data["error_type"] == "EtherscanQuotaExceededError"
            assert "daily quota" in data["message"].lower()


def test_assess_wallet_timeout_returns_504(client):
    """Verify timeout exception returns 504 Gateway Timeout."""
    with patch.dict("os.environ", {"ETHERSCAN_API_KEY": "test_mock_key"}):
        with patch("src.etherscan_client.EtherscanClient.fetch_wallet_history", side_effect=EtherscanTimeoutError("Timed out after 15s")):
            resp = client.post("/api/assess-wallet", json={"address": TARGET_WALLET})
            assert resp.status_code == 504
            data = resp.json()["detail"]
            assert data["error_type"] == "EtherscanTimeoutError"


def test_assess_wallet_network_error_returns_502(client):
    """Verify connection / network failure returns 502 Bad Gateway."""
    with patch.dict("os.environ", {"ETHERSCAN_API_KEY": "test_mock_key"}):
        with patch("src.etherscan_client.EtherscanClient.fetch_wallet_history", side_effect=EtherscanNetworkError("Connection refused")):
            resp = client.post("/api/assess-wallet", json={"address": TARGET_WALLET})
            assert resp.status_code == 502
            data = resp.json()["detail"]
            assert data["error_type"] == "EtherscanNetworkError"


# -------------------------------------------------------------
# 4. Custom Feature Assessment Endpoint
# -------------------------------------------------------------

def test_assess_custom_features(client):
    """Verify direct feature evaluation endpoint /api/assess-features."""
    from src.presets import DEMO_PRESETS
    features = DEMO_PRESETS["normal"]["features"]

    resp = client.post("/api/assess-features", json={"features": features, "address": TARGET_WALLET})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_scoreable"] is True
    assert 0.0 <= data["risk_score"] <= 1.0


# -------------------------------------------------------------
# 5. Static Web Interface Serving Tests
# -------------------------------------------------------------

def test_serve_index_page(client):
    """Verify root route / serves index.html with disclaimer and input elements."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    text = resp.text

    # Persistent Academic Disclaimer
    assert "ACADEMIC CASE STUDY" in text
    assert "not calibrated fraud probabilities or proof of fraud" in text

    # Core UI Elements
    assert "ChainSentinel" in text
    assert "Analyze Wallet" in text
    assert "Demonstration Scenarios" in text
    assert "Insufficient On-Chain Activity" in text
    assert "Cold-Start Alert" in text
    assert "Canonical Feature Summary" in text


def test_serve_static_assets(client):
    """Verify static assets style.css and app.js are served properly."""
    css_resp = client.get("/static/style.css")
    assert css_resp.status_code == 200
    assert "text/css" in css_resp.headers["content-type"]

    js_resp = client.get("/static/app.js")
    assert js_resp.status_code == 200
    assert ("javascript" in js_resp.headers["content-type"] or "text/plain" in js_resp.headers["content-type"])


def test_assess_wallet_invalid_api_key_returns_502(client):
    """Verify that an invalid or rejected API key returns HTTP 502 with informative message."""
    with patch.dict("os.environ", {"ETHERSCAN_API_KEY": "invalid_rejected_key"}):
        with patch(
            "src.etherscan_client.EtherscanClient.fetch_wallet_history",
            side_effect=EtherscanAPIError("Etherscan API key rejected: Invalid API Key"),
        ):
            resp = client.post("/api/assess-wallet", json={"address": TARGET_WALLET})
            assert resp.status_code == 502
            data = resp.json()["detail"]
            assert data["error_type"] == "EtherscanAPIError"
            assert "rejected" in data["message"].lower()


def test_assess_wallet_invokes_pipeline_not_preset(client):
    """
    Requirement 4:
    Verify that live endpoint invokes Etherscan, extracts features, and calls model pipeline,
    rather than returning static demo presets.
    """
    mock_history = {
        "address": TARGET_WALLET.lower(),
        "normal_transactions": [
            {"timeStamp": "1000", "from": "0xsender", "to": TARGET_WALLET, "value": "5000000000000000000", "isError": "0"},
            {"timeStamp": "4000", "from": TARGET_WALLET, "to": "0xrecip", "value": "2000000000000000000", "isError": "0"},
        ],
        "erc20_transfers": [
            {"timeStamp": "2000", "from": "0xsender", "to": TARGET_WALLET, "value": "100", "contractAddress": "0xToken", "tokenSymbol": "TKN"}
        ],
        "is_history_complete": True,
        "total_normal_fetched": 2,
        "total_erc20_fetched": 1,
    }

    with patch.dict("os.environ", {"ETHERSCAN_API_KEY": "valid_mock_key"}):
        with patch("src.etherscan_client.EtherscanClient.fetch_wallet_history", return_value=mock_history) as mock_fetch:
            resp = client.post("/api/assess-wallet", json={"address": TARGET_WALLET})
            assert resp.status_code == 200
            data = resp.json()

            # Confirm fetch_wallet_history was actually invoked with the target address
            mock_fetch.assert_called_once_with(TARGET_WALLET, max_pages=config.DEFAULT_MAX_PAGES)

            # Confirm the extracted features match the mock history (not a preset)
            assert data["raw_normal_count"] == 2
            assert data["raw_erc20_count"] == 1
            assert data["features"]["total_transactions_count"] == 2.0
            assert data["features"]["erc20_total_tnx"] == 1.0
            assert "preset_id" not in data  # Proves it is not a demo preset
            assert data["analysis_mode"] == "LIVE_ON_CHAIN"


