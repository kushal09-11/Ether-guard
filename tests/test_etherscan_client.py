"""
Unit tests for Etherscan API V2 Client (ChainSentinel - Phase 3).
Validates credential isolation, mocked network responses, rate limit handling,
pagination, address validation, and genuine empty wallet detection.
Requires NO external network access and NO real API key.
"""

from unittest.mock import Mock, patch
import pytest
import requests

from src.etherscan_client import (
    EtherscanClient,
    is_valid_ethereum_address,
    EtherscanConfigError,
    InvalidAddressError,
    EtherscanRateLimitError,
    EtherscanQuotaExceededError,
    EtherscanTimeoutError,
    EtherscanNetworkError,
    EtherscanAPIError,
    RateLimiter,
    clear_wallet_history_cache,
)

VALID_ADDRESS = "0x000000000000000000000000000000000000dEaD"
INVALID_ADDRESS = "0xInvalidEthAddress123"


def test_client_init_without_key_does_not_raise(monkeypatch):
    """Verify that instantiating EtherscanClient without an API key does NOT raise during initialization."""
    monkeypatch.delenv("ETHERSCAN_API_KEY", raising=False)
    client = EtherscanClient(api_key=None)
    assert not client.has_api_key
    assert "api_key_configured=False" in repr(client)


def test_client_repr_never_leaks_api_key():
    """Verify that EtherscanClient __repr__ hides the raw API key."""
    client = EtherscanClient(api_key="super_secret_key_12345")
    repr_str = repr(client)
    assert "super_secret_key_12345" not in repr_str
    assert "api_key_configured=True" in repr_str


def test_missing_api_key_raises_config_error_on_request(monkeypatch):
    """Verify that missing API key raises EtherscanConfigError only when a request is attempted."""
    monkeypatch.delenv("ETHERSCAN_API_KEY", raising=False)
    client = EtherscanClient(api_key=None)
    with pytest.raises(EtherscanConfigError, match="ETHERSCAN_API_KEY environment variable"):
        client.get_normal_transactions(VALID_ADDRESS)


def test_address_validation():
    """Verify Ethereum address format validation regex."""
    assert is_valid_ethereum_address("0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae")
    assert is_valid_ethereum_address("0x0000000000000000000000000000000000000000")
    assert not is_valid_ethereum_address("not_an_address")
    assert not is_valid_ethereum_address("0x123")  # too short
    assert not is_valid_ethereum_address("0x" + "g" * 40)  # non-hex
    assert not is_valid_ethereum_address(None)


def test_invalid_address_raises_error():
    """Verify that attempting to fetch transactions for an invalid address raises InvalidAddressError."""
    client = EtherscanClient(api_key="mock_key")
    with pytest.raises(InvalidAddressError, match="Invalid Ethereum address format"):
        client.get_normal_transactions("0xinvalid")


def test_get_normal_transactions_success():
    """Verify successful retrieval and parsing of normal transactions via mocked response."""
    mock_session = Mock(spec=requests.Session)
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "1",
        "message": "OK",
        "result": [
            {"hash": "0xabc1", "from": VALID_ADDRESS, "to": "0xrecip", "value": "1000000000000000000", "timeStamp": "1600000000", "isError": "0"},
            {"hash": "0xabc2", "from": "0xsender", "to": VALID_ADDRESS, "value": "2000000000000000000", "timeStamp": "1600000500", "isError": "0"},
        ],
    }
    mock_session.get.return_value = mock_response

    client = EtherscanClient(api_key="mock_key", session=mock_session)
    txs = client.get_normal_transactions(VALID_ADDRESS, page=1, offset=10)

    assert len(txs) == 2
    assert txs[0]["hash"] == "0xabc1"
    assert txs[1]["hash"] == "0xabc2"


def test_genuine_empty_wallet_returns_empty_list():
    """
    Verify that Etherscan's genuine empty wallet response
    (status='0', message='No transactions found') is parsed as an empty list [].
    """
    mock_session = Mock(spec=requests.Session)
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "0",
        "message": "No transactions found",
        "result": [],
    }
    mock_session.get.return_value = mock_response

    client = EtherscanClient(api_key="mock_key", session=mock_session)
    txs = client.get_normal_transactions(VALID_ADDRESS)
    assert txs == []


def test_api_rate_limit_error_handling():
    """Verify that Etherscan rate limit error (status='0', Max rate limit reached) raises EtherscanRateLimitError after retries."""
    mock_session = Mock(spec=requests.Session)
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "0",
        "message": "NOTOK",
        "result": "Max rate limit reached",
    }
    mock_session.get.return_value = mock_response

    client = EtherscanClient(api_key="mock_key", session=mock_session, max_retries=2, base_backoff=0.001)
    with pytest.raises(EtherscanRateLimitError, match="rate limit reached"):
        client.get_normal_transactions(VALID_ADDRESS)


def test_http_429_rate_limit_handling():
    """Verify that HTTP 429 status code raises EtherscanRateLimitError after retries."""
    mock_session = Mock(spec=requests.Session)
    mock_response = Mock()
    mock_response.status_code = 429
    mock_response.headers = {}
    mock_session.get.return_value = mock_response

    client = EtherscanClient(api_key="mock_key", session=mock_session, max_retries=2, base_backoff=0.001)
    with pytest.raises(EtherscanRateLimitError, match="rate limit exceeded"):
        client.get_normal_transactions(VALID_ADDRESS)


def test_invalid_api_key_error_handling():
    """Verify that invalid API key response raises EtherscanAPIError."""
    mock_session = Mock(spec=requests.Session)
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "0",
        "message": "NOTOK",
        "result": "Invalid API Key",
    }
    mock_session.get.return_value = mock_response

    client = EtherscanClient(api_key="bad_key", session=mock_session)
    with pytest.raises(EtherscanAPIError, match="API key rejected"):
        client.get_normal_transactions(VALID_ADDRESS)


def test_network_timeout_handling():
    """Verify that requests.exceptions.Timeout raises EtherscanTimeoutError."""
    mock_session = Mock(spec=requests.Session)
    mock_session.get.side_effect = requests.exceptions.Timeout("Connection timed out")

    client = EtherscanClient(api_key="mock_key", session=mock_session)
    with pytest.raises(EtherscanTimeoutError, match="timed out"):
        client.get_normal_transactions(VALID_ADDRESS)


def test_pagination_and_deduplication():
    """
    Verify multi-page transaction retrieval and hash deduplication:
    Page 1: 3 txs (offset=3)
    Page 2: 2 txs (with 1 overlapping hash)
    Total unique: 4 txs, is_complete=True
    """
    mock_session = Mock(spec=requests.Session)
    
    resp_page1 = Mock()
    resp_page1.status_code = 200
    resp_page1.json.return_value = {
        "status": "1",
        "message": "OK",
        "result": [
            {"hash": "0x1", "timeStamp": "100"},
            {"hash": "0x2", "timeStamp": "200"},
            {"hash": "0x3", "timeStamp": "300"},
        ],
    }

    resp_page2 = Mock()
    resp_page2.status_code = 200
    resp_page2.json.return_value = {
        "status": "1",
        "message": "OK",
        "result": [
            {"hash": "0x3", "timeStamp": "300"},  # duplicate
            {"hash": "0x4", "timeStamp": "400"},
        ],
    }

    mock_session.get.side_effect = [resp_page1, resp_page2]

    client = EtherscanClient(api_key="mock_key", session=mock_session)
    all_txs, is_complete = client.get_all_normal_transactions(VALID_ADDRESS, max_pages=5, offset=3)

    assert len(all_txs) == 4
    assert [tx["hash"] for tx in all_txs] == ["0x1", "0x2", "0x3", "0x4"]
    assert is_complete is True


def test_fetch_wallet_history_structure():
    """Verify fetch_wallet_history aggregates both normal transactions and ERC-20 transfers."""
    mock_session = Mock(spec=requests.Session)
    
    # Normal tx response
    resp_norm = Mock()
    resp_norm.status_code = 200
    resp_norm.json.return_value = {
        "status": "1",
        "message": "OK",
        "result": [{"hash": "0xnorm1", "timeStamp": "100"}],
    }
    
    # Token tx response
    resp_token = Mock()
    resp_token.status_code = 200
    resp_token.json.return_value = {
        "status": "1",
        "message": "OK",
        "result": [{"hash": "0xtoken1", "contractAddress": "0xUSDC", "tokenSymbol": "USDC"}],
    }

    mock_session.get.side_effect = [resp_norm, resp_token]

    client = EtherscanClient(api_key="mock_key", session=mock_session)
    history = client.fetch_wallet_history(VALID_ADDRESS, max_pages=1)

    assert history["address"] == VALID_ADDRESS.lower()
    assert history["total_normal_fetched"] == 1
    assert history["total_erc20_fetched"] == 1
    assert history["is_history_complete"] is True


def test_sanitize_log_message_redacts_api_key():
    """Verify that sanitize_log_message redacts apikey parameter in all formats."""
    from src.etherscan_client import sanitize_log_message

    raw_url = "https://api.etherscan.io/v2/api?chainid=1&apikey=SECRET_KEY_12345&module=account"
    cleaned = sanitize_log_message(raw_url)
    assert "SECRET_KEY_12345" not in cleaned
    assert "apikey=[REDACTED]" in cleaned


def test_network_error_scrubs_api_key_from_exception():
    """Verify that requests network error containing URL with API key has the key scrubbed."""
    mock_session = Mock(spec=requests.Session)
    mock_session.get.side_effect = requests.exceptions.ConnectionError(
        "Failed to connect to https://api.etherscan.io/v2/api?chainid=1&apikey=SUPER_SECRET_KEY_999"
    )

    client = EtherscanClient(api_key="SUPER_SECRET_KEY_999", session=mock_session)
    with pytest.raises(EtherscanNetworkError) as exc_info:
        client.get_normal_transactions(VALID_ADDRESS)

    err_msg = str(exc_info.value)
    assert "SUPER_SECRET_KEY_999" not in err_msg
    assert "[REDACTED]" in err_msg


def test_safe_logging_never_exposes_key(caplog):
    """Verify that logging during request execution never leaks the raw API key."""
    import logging
    mock_session = Mock(spec=requests.Session)
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "1", "message": "OK", "result": []}
    mock_session.get.return_value = mock_response

    secret_key = "MY_CONFIDENTIAL_KEY_XYZ789"
    client = EtherscanClient(api_key=secret_key, session=mock_session)

    with caplog.at_level(logging.DEBUG):
        client.get_normal_transactions(VALID_ADDRESS)

    assert secret_key not in caplog.text


def test_rate_limiter_pacing():
    """Verify that RateLimiter enforces minimum spacing between calls."""
    import time
    # Test with 20 RPS (interval = 0.05s) to keep test fast yet measurable
    limiter = RateLimiter(max_rps=20.0)
    t0 = time.monotonic()
    limiter.wait()
    limiter.wait()
    limiter.wait()
    elapsed = time.monotonic() - t0
    # 3 calls at 20 RPS must take at least 2 * 0.05 = 0.10s
    assert elapsed >= 0.08, f"Expected elapsed >= 0.08s, got {elapsed:.3f}s"


def test_transient_payload_rate_limit_retry_succeeds():
    """Verify that a transient rate limit response is retried and succeeds on attempt 2."""
    mock_session = Mock(spec=requests.Session)
    rate_limit_response = Mock()
    rate_limit_response.status_code = 200
    rate_limit_response.json.return_value = {
        "status": "0",
        "message": "NOTOK",
        "result": "Max rate limit reached",
    }

    success_response = Mock()
    success_response.status_code = 200
    success_response.json.return_value = {
        "status": "1",
        "message": "OK",
        "result": [{"hash": "0xsuccess1", "value": "1000"}],
    }

    mock_session.get.side_effect = [rate_limit_response, success_response]

    client = EtherscanClient(
        api_key="mock_key",
        session=mock_session,
        max_retries=2,
        base_backoff=0.001,
    )
    txs = client.get_normal_transactions(VALID_ADDRESS)

    assert len(txs) == 1
    assert txs[0]["hash"] == "0xsuccess1"
    assert mock_session.get.call_count == 2


def test_transient_http_429_retry_succeeds():
    """Verify that a transient HTTP 429 response is retried and succeeds on attempt 2."""
    mock_session = Mock(spec=requests.Session)
    rate_limit_response = Mock()
    rate_limit_response.status_code = 429
    rate_limit_response.headers = {"Retry-After": "0.01"}

    success_response = Mock()
    success_response.status_code = 200
    success_response.json.return_value = {
        "status": "1",
        "message": "OK",
        "result": [{"hash": "0x429recovered", "value": "5000"}],
    }

    mock_session.get.side_effect = [rate_limit_response, success_response]

    client = EtherscanClient(
        api_key="mock_key",
        session=mock_session,
        max_retries=2,
        base_backoff=0.001,
    )
    txs = client.get_normal_transactions(VALID_ADDRESS)

    assert len(txs) == 1
    assert txs[0]["hash"] == "0x429recovered"
    assert mock_session.get.call_count == 2


def test_http_429_retry_after_header_parsed():
    """Verify that HTTP 429 respects the Retry-After header."""
    mock_session = Mock(spec=requests.Session)
    rate_limit_response = Mock()
    rate_limit_response.status_code = 429
    rate_limit_response.headers = {"Retry-After": "0.05"}

    success_response = Mock()
    success_response.status_code = 200
    success_response.json.return_value = {
        "status": "1",
        "message": "OK",
        "result": [],
    }

    mock_session.get.side_effect = [rate_limit_response, success_response]

    with patch("time.sleep") as mock_sleep:
        client = EtherscanClient(
            api_key="mock_key",
            session=mock_session,
            max_retries=2,
            base_backoff=1.0,
        )
        client.get_normal_transactions(VALID_ADDRESS)

        # Confirm time.sleep was called with Retry-After value
        assert any(abs(call.args[0] - 0.05) < 0.001 for call in mock_sleep.call_args_list)


def test_daily_quota_exceeded_error_handling():
    """Verify that Etherscan daily quota exhaustion raises EtherscanQuotaExceededError immediately without retrying."""
    mock_session = Mock(spec=requests.Session)
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "0",
        "message": "NOTOK",
        "result": "Endpoint daily limit reached: 100,000 calls per day",
    }
    mock_session.get.return_value = mock_response

    client = EtherscanClient(api_key="mock_key", session=mock_session, max_retries=3, base_backoff=0.001)
    with pytest.raises(EtherscanQuotaExceededError, match="daily quota exhausted"):
        client.get_normal_transactions(VALID_ADDRESS)

    # Must fail immediately on attempt 1 without wasting retries
    assert mock_session.get.call_count == 1


def test_wallet_history_cache_and_clear():
    """Verify that fetch_wallet_history caches results and clear_wallet_history_cache clears them."""
    clear_wallet_history_cache()

    mock_session = Mock(spec=requests.Session)
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "1",
        "message": "OK",
        "result": [{"hash": "0xcachedtx1"}],
    }
    mock_session.get.return_value = mock_response

    client = EtherscanClient(api_key="mock_key", session=mock_session, cache_ttl=60.0)

    # First call: makes network calls (one for normal tx, one for erc20)
    history1 = client.fetch_wallet_history(VALID_ADDRESS, max_pages=1)
    first_call_count = mock_session.get.call_count
    assert first_call_count >= 2

    # Second call within TTL: served from in-memory cache, no additional network calls
    history2 = client.fetch_wallet_history(VALID_ADDRESS, max_pages=1)
    assert mock_session.get.call_count == first_call_count
    assert history1["total_normal_fetched"] == history2["total_normal_fetched"]

    # Clear cache and verify subsequent call performs network requests again
    clear_wallet_history_cache()
    client.fetch_wallet_history(VALID_ADDRESS, max_pages=1)
    assert mock_session.get.call_count > first_call_count


