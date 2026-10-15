"""
Etherscan API V2 Client (ChainSentinel - Phase 3).
Provides configurable, robust retrieval of Ethereum normal transactions and ERC-20 token transfers.
Strictly isolates credentials, handles rate limits, pagination, network timeouts, and genuine empty wallets.
"""

import os
import re
from typing import Dict, Any, List, Tuple, Optional
import requests

import config

# Regex pattern for Ethereum address validation (0x followed by 40 hex characters)
ETH_ADDRESS_REGEX = re.compile(r"^0x[a-fA-F0-9]{40}$")


class EtherscanError(Exception):
    """Base exception for all Etherscan client errors."""
    pass


class EtherscanConfigError(EtherscanError):
    """Raised when the Etherscan API key is missing or improperly configured."""
    pass


class InvalidAddressError(EtherscanError):
    """Raised when an invalid Ethereum wallet address format is supplied."""
    pass


class EtherscanRateLimitError(EtherscanError):
    """Raised when the Etherscan API rate limit is reached."""
    pass


class EtherscanQuotaExceededError(EtherscanError):
    """Raised when the Etherscan daily API request quota (e.g. 100,000 calls/day) is exhausted."""
    pass


class EtherscanTimeoutError(EtherscanError):
    """Raised when an API request times out."""
    pass


class EtherscanNetworkError(EtherscanError):
    """Raised on connection failure, DNS resolution failure, or HTTP 5xx errors."""
    pass


class EtherscanAPIError(EtherscanError):
    """Raised when the Etherscan API returns an application-level error (status='0', NOTOK)."""
    pass


def is_valid_ethereum_address(address: str) -> bool:
    """Validates if a string is a well-formed Ethereum address."""
    if not isinstance(address, str):
        return False
    return bool(ETH_ADDRESS_REGEX.match(address.strip()))


import logging
import random
import threading
import time
from src.utils import mask_address

# Configure module-level logger
logger = logging.getLogger("chainsentinel.etherscan")


class RateLimiter:
    """
    Thread-safe interval rate limiter enforcing a maximum request rate.
    Maintains a minimum spacing between outgoing requests to respect Etherscan's free-tier rate limits.
    Default: <= 3 requests per second (conservative target below the 5 req/s limit).
    """

    def __init__(self, max_rps: float = config.DEFAULT_ETHERSCAN_RATE_LIMIT_RPS):
        self.max_rps = max_rps
        self.interval = 1.0 / max_rps if max_rps > 0 else 0.0
        self._lock = threading.Lock()
        self._last_call_time = 0.0

    def wait(self) -> None:
        """Blocks until the required interval has elapsed since the previous request."""
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_time
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self._last_call_time = time.monotonic()


# Shared singleton rate limiter across client instances
_GLOBAL_RATE_LIMITER = RateLimiter(config.DEFAULT_ETHERSCAN_RATE_LIMIT_RPS)

# Thread-safe in-memory cache for wallet histories
# Key: (address.lower(), max_pages) -> (timestamp, result_dict)
_HISTORY_CACHE: Dict[Tuple[str, int], Tuple[float, Dict[str, Any]]] = {}
_CACHE_LOCK = threading.Lock()


def clear_wallet_history_cache() -> None:
    """Clears the in-memory wallet history cache. Used for testing and cache invalidation."""
    with _CACHE_LOCK:
        _HISTORY_CACHE.clear()


def sanitize_log_message(msg: Any) -> str:
    """
    Safely redacts any Etherscan API key instances from error messages, URLs, or log strings.
    Guarantees credentials are never exposed in logs or exception tracebacks.
    """
    text = str(msg)
    return re.sub(r"apikey=[^&\s'\"]+", "apikey=[REDACTED]", text, flags=re.IGNORECASE)


class EtherscanClient:
    """
    Client for interacting with Etherscan API V2 endpoints.
    Reads credentials from environment variables or explicit parameters.
    Never exposes or logs the API key.
    Enforces shared rate limiting, bounded exponential retries with jitter, and response caching.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = config.ETHERSCAN_API_V2_BASE_URL,
        chain_id: str = config.DEFAULT_ETHERSCAN_CHAIN_ID,
        timeout: int = config.DEFAULT_REQUEST_TIMEOUT,
        session: Optional[requests.Session] = None,
        rate_limiter: Optional[RateLimiter] = None,
        max_retries: int = config.DEFAULT_ETHERSCAN_MAX_RETRIES,
        base_backoff: float = config.DEFAULT_ETHERSCAN_RETRY_BACKOFF,
        cache_ttl: float = config.DEFAULT_ETHERSCAN_CACHE_TTL_SECS,
    ):
        # Read API key from parameter or environment variable
        raw_key = api_key if api_key is not None else os.environ.get("ETHERSCAN_API_KEY", "")
        self._api_key = raw_key.strip() if isinstance(raw_key, str) else ""
        self.base_url = base_url
        self.chain_id = str(chain_id)
        self.timeout = timeout
        self._session = session or requests.Session()
        self.rate_limiter = rate_limiter or _GLOBAL_RATE_LIMITER
        self.max_retries = max(0, int(max_retries))
        self.base_backoff = max(0.0, float(base_backoff))
        self.cache_ttl = max(0.0, float(cache_ttl))

    @property
    def has_api_key(self) -> bool:
        """Returns True if a non-empty API key is present."""
        return bool(self._api_key)

    def __repr__(self) -> str:
        """Safe representation hiding the raw API key."""
        return (
            f"EtherscanClient(base_url='{self.base_url}', chain_id='{self.chain_id}', "
            f"api_key_configured={self.has_api_key})"
        )

    def _ensure_api_key(self) -> None:
        """
        Validates API key presence only when a network call is initiated.
        Allows offline development, instantiation, and mocked testing without a key.
        """
        if not self._api_key or self._api_key == "your_etherscan_api_key_here":
            logger.warning("Etherscan API key check failed: API key missing or unconfigured placeholder.")
            raise EtherscanConfigError(
                "Etherscan API key is missing or not configured. "
                "Please set the ETHERSCAN_API_KEY environment variable (see .env.example) "
                "or pass an explicit api_key to EtherscanClient."
            )
        logger.debug("Etherscan API key verified present (length=%d)", len(self._api_key))

    def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes an HTTP GET request to the Etherscan API V2 endpoint.
        Paces requests using rate_limiter, handles HTTP 429 and payload rate limits
        with bounded exponential backoff and jitter, and distinguishes quotas, invalid keys,
        and genuine empty wallet responses.
        """
        self._ensure_api_key()

        action = params.get("action", "unknown")
        module = params.get("module", "unknown")
        addr = params.get("address", "")
        masked_addr = mask_address(addr) if addr else "N/A"

        # Build full request parameters
        request_params = {
            "chainid": self.chain_id,
            "apikey": self._api_key,
            **params,
        }

        for attempt in range(self.max_retries + 1):
            # Enforce pacing before initiating the request
            if self.rate_limiter:
                self.rate_limiter.wait()

            logger.info(
                "Etherscan V2 request (attempt %d/%d): module=%s, action=%s, address=%s, chainid=%s, page=%s",
                attempt + 1, self.max_retries + 1, module, action, masked_addr, self.chain_id, params.get("page", 1)
            )

            try:
                response = self._session.get(
                    self.base_url,
                    params=request_params,
                    timeout=self.timeout,
                )
            except requests.exceptions.Timeout as err:
                clean_err = sanitize_log_message(err)
                logger.error("Etherscan request timed out: %s", clean_err)
                raise EtherscanTimeoutError(f"Etherscan request timed out after {self.timeout}s: {clean_err}") from None
            except requests.exceptions.RequestException as err:
                clean_err = sanitize_log_message(err)
                logger.error("Etherscan network error: %s", clean_err)
                raise EtherscanNetworkError(f"Network error during Etherscan API request: {clean_err}") from None

            # Handle HTTP 429 Rate Limit
            if response.status_code == 429:
                if attempt < self.max_retries:
                    # Respect Retry-After header if provided
                    retry_after_hdr = response.headers.get("Retry-After") if hasattr(response, "headers") else None
                    retry_delay = None
                    if retry_after_hdr:
                        try:
                            retry_delay = min(max(float(retry_after_hdr), 0.0), 30.0)
                        except (ValueError, TypeError):
                            retry_delay = None

                    if retry_delay is None:
                        retry_delay = self.base_backoff * (2 ** attempt) + random.uniform(0.1, 0.4)

                    logger.warning(
                        "Etherscan HTTP 429 rate limit on %s (attempt %d/%d). Backing off for %.2fs...",
                        action, attempt + 1, self.max_retries + 1, retry_delay
                    )
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.warning("Etherscan HTTP 429 Rate Limit exceeded after %d retries for %s", self.max_retries, action)
                    raise EtherscanRateLimitError("Etherscan API rate limit exceeded (HTTP 429).")

            if response.status_code != 200:
                clean_text = sanitize_log_message(response.text[:200])
                logger.error("Etherscan HTTP error status=%d: %s", response.status_code, clean_text)
                raise EtherscanNetworkError(
                    f"Etherscan API returned HTTP {response.status_code}: {clean_text}"
                )

            try:
                data = response.json()
            except ValueError as err:
                clean_text = sanitize_log_message(response.text[:200])
                logger.error("Failed to parse Etherscan JSON response: %s", clean_text)
                raise EtherscanAPIError(f"Failed to parse Etherscan JSON response: {clean_text}") from None

            # Inspect Etherscan status protocol
            status = str(data.get("status", ""))
            message = str(data.get("message", ""))
            result = data.get("result")

            logger.debug(
                "Etherscan response: status=%s, message=%s, result_type=%s, result_len=%s",
                status, message, type(result).__name__, len(result) if isinstance(result, list) else "N/A"
            )

            # Check status 0 conditions
            if status == "0":
                res_str = str(result).lower()
                msg_str = message.lower()

                # Case 1: Genuine empty wallet response
                # Etherscan returns status='0', message='No transactions found', result=[] (or string message)
                if "no transactions found" in res_str or "no transactions found" in msg_str:
                    logger.info("Etherscan returned genuine empty transaction history for address=%s", masked_addr)
                    return {"status": "1", "message": "OK", "result": []}
                if result == [] or result is None:
                    if "ok" in msg_str or "no transactions" in msg_str:
                        logger.info("Etherscan returned empty result for address=%s", masked_addr)
                        return {"status": "1", "message": "OK", "result": []}

                # Case 2: Daily request quota exhausted
                if any(q in res_str or q in msg_str for q in ["daily limit", "daily quota", "limit reached: daily", "endpoint daily limit"]):
                    clean_res = sanitize_log_message(result)
                    logger.error("Etherscan daily quota exhausted: %s", clean_res)
                    raise EtherscanQuotaExceededError(f"Etherscan API daily quota exhausted: {clean_res}")

                # Case 3: Rate limit reached in application payload ("Max rate limit reached")
                if "rate limit" in res_str or "max rate limit" in res_str:
                    clean_res = sanitize_log_message(result)
                    if attempt < self.max_retries:
                        retry_delay = self.base_backoff * (2 ** attempt) + random.uniform(0.1, 0.4)
                        logger.warning(
                            "Etherscan payload rate limit: %s (attempt %d/%d). Backing off for %.2fs...",
                            clean_res, attempt + 1, self.max_retries + 1, retry_delay
                        )
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.warning("Etherscan payload rate limit persisted after %d retries: %s", self.max_retries, clean_res)
                        raise EtherscanRateLimitError(f"Etherscan API rate limit reached: {clean_res}")

                # Case 4: Invalid API key
                if "invalid api key" in res_str or "missing/invalid api key" in res_str:
                    clean_res = sanitize_log_message(result)
                    logger.error("Etherscan API key rejected: %s", clean_res)
                    raise EtherscanAPIError(f"Etherscan API key rejected: {clean_res}")

                # Case 5: Other API errors
                clean_res = sanitize_log_message(result)
                logger.error("Etherscan API error (message='%s'): %s", message, clean_res)
                raise EtherscanAPIError(f"Etherscan API error (message='{message}'): {clean_res}")

            return data

        raise EtherscanRateLimitError("Etherscan API rate limit exceeded after maximum retries.")

    def get_normal_transactions(
        self,
        address: str,
        page: int = 1,
        offset: int = config.DEFAULT_PAGE_SIZE,
        sort: str = "asc",
        startblock: int = 0,
        endblock: int = 99999999,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves a single page of normal transactions for an Ethereum address.
        """
        if not is_valid_ethereum_address(address):
            raise InvalidAddressError(f"Invalid Ethereum address format: '{address}'")

        params = {
            "module": "account",
            "action": "txlist",
            "address": address.strip(),
            "startblock": int(startblock),
            "endblock": int(endblock),
            "page": int(page),
            "offset": int(offset),
            "sort": sort,
        }

        data = self._make_request(params)
        result = data.get("result", [])
        if isinstance(result, list):
            return result
        return []

    def get_all_normal_transactions(
        self,
        address: str,
        max_pages: int = config.DEFAULT_MAX_PAGES,
        offset: int = config.DEFAULT_PAGE_SIZE,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Retrieves all normal transactions across paginated requests.
        Returns:
            Tuple[List[Dict], bool]: (all_transactions, is_history_complete)
        """
        all_txs: List[Dict[str, Any]] = []
        seen_hashes = set()
        is_complete = True

        for page in range(1, max_pages + 1):
            page_txs = self.get_normal_transactions(address, page=page, offset=offset, sort="asc")
            if not page_txs:
                break

            for tx in page_txs:
                tx_hash = tx.get("hash")
                if tx_hash and tx_hash in seen_hashes:
                    continue
                if tx_hash:
                    seen_hashes.add(tx_hash)
                all_txs.append(tx)

            # If fewer records than offset were returned, we have reached the end of history
            if len(page_txs) < offset:
                is_complete = True
                break
        else:
            # If the loop finished without encountering a short page, max_pages was reached
            is_complete = False

        return all_txs, is_complete

    def get_erc20_transfers(
        self,
        address: str,
        page: int = 1,
        offset: int = config.DEFAULT_PAGE_SIZE,
        sort: str = "asc",
        startblock: int = 0,
        endblock: int = 99999999,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves a single page of ERC-20 token transfers for an Ethereum address.
        """
        if not is_valid_ethereum_address(address):
            raise InvalidAddressError(f"Invalid Ethereum address format: '{address}'")

        params = {
            "module": "account",
            "action": "tokentx",
            "address": address.strip(),
            "startblock": int(startblock),
            "endblock": int(endblock),
            "page": int(page),
            "offset": int(offset),
            "sort": sort,
        }

        data = self._make_request(params)
        result = data.get("result", [])
        if isinstance(result, list):
            return result
        return []

    def get_all_erc20_transfers(
        self,
        address: str,
        max_pages: int = config.DEFAULT_MAX_PAGES,
        offset: int = config.DEFAULT_PAGE_SIZE,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Retrieves all ERC-20 token transfers across paginated requests.
        Returns:
            Tuple[List[Dict], bool]: (all_transfers, is_history_complete)
        """
        all_transfers: List[Dict[str, Any]] = []
        seen_transfer_ids = set()
        is_complete = True

        for page in range(1, max_pages + 1):
            page_transfers = self.get_erc20_transfers(address, page=page, offset=offset, sort="asc")
            if not page_transfers:
                break

            for transfer in page_transfers:
                # Deduplicate by tx hash + logIndex or hash + contract + value
                tx_hash = transfer.get("hash", "")
                log_idx = transfer.get("logIndex", "")
                tx_id = f"{tx_hash}_{log_idx}"
                if tx_id in seen_transfer_ids:
                    continue
                seen_transfer_ids.add(tx_id)
                all_transfers.append(transfer)

            if len(page_transfers) < offset:
                is_complete = True
                break
        else:
            is_complete = False

        return all_transfers, is_complete

    def fetch_wallet_history(
        self,
        address: str,
        max_pages: int = config.DEFAULT_MAX_PAGES,
    ) -> Dict[str, Any]:
        """
        Fetches both normal transactions and ERC-20 token transfers for an address.
        Reuses cached transaction data within cache_ttl seconds to prevent redundant network bursts.
        """
        if not is_valid_ethereum_address(address):
            raise InvalidAddressError(f"Invalid Ethereum address format: '{address}'")

        target_addr = address.strip().lower()
        masked = mask_address(target_addr)

        # Check in-memory cache first
        cache_key = (target_addr, max_pages)
        if self.cache_ttl > 0:
            with _CACHE_LOCK:
                if cache_key in _HISTORY_CACHE:
                    cached_time, cached_data = _HISTORY_CACHE[cache_key]
                    age = time.time() - cached_time
                    if age < self.cache_ttl:
                        logger.info("Reusing cached wallet history for %s (age=%.1fs, ttl=%.1fs)", masked, age, self.cache_ttl)
                        return dict(cached_data)

        logger.info("Starting fresh wallet history retrieval for %s (max_pages=%d)", masked, max_pages)

        norm_txs, norm_complete = self.get_all_normal_transactions(target_addr, max_pages=max_pages)
        erc20_txs, erc20_complete = self.get_all_erc20_transfers(target_addr, max_pages=max_pages)

        logger.info(
            "Retrieved wallet history for %s: %d normal txs (complete=%s), %d ERC-20 transfers (complete=%s)",
            masked, len(norm_txs), norm_complete, len(erc20_txs), erc20_complete
        )

        result = {
            "address": target_addr,
            "normal_transactions": norm_txs,
            "erc20_transfers": erc20_txs,
            "is_normal_complete": norm_complete,
            "is_erc20_complete": erc20_complete,
            "is_history_complete": norm_complete and erc20_complete,
            "total_normal_fetched": len(norm_txs),
            "total_erc20_fetched": len(erc20_txs),
        }

        # Store in cache
        if self.cache_ttl > 0:
            with _CACHE_LOCK:
                _HISTORY_CACHE[cache_key] = (time.time(), result)

        return result
