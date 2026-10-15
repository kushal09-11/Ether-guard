"""
ChainSentinel FastAPI Web Application (Phase 4).
Serves the REST API for Ethereum wallet risk assessment, model introspection,
and the interactive demonstration interface.
Strictly routes live wallet requests to WalletPredictionService, validates address inputs,
maps Etherscan exceptions cleanly without masking API errors as empty wallets,
and hosts offline demonstration presets.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import os
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from src.utils import mask_address
from src.prediction_service import WalletPredictionService
from src.etherscan_client import (
    EtherscanClient,
    is_valid_ethereum_address,
    InvalidAddressError,
    EtherscanConfigError,
    EtherscanRateLimitError,
    EtherscanQuotaExceededError,
    EtherscanTimeoutError,
    EtherscanNetworkError,
    EtherscanAPIError,
    sanitize_log_message,
)
from src.presets import DEMO_PRESETS

# Configure root and module logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chainsentinel.api")

# Base directory for static web assets
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="ChainSentinel API",
    description="Academic Ethereum Wallet Fraud Risk Assessment Service",
    version="1.0.0",
)

# Enable CORS for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared prediction service instance
prediction_service = WalletPredictionService()


# ---------------------------------------------------------
# Request Models
# ---------------------------------------------------------

class WalletAssessmentRequest(BaseModel):
    address: str = Field(..., description="Target Ethereum wallet address (0x + 40 hex chars)")
    max_pages: Optional[int] = Field(default=config.DEFAULT_MAX_PAGES, ge=1, le=20, description="Max transaction pages to fetch (1000 tx/page)")


class CustomFeatureAssessmentRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Dictionary containing the 24 canonical features")
    address: Optional[str] = Field(default="0x0000000000000000000000000000000000000000")


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.get("/api/health")
def get_health() -> Dict[str, Any]:
    """Health check endpoint indicating model state and Etherscan configuration."""
    # Read fresh environment state for API key check
    api_key = os.environ.get("ETHERSCAN_API_KEY", "").strip()
    has_key = bool(api_key and api_key != "your_etherscan_api_key_here")
    return {
        "status": "healthy",
        "service": "ChainSentinel",
        "model_loaded": prediction_service.pipeline is not None,
        "etherscan_configured": has_key,
        "decision_threshold": config.OPERATIONAL_DECISION_THRESHOLD,
        "canonical_features_count": len(config.CANONICAL_24_FEATURES),
    }


@app.get("/api/model-info")
def get_model_info() -> Dict[str, Any]:
    """
    Returns verified model architecture, held-out test evaluation metrics,
    training dataset metadata, and documented limitations.
    """
    return {
        "model_type": "HistGradientBoostingClassifier",
        "hyperparameters": {
            "max_depth": 6,
            "min_samples_leaf": 20,
            "random_state": 42,
        },
        "training_dataset": {
            "name": "Kaggle Ethereum Fraud Detection Dataset (Cleaned)",
            "total_wallets": 9816,
            "unique_addresses": 9816,
            "fraud_prevalence_pct": 21.79,
            "stratified_splits": "70% Train (6,870) / 15% Val (1,473) / 15% Test (1,473)",
        },
        "held_out_test_metrics": {
            "eval_split": "Held-Out Test Set (N=1,473)",
            "decision_threshold": 0.51,
            "confusion_matrix": {
                "true_negatives": 1131,
                "false_positives": 15,
                "false_negatives": 41,
                "true_positives": 286,
                "total": 1473,
            },
            "precision": 0.9502,
            "recall": 0.8746,
            "fraud_f1": 0.9108,
            "macro_f1": 0.9433,
            "pr_auc": 0.9736,
            "roc_auc": 0.9897,
            "brier_score": 0.0283,
        },
        "feature_parity_status": {
            "verified_equivalent": 0,
            "approximate_equivalent": 9,
            "not_verifiable_from_dataset": 15,
            "total_canonical": 24,
            "note": "15 features remain unverifiable because the historical Kaggle dataset discarded transaction-level error flags and receipts.",
        },
        "cold_start_policy": {
            "threshold_mins": config.COLD_START_THRESHOLD_MINS,
            "description": "Wallets with lifespan < 24 hours (1440 mins) or <= 1 tx trigger prominent false-positive caution.",
        },
        "zero_activity_policy": {
            "description": "Wallets with zero normal and zero ERC-20 transactions are rejected as unscoreable (INSUFFICIENT_ACTIVITY). No numeric risk score is computed.",
        },
        "canonical_features": config.CANONICAL_24_FEATURES,
    }


@app.get("/api/presets")
def get_presets() -> Dict[str, Any]:
    """Returns curated demonstration scenarios for offline testing and evaluation."""
    return {
        "presets": [
            {
                "id": p["id"],
                "name": p["name"],
                "badge": p["badge"],
                "badge_color": p["badge_color"],
                "address": p["address"],
                "description": p["description"],
                "is_live_query": p["is_live_query"],
            }
            for p in DEMO_PRESETS.values()
        ]
    }


@app.post("/api/assess-wallet")
def assess_wallet(req: WalletAssessmentRequest) -> Dict[str, Any]:
    """
    Main live assessment endpoint.
    Retrieves transaction history via Etherscan API V2, extracts features, validates schema,
    and returns model evaluation.
    Maps Etherscan API exceptions to clean, readable HTTP errors without ever converting API failures into zero-activity results.
    """
    address = req.address.strip()
    masked = mask_address(address)
    logger.info("Received assess-wallet request for %s (max_pages=%d)", masked, req.max_pages)

    if not is_valid_ethereum_address(address):
        logger.warning("Rejected invalid Ethereum address format: '%s'", address)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid Ethereum address format: '{address}'. Expected 0x followed by 40 hexadecimal characters.",
        )

    try:
        # Re-check API key dynamically in case environment was modified at runtime
        api_key = os.environ.get("ETHERSCAN_API_KEY", "").strip()
        client = EtherscanClient(api_key=api_key)
        service = WalletPredictionService(etherscan_client=client)

        result = service.assess_wallet(address, max_pages=req.max_pages)
        result["analysis_mode"] = "LIVE_ON_CHAIN"
        result["source_description"] = "Live on-chain analysis via Etherscan API V2 (Ethereum Mainnet)"
        logger.info(
            "Completed assessment for %s: mode=%s, is_scoreable=%s, decision=%s, risk_score=%s, cold_start=%s",
            masked, result["analysis_mode"], result["is_scoreable"], result["decision"], result.get("risk_score"), result.get("has_cold_start")
        )
        return result

    except InvalidAddressError as e:
        logger.warning("InvalidAddressError for %s: %s", masked, sanitize_log_message(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except EtherscanConfigError as e:
        logger.warning("EtherscanConfigError for %s: API key missing or unconfigured.", masked)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_type": "EtherscanConfigError",
                "message": "Etherscan API key is missing or not configured.",
                "help": "To analyze live on-chain wallets, create a free API key at https://etherscan.io and set the ETHERSCAN_API_KEY environment variable (see .env.example). You can still test the system offline using the demonstration presets.",
            },
        )

    except EtherscanRateLimitError as e:
        logger.warning("EtherscanRateLimitError for %s: %s", masked, sanitize_log_message(e))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error_type": "EtherscanRateLimitError",
                "message": "Etherscan API rate limit exceeded.",
                "help": "Free-tier Etherscan keys allow up to 5 requests per second. The client automatically retried with backoff, but the limit persisted. Please wait a few moments and retry.",
            },
        )

    except EtherscanQuotaExceededError as e:
        clean_err = sanitize_log_message(e)
        logger.error("EtherscanQuotaExceededError for %s: %s", masked, clean_err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_type": "EtherscanQuotaExceededError",
                "message": "Etherscan API 24-hour daily quota exhausted.",
                "help": "Your Etherscan API key has exceeded its daily quota (100,000 requests/day on free tier). The quota resets at midnight UTC. You can continue testing offline using the demonstration presets.",
            },
        )

    except EtherscanTimeoutError as e:
        logger.warning("EtherscanTimeoutError for %s: %s", masked, sanitize_log_message(e))
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "error_type": "EtherscanTimeoutError",
                "message": f"Etherscan API request timed out ({config.DEFAULT_REQUEST_TIMEOUT}s).",
                "help": "The blockchain API server took too long to respond. Please retry your analysis.",
            },
        )

    except (EtherscanNetworkError, EtherscanAPIError) as e:
        clean_err = sanitize_log_message(e)
        logger.error("Upstream Etherscan error (%s) for %s: %s", type(e).__name__, masked, clean_err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error_type": type(e).__name__,
                "message": clean_err,
                "help": "An upstream error occurred while communicating with Etherscan API V2. Please check your network connection and retry.",
            },
        )

    except Exception as e:
        clean_err = sanitize_log_message(e)
        logger.error("Unexpected error assessing %s: %s", masked, clean_err, exc_info=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_type": "InternalServerError",
                "message": f"Unexpected error during wallet assessment: {clean_err}",
            },
        )


@app.post("/api/assess-preset/{preset_id}")
def assess_preset(preset_id: str) -> Dict[str, Any]:
    """
    Evaluates one of the curated demonstration presets offline without requiring an Etherscan API key.
    """
    if preset_id not in DEMO_PRESETS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown preset '{preset_id}'. Available presets: {list(DEMO_PRESETS.keys())}",
        )

    preset = DEMO_PRESETS[preset_id]
    if preset["is_live_query"]:
        # Delegate live queries to assess_wallet
        return assess_wallet(WalletAssessmentRequest(address=preset["address"]))

    # Offline evaluation through prediction service
    result = prediction_service.assess_from_features(
        features=preset["features"],
        address=preset["address"],
        raw_normal_count=preset.get("raw_normal_count"),
        raw_erc20_count=preset.get("raw_erc20_count"),
        is_history_complete=preset.get("is_history_complete", True),
    )
    result["analysis_mode"] = "OFFLINE_PRESET"
    result["preset_id"] = preset_id
    result["preset_name"] = preset["name"]
    result["source_description"] = f"Curated offline demonstration scenario: {preset['name']}"
    return result


@app.post("/api/assess-features")
def assess_features(req: CustomFeatureAssessmentRequest) -> Dict[str, Any]:
    """
    Direct model evaluation from a custom 24-feature dictionary.
    Enables custom scenario testing and automated validation.
    """
    try:
        result = prediction_service.assess_from_features(
            features=req.features,
            address=req.address,
        )
        result["analysis_mode"] = "CUSTOM_FEATURES"
        result["source_description"] = "Custom feature vector evaluation (direct model inference)"
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Feature validation or inference failed: {str(e)}",
        )


# ---------------------------------------------------------
# Static Frontend Serving
# ---------------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index() -> FileResponse:
    """Serves the single-page application frontend."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse(
            status_code=200,
            content={
                "message": "ChainSentinel API is running. index.html not yet initialized.",
                "docs_url": "/docs",
            },
        )
    return FileResponse(str(index_file))


def start_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Entry point to run the web application server locally."""
    uvicorn.run("src.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    start_server()
