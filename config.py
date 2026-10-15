"""
Project Configuration for Ethereum Wallet Fraud Detection (Phase 1).
Defines paths, column constants, data-cleaning parameters, and Etherscan API V2 metadata.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Automatically load environment variables from .env in project root if present
ENV_FILE_PATH = PROJECT_ROOT / ".env"
try:
    from dotenv import load_dotenv
    if ENV_FILE_PATH.is_file():
        load_dotenv(dotenv_path=ENV_FILE_PATH, override=False)
except ImportError:
    # Graceful fallback parser if python-dotenv is not yet installed
    if ENV_FILE_PATH.is_file():
        import os
        with open(ENV_FILE_PATH, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    _k = _k.strip()
                    _v = _v.strip().strip("'\"")
                    if _k and _k not in os.environ:
                        os.environ[_k] = _v

# Directory paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"

# Dataset paths
DEFAULT_RAW_CSV_NAME = "transaction_dataset.csv"
RAW_DATASET_PATH = RAW_DATA_DIR / DEFAULT_RAW_CSV_NAME
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "cleaned_ethereum_fraud_dataset.csv"
FEATURE_METADATA_PATH = REPORTS_DIR / "feature_metadata.json"

# Reports paths
INSPECTION_REPORT_PATH = REPORTS_DIR / "data_inspection_report.md"
FEATURE_AUDIT_CSV_PATH = REPORTS_DIR / "feature_compatibility_audit.csv"
FEATURE_AUDIT_MD_PATH = REPORTS_DIR / "feature_compatibility_audit.md"
COMMON_FEATURE_SCHEMA_PATH = REPORTS_DIR / "common_feature_schema.json"
PHASE1_SUMMARY_PATH = REPORTS_DIR / "phase1_summary.md"

# Target column configuration
TARGET_COLUMN = "FLAG"
TARGET_CLASS_LABELS = {
    0: "Non-Fraudulent / Normal Wallet",
    1: "Fraudulent / Illicit Wallet",
}

# Identifier and metadata columns to exclude from predictive features
IDENTIFIER_COLUMNS = [
    "Unnamed: 0",
    "Index",
    "Address",
]

# Zero-variance / constant columns in the raw dataset
CONSTANT_COLUMNS = [
    " ERC20 avg time between sent tnx",
    " ERC20 avg time between rec tnx",
    " ERC20 avg time between rec 2 tnx",
    " ERC20 avg time between contract tnx",
    " ERC20 min val sent contract",
    " ERC20 max val sent contract",
    " ERC20 avg val sent contract",
]

# Phase 2 Paths and Models
MODELS_DIR = PROJECT_ROOT / "models"
PHASE2_RESULTS_CSV = REPORTS_DIR / "phase2_evaluation_results.csv"
PHASE2_FEATURE_IMPORTANCE_CSV = REPORTS_DIR / "phase2_feature_importance.csv"
PHASE2_COMPARISON_MD = REPORTS_DIR / "phase2_model_comparison.md"
BEST_MODEL_PATH = MODELS_DIR / "best_model_pipeline.joblib"

# Canonical 24-feature baseline
CANONICAL_24_FEATURES = [
    "avg_min_between_sent_tnx",
    "avg_min_between_received_tnx",
    "time_diff_first_last_mins",
    "sent_tnx",
    "received_tnx",
    "created_contracts_count",
    "unique_received_from_addresses",
    "unique_sent_to_addresses",
    "min_value_received_eth",
    "max_value_received_eth",
    "avg_value_received_eth",
    "min_value_sent_eth",
    "max_value_sent_eth",
    "avg_value_sent_eth",
    "total_transactions_count",
    "total_ether_sent",
    "total_ether_received",
    "total_ether_balance",
    "erc20_total_tnx",
    "erc20_unique_sent_addr",
    "erc20_unique_received_addr",
    "erc20_unique_received_contract_addr",
    "erc20_unique_sent_token_name",
    "erc20_unique_received_token_name",
]

# Reduced feature set (21 features): removes 3 algebraic redundancies/collinearities
# 1. total_transactions_count (exact sum of sent_tnx + received_tnx + created_contracts_count)
# 2. total_ether_balance (exact difference of total_ether_received - total_ether_sent)
# 3. erc20_unique_received_contract_addr (r=0.9996 with erc20_unique_received_token_name)
REDUCED_21_FEATURES = [
    f for f in CANONICAL_24_FEATURES
    if f not in [
        "total_transactions_count",
        "total_ether_balance",
        "erc20_unique_received_contract_addr",
    ]
]

# Ablation feature set (18 features): base ETH only, removing all 6 ERC-20 features
ETH_ONLY_18_FEATURES = [
    f for f in CANONICAL_24_FEATURES
    if not f.startswith("erc20_")
]

# Random seed for reproducibility
RANDOM_SEED = 42

# Etherscan API V2 Reference Documentation
ETHERSCAN_DOCS_URL = "https://docs.etherscan.io/"
ETHERSCAN_API_V2_BASE_URL = "https://api.etherscan.io/v2/api"

# Phase 3 Configuration
DEFAULT_ETHERSCAN_CHAIN_ID = "1"
DEFAULT_REQUEST_TIMEOUT = 15
DEFAULT_PAGE_SIZE = 1000
DEFAULT_MAX_PAGES = 10
COLD_START_THRESHOLD_MINS = 1440.0  # 24 hours (wallets with lifespan < 24h trigger cold-start warning)
PHASE3_REPORT_PATH = REPORTS_DIR / "phase3_api_feature_compatibility_report.md"

# Rate Limiting, Retry, and Caching Configuration (Etherscan Free Tier: 5 req/s max)
DEFAULT_ETHERSCAN_RATE_LIMIT_RPS = 3.0   # Conservative target <= 3 requests per second
DEFAULT_ETHERSCAN_MAX_RETRIES = 3        # Maximum retry attempts for transient rate limits
DEFAULT_ETHERSCAN_RETRY_BACKOFF = 1.0    # Base exponential backoff delay in seconds
DEFAULT_ETHERSCAN_CACHE_TTL_SECS = 60.0  # In-memory wallet history cache TTL in seconds

# Operational Fraud Risk Decision Threshold (Tuned on Validation split during Phase 2)
OPERATIONAL_DECISION_THRESHOLD = 0.51

# Ensure runtime directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


