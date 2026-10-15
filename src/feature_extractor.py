"""
Live Feature Extraction Engine for Ethereum Wallets (ChainSentinel - Phase 3).
Converts raw Etherscan API V2 transaction records into the canonical 24-feature schema
ready for validation and model inference.
Documents compatibility classifications, edge cases, and cold-start flags.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import pandas as pd

import config
from src.feature_calculator import calculate_common_features_from_etherscan, WEI_IN_ETHER


@dataclass
class FeatureExtractionResult:
    """Encapsulates computed wallet features and metadata flags."""
    address: str
    features: Dict[str, float]
    raw_normal_count: int
    raw_erc20_count: int
    is_zero_activity: bool
    is_history_complete: bool
    has_cold_start: bool
    warnings: List[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts features to a single-row DataFrame in the canonical feature order."""
        ordered_data = {col: [self.features[col]] for col in config.CANONICAL_24_FEATURES}
        return pd.DataFrame(ordered_data, columns=config.CANONICAL_24_FEATURES)


class LiveFeatureExtractor:
    """
    Extracts the canonical 24 features from raw Etherscan transaction lists.
    Validates completeness, flags cold-start accounts, and surfaces compatibility limitations.
    """

    def __init__(
        self,
        cold_start_threshold_mins: float = config.COLD_START_THRESHOLD_MINS,
    ):
        self.cold_start_threshold_mins = cold_start_threshold_mins

    def extract_features(
        self,
        address: str,
        normal_transactions: List[Dict[str, Any]],
        erc20_transfers: Optional[List[Dict[str, Any]]] = None,
        is_history_complete: bool = True,
    ) -> FeatureExtractionResult:
        """
        Extracts 24 canonical features for an address from raw transaction responses.
        """
        norm_txs = normal_transactions or []
        erc_txs = erc20_transfers or []
        target = address.strip().lower()
        warnings: List[str] = []

        raw_norm_len = len(norm_txs)
        raw_erc_len = len(erc_txs)
        is_zero_act = (raw_norm_len == 0 and raw_erc_len == 0)

        # 1. Compute canonical 24 features
        raw_features = calculate_common_features_from_etherscan(
            wallet_address=target,
            normal_transactions=norm_txs,
            token_transactions=erc_txs,
        )

        # Enforce canonical ordering and float types
        canonical_features: Dict[str, float] = {}
        for feat in config.CANONICAL_24_FEATURES:
            canonical_features[feat] = float(raw_features.get(feat, 0.0))

        # 2. Check for zero on-chain activity
        if is_zero_act:
            warnings.append(
                "Wallet has zero recorded transactions on Ethereum Mainnet. "
                "Feature vector consists of baseline zeroes."
            )

        # 3. Check for incomplete history (pagination limit reached)
        if not is_history_complete:
            warnings.append(
                "Transaction history is incomplete due to API pagination limits. "
                "Derived metrics represent partial wallet activity and may underestimate totals."
            )

        # 4. Check for cold-start (short lifespan or minimal transaction volume)
        lifespan_mins = canonical_features["time_diff_first_last_mins"]
        total_txs = canonical_features["total_transactions_count"]
        has_cold_start = (lifespan_mins < self.cold_start_threshold_mins) or (total_txs <= 1)

        if has_cold_start and not is_zero_act:
            warnings.append(
                f"COLD-START WARNING: Wallet observation lifespan is {lifespan_mins:.1f} minutes "
                f"({lifespan_mins / 60.0:.2f} hours) across {int(total_txs)} transactions. "
                "Because models heavily weight transaction lifespan (time_diff_first_last_mins), "
                "recently created legitimate wallets are at elevated risk of false-positive fraud flagging."
            )

        return FeatureExtractionResult(
            address=target,
            features=canonical_features,
            raw_normal_count=raw_norm_len,
            raw_erc20_count=raw_erc_len,
            is_zero_activity=is_zero_act,
            is_history_complete=is_history_complete,
            has_cold_start=has_cold_start,
            warnings=warnings,
        )
