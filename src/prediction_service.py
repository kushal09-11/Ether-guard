"""
Prediction Service for Ethereum Wallet Fraud Risk Assessment (ChainSentinel - Phase 3).
Loads the trained model pipeline, integrates live feature extraction and schema validation,
applies validation-tuned decision thresholds, and generates structured risk assessments.
Provides explicit cold-start warnings and disclaims real-world calibrated probabilities.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

import config
from src.model_pipeline import load_model_pipeline
from src.etherscan_client import EtherscanClient
from src.feature_extractor import LiveFeatureExtractor, FeatureExtractionResult
from src.model_validator import validate_model_input, ModelInputValidationError


class WalletPredictionService:
    """
    Thin prediction service for live Ethereum wallet risk assessment.
    Integrates retrieval, feature calculation, validation, and pipeline inference.
    """

    def __init__(
        self,
        model_path: Path = config.BEST_MODEL_PATH,
        etherscan_client: Optional[EtherscanClient] = None,
        feature_extractor: Optional[LiveFeatureExtractor] = None,
        decision_threshold: float = config.OPERATIONAL_DECISION_THRESHOLD,
    ):
        self.model_path = model_path
        self.client = etherscan_client or EtherscanClient()
        self.extractor = feature_extractor or LiveFeatureExtractor()
        self.decision_threshold = decision_threshold
        self._pipeline = None

    @property
    def pipeline(self):
        """Lazily loads and caches the serialized model pipeline."""
        if self._pipeline is None:
            self._pipeline = load_model_pipeline(self.model_path)
        return self._pipeline

    def assess_from_features(
        self,
        features: Dict[str, Any],
        address: str = "0x0000000000000000000000000000000000000000",
        is_history_complete: bool = True,
        raw_normal_count: Optional[int] = None,
        raw_erc20_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Produces a risk assessment directly from an extracted feature vector.
        Useful for offline evaluation, unit testing, and pre-extracted datasets.
        """
        target = address.strip().lower()
        warnings: List[str] = []

        # Derive counts if not explicitly supplied
        norm_count = int(features.get("total_transactions_count", 0)) if raw_normal_count is None else raw_normal_count
        token_count = int(features.get("erc20_total_tnx", 0)) if raw_erc20_count is None else raw_erc20_count

        # 1. Check for zero-activity inputs
        is_zero_activity = (
            features.get("total_transactions_count", 0) == 0
            and features.get("erc20_total_tnx", 0) == 0
        )
        if is_zero_activity:
            return {
                "address": target,
                "is_scoreable": False,
                "decision": "INSUFFICIENT_ACTIVITY",
                "risk_score": None,
                "decision_threshold": self.decision_threshold,
                "risk_level": "UNKNOWN",
                "has_cold_start": True,
                "raw_normal_count": 0 if raw_normal_count is None else raw_normal_count,
                "raw_erc20_count": 0 if raw_erc20_count is None else raw_erc20_count,
                "is_history_complete": is_history_complete,
                "warnings": [
                    "Wallet has zero recorded transactions. Unscoreable due to absence of on-chain activity."
                ],
                "features": features,
                "disclaimers": self._get_standard_disclaimers(),
            }

        # 2. Check for cold-start (short lifespan or single transaction)
        lifespan_mins = float(features.get("time_diff_first_last_mins", 0.0))
        total_txs = float(features.get("total_transactions_count", 0.0))
        has_cold_start = (lifespan_mins < config.COLD_START_THRESHOLD_MINS) or (total_txs <= 1.0)

        if has_cold_start:
            warnings.append(
                f"COLD-START WARNING: Wallet observation lifespan is {lifespan_mins:.1f} minutes "
                f"({lifespan_mins / 60.0:.2f} hours) across {int(total_txs)} transaction(s). "
                "Because tree-based models heavily penalize short lifespans, "
                "recently created legitimate wallets are at elevated risk of false-positive classification. "
                "Do NOT treat this risk score as confirmed fraud."
            )

        if not is_history_complete:
            warnings.append(
                "WARNING: Transaction history was truncated by pagination limits. "
                "Calculated features represent partial activity."
            )

        # 3. Strict schema and numerical validation
        df = validate_model_input(features)

        # 4. Model inference
        prob_array = self.pipeline.predict_proba(df)
        fraud_risk_score = float(round(prob_array[0, 1], 4))

        # 5. Threshold application
        is_flagged = bool(fraud_risk_score >= self.decision_threshold)
        decision = "FLAGGED" if is_flagged else "NORMAL"

        if fraud_risk_score >= 0.75:
            risk_level = "HIGH"
        elif fraud_risk_score >= self.decision_threshold:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "address": target,
            "is_scoreable": True,
            "decision": decision,
            "risk_score": fraud_risk_score,
            "decision_threshold": self.decision_threshold,
            "risk_level": risk_level,
            "has_cold_start": has_cold_start,
            "raw_normal_count": norm_count,
            "raw_erc20_count": token_count,
            "is_history_complete": is_history_complete,
            "warnings": warnings,
            "features": {col: float(df[col].iloc[0]) for col in config.CANONICAL_24_FEATURES},
            "disclaimers": self._get_standard_disclaimers(),
        }

    def assess_wallet(
        self,
        address: str,
        max_pages: int = config.DEFAULT_MAX_PAGES,
    ) -> Dict[str, Any]:
        """
        End-to-end wallet assessment:
        Retrieves live data via Etherscan API V2, extracts features, validates input, and evaluates risk.
        """
        # 1. Fetch live blockchain transaction history
        history = self.client.fetch_wallet_history(address, max_pages=max_pages)

        # 2. Extract 24 canonical features
        ext_result = self.extractor.extract_features(
            address=address,
            normal_transactions=history["normal_transactions"],
            erc20_transfers=history["erc20_transfers"],
            is_history_complete=history["is_history_complete"],
        )

        # 3. Assess risk from extracted features
        assessment = self.assess_from_features(
            features=ext_result.features,
            address=address,
            is_history_complete=ext_result.is_history_complete,
            raw_normal_count=ext_result.raw_normal_count,
            raw_erc20_count=ext_result.raw_erc20_count,
        )

        # Merge warnings from feature extraction if any
        for w in ext_result.warnings:
            if w not in assessment["warnings"]:
                assessment["warnings"].append(w)

        return assessment

    def _get_standard_disclaimers(self) -> List[str]:
        """Standard scientific disclaimers for model predictions."""
        return [
            "EXPERIMENTAL CASE STUDY: This risk score is generated by an academic model trained on a historical Kaggle dataset.",
            "UNCALIBRATED RISK SCORE: The output score represents a classification decision metric, NOT a calibrated posterior probability of real-world fraud.",
            "SURVIVAL BIAS CAVEAT: Wallets with short activity spans are heavily penalized by historical feature patterns.",
            "NO FINANCIAL WARRANTY: This assessment must not be used as sole evidence for transaction blocking, asset freezing, or financial decisions.",
        ]
