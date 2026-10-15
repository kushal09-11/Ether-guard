"""
Model Input Validation Module (ChainSentinel - Phase 3).
Enforces strict schema compliance, rejects missing, extra, non-numeric, NaN, or infinite features.
Guarantees data integrity and prevents unscoreable data fabrication before model inference.
"""

from typing import Union, Dict, Any, List
import numpy as np
import pandas as pd

import config


class ModelInputValidationError(Exception):
    """Raised when an input feature vector fails schema or numerical integrity checks."""
    pass


def validate_model_input(
    features: Union[pd.DataFrame, Dict[str, Any]],
    expected_features: List[str] = config.CANONICAL_24_FEATURES,
) -> pd.DataFrame:
    """
    Validates that the input features strictly match the required schema:
    - Exactly the expected features in expected order.
    - No missing features.
    - No unexpected extra features.
    - All values must be finite numerical types (no NaN, nulls, inf, or non-numeric strings).
    - Never fabricates or imputes values to force scoreability.
    
    Returns:
        pd.DataFrame: Validated single- or multi-row DataFrame ready for pipeline inference.
    
    Raises:
        ModelInputValidationError: If any validation rule is violated.
    """
    # 1. Convert dictionary to DataFrame if needed
    if isinstance(features, dict):
        df = pd.DataFrame([features])
    elif isinstance(features, pd.DataFrame):
        df = features.copy()
    else:
        raise ModelInputValidationError(
            f"Unsupported feature input type: {type(features)}. Expected pd.DataFrame or dict."
        )

    if df.empty:
        raise ModelInputValidationError("Input feature DataFrame is empty.")

    # 2. Check for missing columns
    current_cols = list(df.columns)
    missing_cols = [col for col in expected_features if col not in current_cols]
    if missing_cols:
        raise ModelInputValidationError(
            f"Input features validation failed: Missing {len(missing_cols)} required feature(s): {missing_cols}"
        )

    # 3. Check for extra unexpected columns
    extra_cols = [col for col in current_cols if col not in expected_features]
    if extra_cols:
        raise ModelInputValidationError(
            f"Input features validation failed: Unexpected extra feature(s) detected: {extra_cols}"
        )

    # 4. Enforce strict canonical column ordering
    df = df[expected_features]

    # 5. Check for non-numeric data types and values
    for col in expected_features:
        # Attempt conversion or type verification
        try:
            numeric_series = pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError) as err:
            raise ModelInputValidationError(
                f"Feature '{col}' contains non-numeric values: {err}"
            ) from err

        # Check for NaN / missing values
        if numeric_series.isna().any():
            raise ModelInputValidationError(
                f"Feature '{col}' contains missing (NaN/None) values. Data fabrication is rejected."
            )

        # Check for infinite values (inf, -inf)
        vals = numeric_series.to_numpy(dtype=float)
        if np.isinf(vals).any():
            raise ModelInputValidationError(
                f"Feature '{col}' contains non-finite (infinite) numerical values."
            )

        df[col] = numeric_series.astype(float)

    return df
