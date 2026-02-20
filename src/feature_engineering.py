"""
Feature engineering module for fraud detection system.

Creates additional features from raw transaction data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

from src.utils import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Feature creation
# ------------------------------------------------------------------

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create hour-of-day feature from Time column."""
    if "Time" not in df.columns:
        raise ValueError("'Time' column not found")

    df = df.copy()

    seconds_per_day = 24 * 60 * 60
    df["hour_of_day"] = ((df["Time"] % seconds_per_day) / 3600).astype(int)

    if df["hour_of_day"].min() < 0 or df["hour_of_day"].max() > 23:
        logger.warning("hour_of_day values outside expected range [0, 23]")

    logger.info("Created time-based features")
    return df


def create_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create log-transformed and scalable amount features."""
    if "Amount" not in df.columns:
        raise ValueError("'Amount' column not found")

    df = df.copy()

    # Ensure non-negative values
    if (df["Amount"] < 0).any():
        logger.warning("Negative Amount values detected, setting to 0")
        df.loc[df["Amount"] < 0, "Amount"] = 0

    df["amount_log"] = np.log1p(df["Amount"])
    df["amount_scaled"] = df["Amount"]

    logger.info("Created amount-based features")
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features between engineered variables."""
    df = df.copy()

    if {"amount_log", "hour_of_day"}.issubset(df.columns):
        df["amount_hour_interaction"] = df["amount_log"] * df["hour_of_day"]
        logger.info("Created interaction features")

    return df


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def validate_features(df: pd.DataFrame, config: Dict[str, Any]) -> bool:
    """
    Basic feature validation:
    - No NaN or infinite values
    - Expected engineered features exist
    """
    if df.isnull().any().any():
        logger.warning("NaN values detected after feature engineering")
        return False

    numeric_cols = df.select_dtypes(include=np.number)
    if np.isinf(numeric_cols).any().any():
        logger.warning("Infinite values detected in numeric features")
        return False

    expected = config["features"].get("engineered", [])
    missing = [f for f in expected if f not in df.columns]
    if missing:
        logger.warning(f"Missing engineered features: {missing}")
        return False

    return True


# ------------------------------------------------------------------
# Pipeline orchestration
# ------------------------------------------------------------------

def engineer_all_features(
    df: pd.DataFrame,
    config: Dict[str, Any],
    validate: bool = True,
) -> pd.DataFrame:
    """Run full feature engineering pipeline."""
    logger.info("Starting feature engineering")

    original_cols = df.shape[1]

    df = create_time_features(df)
    df = create_amount_features(df)
    df = create_interaction_features(df)

    logger.info(
        f"Feature engineering complete "
        f"(added {df.shape[1] - original_cols} features)"
    )

    if validate and not validate_features(df, config):
        raise ValueError("Feature validation failed")

    return df


# ------------------------------------------------------------------
# Feature list helper
# ------------------------------------------------------------------

def get_feature_names(
    config: Dict[str, Any],
    include_engineered: bool = True,
) -> list:
    """Return ordered list of feature names from config."""
    features = []
    features.extend(config["features"]["numerical"])
    features.extend(config["features"]["pca_features"])

    if include_engineered:
        features.extend(config["features"].get("engineered", []))

    return features
