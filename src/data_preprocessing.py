from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import get_logger

logger = get_logger(__name__)


# Data loading & cleaning
def load_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Load and validate raw credit card transaction data."""
    data_path = Path(config["data"]["raw_path"])

    logger.info(f"Loading data from {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    if df.empty:
        raise ValueError("Loaded dataset is empty")

    if df.isnull().any().any():
        logger.warning("Missing values detected, handling...")
        df = handle_missing_values(df)

    if "Class" in df.columns:
        _log_class_distribution(df, "Full dataset")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric NaNs with median and drop rows with missing target."""
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    if "Class" in df.columns and df["Class"].isnull().any():
        before = len(df)
        df = df.dropna(subset=["Class"])
        logger.warning(f"Dropped {before - len(df)} rows with missing target")

    return df


# Data splitting
def split_data(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, validation, and test sets."""
    train_r = config["data"]["train_split"]
    val_r = config["data"]["val_split"]
    test_r = config["data"]["test_split"]
    random_state = config["project"]["random_state"]

    if not np.isclose(train_r + val_r + test_r, 1.0):
        raise ValueError("Train/val/test ratios must sum to 1.0")

    logger.info(f"Splitting data (train={train_r}, val={val_r}, test={test_r})")

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_r,
        stratify=df["Class"],
        random_state=random_state,
    )

    val_adjusted = val_r / (train_r + val_r)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_adjusted,
        stratify=train_val_df["Class"],
        random_state=random_state,
    )

    _log_class_distribution(train_df, "Train")
    _log_class_distribution(val_df, "Validation")
    _log_class_distribution(test_df, "Test")

    return train_df, val_df, test_df


# Feature scaling
def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    numerical_features: list,
    scaler_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scale numerical features using StandardScaler."""
    logger.info(f"Scaling numerical features: {numerical_features}")

    scaler = StandardScaler()
    scaler.fit(X_train[numerical_features])

    def _apply_scaling(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[numerical_features] = scaler.transform(df[numerical_features])
        return df

    X_train = _apply_scaling(X_train)
    X_val = _apply_scaling(X_val)
    X_test = _apply_scaling(X_test)

    if scaler_path:
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

    return X_train, X_val, X_test, scaler


# Class imbalance handling
def handle_imbalance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE if enabled in configuration."""
    if not config["training"]["use_smote"]:
        logger.info("SMOTE disabled")
        return X_train, y_train

    smote_ratio = config["training"]["smote_ratio"]
    random_state = config["project"]["random_state"]

    _log_label_distribution(y_train, "Before SMOTE")

    smote = SMOTE(
        sampling_strategy=smote_ratio,
        random_state=random_state,
        k_neighbors=5,
    )

    try:
        X_res, y_res = smote.fit_resample(X_train, y_train)
    except ValueError as e:
        logger.warning(f"SMOTE skipped: {e}")
        return X_train, y_train

    _log_label_distribution(y_res, "After SMOTE")

    return (
        pd.DataFrame(X_res, columns=X_train.columns),
        pd.Series(y_res, name=y_train.name),
    )


# Feature/target preparation
def prepare_features_and_target(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and target column."""
    target = config["features"]["target"]

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    X = df.drop(columns=[target])
    y = df[target]

    logger.info(f"Prepared X: {X.shape}, y: {y.shape}")

    return X, y


# Saving processed data
def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
) -> None:
    """Save processed datasets to disk."""
    output_dir = Path(config["data"]["processed_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    logger.info(f"Processed datasets saved to {output_dir}")


# Small internal helpers
def _log_class_distribution(df: pd.DataFrame, name: str) -> None:
    fraud_ratio = df["Class"].mean()
    logger.info(f"{name} set fraud ratio: {fraud_ratio:.4f}")


def _log_label_distribution(y: pd.Series, stage: str) -> None:
    logger.info(f"{stage} fraud ratio: {y.mean():.4f}")
