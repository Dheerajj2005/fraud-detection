"""
Shared pytest fixtures for fraud detection tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import logging


# -------------------------
# Global test logging setup
# -------------------------


@pytest.fixture(autouse=True)
def quiet_logging():
    """Reduce log noise during tests."""
    logging.basicConfig(level=logging.WARNING)


# -------------------------
# Temporary workspace
# -------------------------


@pytest.fixture
def temp_dir():
    """Create and cleanup a temporary directory."""
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path)


# -------------------------
# Sample configuration
# -------------------------


@pytest.fixture
def test_config(temp_dir):
    """Minimal config used across tests."""
    return {
        "data": {
            "raw_path": str(temp_dir / "data.csv"),
            "processed_path": str(temp_dir) + "/",
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
        },
        "features": {
            "numerical": ["Amount", "Time"],
            "pca_features": [f"V{i}" for i in range(1, 29)],
            "target": "Class",
            "engineered": ["hour_of_day", "amount_log", "amount_scaled"],
        },
        "project": {"random_state": 42},
        "training": {"use_smote": False, "smote_ratio": 0.3},
        "api": {"threshold": 0.5},
    }


# -------------------------
# Sample datasets
# -------------------------


@pytest.fixture
def sample_dataframe():
    """Small realistic fraud dataset."""
    np.random.seed(42)
    n = 500

    data = {
        "Time": np.random.randint(0, 172800, n),
        "Amount": np.random.lognormal(mean=4, sigma=2, size=n),
        "Class": np.random.choice([0, 1], size=n, p=[0.99, 0.01]),
    }

    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n)

    return pd.DataFrame(data)


@pytest.fixture
def saved_csv(sample_dataframe, test_config):
    """Save dataset to disk and return path."""
    path = Path(test_config["data"]["raw_path"])
    sample_dataframe.to_csv(path, index=False)
    return path


# -------------------------
# API transactions
# -------------------------


@pytest.fixture
def sample_transaction():
    """Valid single transaction."""
    tx = {"Time": 12345.0, "Amount": 120.5}
    for i in range(1, 29):
        tx[f"V{i}"] = float(i) * 0.01
    return tx


@pytest.fixture
def fraud_transaction(sample_transaction):
    """High-risk transaction."""
    tx = sample_transaction.copy()
    tx["Amount"] = 5000.0
    return tx


# -------------------------
# Mock MLflow
# -------------------------


@pytest.fixture
def mock_mlflow():
    """Simple MLflow mock."""
    from unittest.mock import MagicMock, Mock

    mlflow = MagicMock()
    mlflow.start_run.return_value.__enter__ = Mock()
    mlflow.start_run.return_value.__exit__ = Mock()
    return mlflow
