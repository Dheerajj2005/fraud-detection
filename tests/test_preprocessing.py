"""
CI-friendly tests for data preprocessing module.

Focus:
- sanity checks
- deterministic behavior
- fast execution
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile

from src.data_preprocessing import (
    load_data,
    split_data,
    scale_features,
    handle_imbalance,
    prepare_features_and_target,
    save_processed_data,
)


# ------------------------------------------------------------------
# Fixtures (small, deterministic)
# ------------------------------------------------------------------


@pytest.fixture
def config():
    return {
        "data": {
            "raw_path": "",
            "processed_path": "",
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
        },
        "features": {
            "target": "Class",
        },
        "project": {
            "random_state": 42,
        },
        "training": {
            "use_smote": True,
            "smote_ratio": 0.5,
        },
    }


@pytest.fixture
def dataframe():
    """Small deterministic dataset."""
    return pd.DataFrame(
        {
            "Time": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "Amount": [100, 200, 150, 300, 250, 400, 500, 350, 200, 150],
            "V1": [0.1, -0.2, 0.3, 0.1, -0.1, 0.2, -0.3, 0.4, 0.1, 0.2],
            "V2": [0.0, 0.1, -0.1, 0.2, -0.2, 0.0, 0.1, -0.2, 0.3, -0.1],
            "Class": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        }
    )


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_load_data_success(dataframe, config, temp_dir):
    path = Path(temp_dir) / "data.csv"
    dataframe.to_csv(path, index=False)

    config["data"]["raw_path"] = str(path)
    df = load_data(config)

    assert not df.empty
    assert "Class" in df.columns


def test_split_data_sizes(dataframe, config):
    train, val, test = split_data(dataframe, config)

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert len(train) + len(val) + len(test) == len(dataframe)


def test_prepare_features_and_target(dataframe, config):
    X, y = prepare_features_and_target(dataframe, config)

    assert "Class" not in X.columns
    assert y.name == "Class"
    assert len(X) == len(y)


def test_scale_features_applies_only_to_numerical(dataframe, config):
    train, val, test = split_data(dataframe, config)

    X_train, _ = prepare_features_and_target(train, config)
    X_val, _ = prepare_features_and_target(val, config)
    X_test, _ = prepare_features_and_target(test, config)

    X_train_s, _, _, scaler = scale_features(
        X_train, X_val, X_test, numerical_features=["Amount", "Time"]
    )

    assert abs(X_train_s["Amount"].mean()) < 1
    assert "V1" in X_train_s.columns  # untouched feature


def test_handle_imbalance_smote_basic(dataframe, config):
    train, _, _ = split_data(dataframe, config)
    X_train, y_train = prepare_features_and_target(train, config)

    X_res, y_res = handle_imbalance(X_train, y_train, config)

    assert len(X_res) >= len(X_train)
    assert y_res.sum() >= y_train.sum()


def test_save_processed_data_creates_files(dataframe, config, temp_dir):
    config["data"]["processed_path"] = temp_dir

    train, val, test = split_data(dataframe, config)
    save_processed_data(train, val, test, config)

    assert (Path(temp_dir) / "train.csv").exists()
    assert (Path(temp_dir) / "val.csv").exists()
    assert (Path(temp_dir) / "test.csv").exists()


def test_simple_preprocessing_pipeline(dataframe, config, temp_dir):
    path = Path(temp_dir) / "data.csv"
    dataframe.to_csv(path, index=False)

    config["data"]["raw_path"] = str(path)
    config["data"]["processed_path"] = temp_dir

    df = load_data(config)
    train, val, test = split_data(df, config)

    X_train, y_train = prepare_features_and_target(train, config)
    X_train_s, _, _, _ = scale_features(
        X_train, X_train, X_train, numerical_features=["Amount", "Time"]
    )

    assert not X_train_s.isnull().any().any()
    assert y_train.notnull().all()
