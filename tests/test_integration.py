"""
Simple integration tests for the fraud detection pipeline.
Focus: end-to-end sanity, data flow, and persistence.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, Mock
import joblib

from src.data_preprocessing import (
    load_data,
    split_data,
    scale_features,
    handle_imbalance,
    prepare_features_and_target,
)
from src.feature_engineering import engineer_all_features
from src.train import train_model


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def dataset(workspace):
    np.random.seed(42)
    n = 500

    data = {
        "Time": np.random.randint(0, 86400, n),
        "Amount": np.random.lognormal(4, 1, n),
        "Class": np.random.choice([0, 1], n, p=[0.98, 0.02]),
    }

    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n)

    df = pd.DataFrame(data)
    path = workspace / "data/raw/creditcard.csv"
    df.to_csv(path, index=False)
    return df, path


@pytest.fixture
def config(workspace):
    return {
        "data": {
            "raw_path": str(workspace / "data/raw/creditcard.csv"),
            "processed_path": str(workspace / "data/processed/") + "/",
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
        },
        "features": {
            "target": "Class",
            "numerical": ["Amount", "Time"],
            "pca_features": [f"V{i}" for i in range(1, 29)],
            "engineered": ["hour_of_day", "amount_log", "amount_scaled"],
        },
        "project": {"random_state": 42},
        "model": {
            "params": {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.1,
                "num_leaves": 15,
                "verbose": -1,
                "random_state": 42,
            }
        },
        "training": {
            "n_estimators": 10,
            "early_stopping_rounds": 3,
            "use_smote": True,
            "smote_ratio": 0.3,
        },
        "mlflow": {
            "tracking_uri": "mlruns",
            "experiment_name": "test",
            "model_name": "test_model",
        },
        "api": {
            "model_path": str(workspace / "models/model.pkl"),
            "scaler_path": str(workspace / "models/scaler.pkl"),
            "threshold": 0.5,
        },
    }


# ---------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------

@patch("src.train.mlflow")
def test_full_pipeline_runs(mock_mlflow, dataset, config):
    mock_mlflow.start_run.return_value.__enter__ = Mock()
    mock_mlflow.start_run.return_value.__exit__ = Mock()

    # Load
    df = load_data(config)
    assert len(df) > 0

    # Feature engineering
    df = engineer_all_features(df, config)
    assert "hour_of_day" in df.columns

    # Split
    train_df, val_df, test_df = split_data(df, config)

    # Prepare
    X_train, y_train = prepare_features_and_target(train_df, config)
    X_val, y_val = prepare_features_and_target(val_df, config)
    X_test, _ = prepare_features_and_target(test_df, config)

    # Scale
    X_train, X_val, X_test, scaler = scale_features(
        X_train,
        X_val,
        X_test,
        ["Amount", "Time", "amount_scaled"],
        config["api"]["scaler_path"],
    )

    assert Path(config["api"]["scaler_path"]).exists()

    # Handle imbalance
    X_train, y_train = handle_imbalance(X_train, y_train, config)

    # Train
    model, metrics = train_model(X_train, y_train, X_val, y_val, config)

    assert model is not None
    assert "f1_score" in metrics
    assert Path(config["api"]["model_path"]).exists()


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------

@patch("src.train.mlflow")
def test_model_and_scaler_reload(mock_mlflow, dataset, config):
    mock_mlflow.start_run.return_value.__enter__ = Mock()
    mock_mlflow.start_run.return_value.__exit__ = Mock()

    df = engineer_all_features(load_data(config), config)
    train_df, val_df, _ = split_data(df, config)

    X_train, y_train = prepare_features_and_target(train_df, config)
    X_val, y_val = prepare_features_and_target(val_df, config)

    X_train, X_val, _, scaler = scale_features(
        X_train,
        X_val,
        X_val,
        ["Amount", "Time", "amount_scaled"],
        config["api"]["scaler_path"],
    )

    model, _ = train_model(X_train, y_train, X_val, y_val, config)

    loaded_model = joblib.load(config["api"]["model_path"])
    loaded_scaler = joblib.load(config["api"]["scaler_path"])

    preds1 = model.predict(X_val.values[:5])
    preds2 = loaded_model.predict(X_val.values[:5])

    np.testing.assert_allclose(preds1, preds2)
    np.testing.assert_allclose(
        scaler.transform(X_val[["Amount", "Time", "amount_scaled"]].values[:5]),
        loaded_scaler.transform(X_val[["Amount", "Time", "amount_scaled"]].values[:5]),
    )


# ---------------------------------------------------------------------
# Consistency & safety
# ---------------------------------------------------------------------

def test_no_data_leakage(dataset, config):
    df = engineer_all_features(load_data(config), config)
    train_df, val_df, test_df = split_data(df, config)

    assert set(train_df.index).isdisjoint(val_df.index)
    assert set(train_df.index).isdisjoint(test_df.index)
    assert set(val_df.index).isdisjoint(test_df.index)


def test_feature_columns_consistent(dataset, config):
    df = engineer_all_features(load_data(config), config)
    train_df, val_df, test_df = split_data(df, config)

    X_train, _ = prepare_features_and_target(train_df, config)
    X_val, _ = prepare_features_and_target(val_df, config)
    X_test, _ = prepare_features_and_target(test_df, config)

    assert X_train.columns.tolist() == X_val.columns.tolist() == X_test.columns.tolist()
