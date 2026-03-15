"""
Simple tests for model training module.
Focused on correctness, stability, and CI friendliness.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

from src.train import calculate_metrics, train_model


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture
def config(tmp_path):
    return {
        "model": {
            "params": {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.1,
                "num_leaves": 15,
                "random_state": 42,
                "verbose": -1,
            }
        },
        "training": {
            "n_estimators": 10,
            "early_stopping_rounds": 3,
            "use_smote": False,
            "smote_ratio": 0.3,
        },
        "mlflow": {
            "tracking_uri": "mlruns",
            "experiment_name": "test",
            "model_name": "test_model",
        },
        "api": {
            "model_path": str(tmp_path / "model.pkl"),
            "threshold": 0.5,
        },
    }


@pytest.fixture
def data():
    np.random.seed(42)

    cols = (
        ["Time", "Amount"]
        + [f"V{i}" for i in range(1, 29)]
        + ["hour_of_day", "amount_log"]
    )

    X_train = pd.DataFrame(np.random.randn(200, len(cols)), columns=cols)
    y_train = pd.Series(np.random.choice([0, 1], size=200, p=[0.95, 0.05]))

    X_val = pd.DataFrame(np.random.randn(50, len(cols)), columns=cols)
    y_val = pd.Series(np.random.choice([0, 1], size=50, p=[0.95, 0.05]))

    return X_train, y_train, X_val, y_val


# -------------------------------------------------------------------
# calculate_metrics
# -------------------------------------------------------------------


def test_calculate_metrics_basic():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.8, 0.9])

    metrics = calculate_metrics(y_true, y_pred, y_prob)

    expected_keys = {
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
        "specificity",
        "false_positive_rate",
    }

    assert expected_keys.issubset(metrics.keys())
    assert 0 <= metrics["f1_score"] <= 1


# -------------------------------------------------------------------
# train_model
# -------------------------------------------------------------------


@patch("src.train.mlflow")
@patch("src.train.plot_confusion_matrix")
@patch("src.train.plot_feature_importance")
def test_train_model_runs(
    mock_plot_fi,
    mock_plot_cm,
    mock_mlflow,
    data,
    config,
):
    X_train, y_train, X_val, y_val = data

    mock_mlflow.start_run.return_value.__enter__ = Mock()
    mock_mlflow.start_run.return_value.__exit__ = Mock()

    model, metrics = train_model(X_train, y_train, X_val, y_val, config)

    assert model is not None
    assert isinstance(metrics, dict)
    assert "f1_score" in metrics


@patch("src.train.mlflow")
def test_model_saved(mock_mlflow, data, config):
    X_train, y_train, X_val, y_val = data

    mock_mlflow.start_run.return_value.__enter__ = Mock()
    mock_mlflow.start_run.return_value.__exit__ = Mock()

    train_model(X_train, y_train, X_val, y_val, config)

    assert Path(config["api"]["model_path"]).exists()


@patch("src.train.mlflow")
def test_model_predicts(mock_mlflow, data, config):
    X_train, y_train, X_val, y_val = data

    mock_mlflow.start_run.return_value.__enter__ = Mock()
    mock_mlflow.start_run.return_value.__exit__ = Mock()

    model, _ = train_model(X_train, y_train, X_val, y_val, config)

    preds = model.predict(X_val.values)

    assert len(preds) == len(X_val)
    assert all(0.0 <= p <= 1.0 for p in preds)


@patch("src.train.mlflow")
def test_invalid_config_raises(mock_mlflow, data):
    X_train, y_train, X_val, y_val = data

    mock_mlflow.start_run.return_value.__enter__ = Mock()
    mock_mlflow.start_run.return_value.__exit__ = Mock()

    with pytest.raises(KeyError):
        train_model(X_train, y_train, X_val, y_val, {"model": {}})
