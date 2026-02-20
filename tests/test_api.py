"""
CI-friendly tests for FastAPI fraud detection API.

Focus:
- request validation
- correct responses
- minimal mocking
- fast & deterministic tests
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import pandas as pd


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def mock_model():
    model = Mock()

    def predict(txn):
        fraud = txn["Amount"] > 500
        return {
            "is_fraud": fraud,
            "fraud_probability": 0.8 if fraud else 0.1,
            "risk_level": "HIGH" if fraud else "LOW",
        }

    model.predict_transaction = predict
    model.predict_batch = lambda txns: [predict(t) for t in txns]
    return model


@pytest.fixture
def client(tmp_path, mock_model):
    config = {
        "api": {
            "model_path": "dummy.pkl",
            "scaler_path": "dummy_scaler.pkl",
            "threshold": 0.5,
        },
        "project": {"version": "1.0.0"},
    }

    log_path = tmp_path / "predictions.csv"
    pd.DataFrame().to_csv(log_path, index=False)

    with patch("api.main.load_config", return_value=config):
        with patch("api.main.ModelLoader", return_value=mock_model):
            with patch("api.main.prediction_log_path", str(log_path)):
                from api.main import app
                yield TestClient(app)


@pytest.fixture
def txn():
    return {
        "Time": 12345,
        "Amount": 100,
        **{f"V{i}": 0.0 for i in range(1, 29)},
    }


# ------------------------------------------------------------------
# Root & health
# ------------------------------------------------------------------

def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Fraud Detection" in r.json()["message"]


def test_health(client):
    r = client.get("/health")
    data = r.json()
    assert r.status_code == 200
    assert data["model_loaded"] is True


# ------------------------------------------------------------------
# Single prediction
# ------------------------------------------------------------------

def test_predict_ok(client, txn):
    r = client.post("/predict", json=txn)
    data = r.json()

    assert r.status_code == 200
    assert "is_fraud" in data
    assert 0 <= data["fraud_probability"] <= 1


def test_predict_fraud_logic(client, txn):
    txn["Amount"] = 1000
    r = client.post("/predict", json=txn)
    assert r.json()["is_fraud"] is True


def test_predict_validation_error(client):
    r = client.post("/predict", json={"Amount": 100})
    assert r.status_code == 422


# ------------------------------------------------------------------
# Batch prediction
# ------------------------------------------------------------------

def test_batch_predict(client, txn):
    batch = {"transactions": [txn, txn]}
    r = client.post("/predict_batch", json=batch)
    data = r.json()

    assert r.status_code == 200
    assert data["total_transactions"] == 2
    assert len(data["predictions"]) == 2


def test_batch_empty(client):
    r = client.post("/predict_batch", json={"transactions": []})
    assert r.json()["total_transactions"] == 0


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def test_metrics(client):
    r = client.get("/metrics")
    data = r.json()

    assert r.status_code == 200
    assert "total_predictions" in data
    assert 0 <= data["fraud_rate"] <= 1


# ------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------

def test_404(client):
    r = client.get("/does-not-exist")
    assert r.status_code == 404
