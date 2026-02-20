"""
CI-friendly tests for feature engineering module.

Focus:
- correctness of feature creation
- basic validation behavior
- fast & readable tests
"""

import pytest
import pandas as pd
import numpy as np

from src.feature_engineering import (
    create_time_features,
    create_amount_features,
    create_interaction_features,
    validate_features,
    engineer_all_features,
    get_feature_names,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def config():
    return {
        "features": {
            "numerical": ["Amount", "Time"],
            "pca_features": [f"V{i}" for i in range(1, 29)],
            "target": "Class",
            "engineered": ["hour_of_day", "amount_log", "amount_scaled"],
        }
    }


@pytest.fixture
def df():
    return pd.DataFrame({
        "Time": [0, 3600, 7200, 90000],
        "Amount": [0, 10, 100, 200],
        "V1": [0.1, -0.2, 0.3, 0.0],
        "V2": [0.0, 0.1, -0.1, 0.2],
        "Class": [0, 0, 0, 1],
    })


# ------------------------------------------------------------------
# Time features
# ------------------------------------------------------------------

def test_time_hour_range(df):
    out = create_time_features(df)
    assert out["hour_of_day"].between(0, 23).all()


def test_time_missing_col():
    with pytest.raises(ValueError):
        create_time_features(pd.DataFrame({"Amount": [10]}))


# ------------------------------------------------------------------
# Amount features
# ------------------------------------------------------------------

def test_amount_features_created(df):
    out = create_amount_features(df)
    assert "amount_log" in out.columns
    assert "amount_scaled" in out.columns


def test_amount_log_zero():
    df0 = pd.DataFrame({"Amount": [0], "Class": [0]})
    out = create_amount_features(df0)
    assert out.loc[0, "amount_log"] == 0


def test_amount_negative_handled():
    dfn = pd.DataFrame({"Amount": [-10, 5], "Class": [0, 1]})
    out = create_amount_features(dfn)
    assert (out["Amount"] >= 0).all()


# ------------------------------------------------------------------
# Interaction features
# ------------------------------------------------------------------

def test_interaction_created(df):
    out = create_time_features(df)
    out = create_amount_features(out)
    out = create_interaction_features(out)
    assert "amount_hour_interaction" in out.columns


def test_interaction_skipped_if_missing(df):
    out = create_interaction_features(df)
    assert "amount_hour_interaction" not in out.columns


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def test_validate_ok(df, config):
    out = engineer_all_features(df, config, validate=False)
    assert validate_features(out, config) is True


def test_validate_fails_nan(df, config):
    out = engineer_all_features(df, config, validate=False)
    out.loc[0, "hour_of_day"] = np.nan
    assert validate_features(out, config) is False


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

def test_engineer_all_adds_features(df, config):
    out = engineer_all_features(df, config)
    for f in config["features"]["engineered"]:
        assert f in out.columns


def test_engineer_no_row_loss(df, config):
    out = engineer_all_features(df, config)
    assert len(out) == len(df)


def test_engineer_validation_raises(config):
    bad = pd.DataFrame({"Time": [np.nan], "Amount": [10], "Class": [0]})
    with pytest.raises(ValueError):
        engineer_all_features(bad, config, validate=True)


# ------------------------------------------------------------------
# Feature names
# ------------------------------------------------------------------

def test_feature_names_all(config):
    names = get_feature_names(config)
    assert "Amount" in names
    assert "V1" in names
    assert "hour_of_day" in names


def test_feature_names_no_engineered(config):
    names = get_feature_names(config, include_engineered=False)
    assert "hour_of_day" not in names
