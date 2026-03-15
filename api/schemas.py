from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, create_model


# Helpers (very limited, internal)
def _pca_fields():
    return {
        f"V{i}": (
            Optional[float],
            Field(
                default=0.0, description=f"PCA component V{i} (optional, default=0.0)"
            ),
        )
        for i in range(1, 29)
    }


# Input Schemas
TransactionInput = create_model(
    "TransactionInput",
    Time=(float, Field(..., ge=0, description="Seconds since first transaction")),
    Amount=(float, Field(..., ge=0, description="Transaction amount")),
    **_pca_fields(),
)

TransactionInput.__config__ = type(
    "Config",
    (),
    {
        "json_schema_extra": {
            "example": {
                "Time": 12345.0,
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536347,
                "V4": 1.378155,
                "V28": -0.021053,
                "Amount": 149.62,
            }
        }
    },
)


class BatchPredictionInput(BaseModel):
    """Schema for batch prediction input."""

    transactions: List["TransactionInput"]

    class Config:
        json_schema_extra = {
            "example": {
                "transactions": [
                    {
                        "Time": 12345.0,
                        "V1": -1.359807,
                        "Amount": 149.62,
                    }
                ]
            }
        }


# Output Schemas
class PredictionOutput(BaseModel):
    """Schema for prediction output."""

    is_fraud: bool
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    timestamp: str
    response_time_ms: float
    model_version: str

    class Config:
        json_schema_extra = {
            "example": {
                "is_fraud": False,
                "fraud_probability": 0.0234,
                "risk_level": "LOW",
                "timestamp": "2026-01-03T10:30:45.123456",
                "response_time_ms": 45.6,
                "model_version": "1.0.0",
            }
        }


class BatchPredictionOutput(BaseModel):
    """Schema for batch prediction output."""

    predictions: List[PredictionOutput]
    total_transactions: int
    fraud_count: int
    total_processing_time_ms: float

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [],
                "total_transactions": 100,
                "fraud_count": 3,
                "total_processing_time_ms": 1234.56,
            }
        }


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str
    model_loaded: bool
    uptime_seconds: float
    version: str

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "uptime_seconds": 3600.0,
                "version": "1.0.0",
            }
        }


class MetricsResponse(BaseModel):
    """Schema for metrics endpoint response."""

    total_predictions: int
    fraud_detected: int
    fraud_rate: float
    average_latency_ms: float
    requests_per_minute: float

    class Config:
        json_schema_extra = {
            "example": {
                "total_predictions": 10000,
                "fraud_detected": 150,
                "fraud_rate": 0.015,
                "average_latency_ms": 47.3,
                "requests_per_minute": 45.2,
            }
        }
