"""
FastAPI application for fraud detection.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from collections import deque

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import uvicorn


from api.schemas import (
    TransactionInput,
    PredictionOutput,
    BatchPredictionInput,
    BatchPredictionOutput,
    HealthResponse,
    MetricsResponse,
)

from monitoring.prometheus_metrics import (
    record_prediction,
    record_latency,
    record_error,
    update_fraud_rate,
    update_uptime
)


from api.model_loader import ModelLoader
from src.utils import load_config, setup_logging, get_logger
from monitoring.alerting import AlertManager

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------

setup_logging(log_level="INFO", log_file="logs/api.log")
logger = get_logger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Metrics (Prometheus)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# API Health Monitoring (NEW)
# ------------------------------------------------------------------

REQUEST_LATENCIES = deque(maxlen=100)
REQUEST_ERRORS = deque(maxlen=100)
alert_manager = AlertManager()

def check_api_health_alerts():
    if len(REQUEST_LATENCIES) < 10:
        return

    avg_latency = sum(REQUEST_LATENCIES) / len(REQUEST_LATENCIES)
    if len(REQUEST_ERRORS) == 0:
     return
    error_rate = sum(REQUEST_ERRORS) / len(REQUEST_ERRORS) * 100


    if avg_latency > 200:
        alert_manager.send_alert(
            alert_type="api_latency",
            severity="WARNING",
            message=f"High API latency detected (avg={avg_latency:.2f}ms)"
        )

    if error_rate > 1.0:
        alert_manager.send_alert(
            alert_type="api_error_rate",
            severity="ERROR",
            message=f"High API error rate detected ({error_rate:.2f}%)"
        )

# ------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------

model_loader: Optional[ModelLoader] = None
config = None
start_time = time.time()

PREDICTION_LOG_PATH = Path("logs/predictions.csv")

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _ensure_model_loaded():
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")


def log_prediction(amount, fraud_probability, is_fraud, risk_level, latency_ms):
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now().isoformat(),
        "amount": amount,
        "fraud_probability": fraud_probability,
        "is_fraud": is_fraud,
        "risk_level": risk_level,
        "latency_ms": latency_ms,
    }

    df = pd.DataFrame([row])
    if PREDICTION_LOG_PATH.exists():
        df.to_csv(PREDICTION_LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(PREDICTION_LOG_PATH, index=False)

# ------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    global model_loader, config
    logger.info("Starting Fraud Detection API")

    config = load_config("config/config.yaml")
    model_loader = ModelLoader(
        model_path=config["api"]["model_path"],
        scaler_path=config["api"]["scaler_path"],
        config=config,
    )
    logger.info("Model loaded successfully")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API")

# ------------------------------------------------------------------
# Middleware
# ------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000

    REQUEST_LATENCIES.append(duration_ms)
    check_api_health_alerts()

    logger.info(
        f"{request.method} {request.url.path} - "
        f"{response.status_code} - {duration_ms:.2f}ms"
    )
    return response

# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/")
async def root():
    return {"message": "Fraud Detection API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if model_loader else "unhealthy",
        model_loaded=model_loader is not None,
        uptime_seconds=time.time() - start_time,
        version="1.0.0",
    )

# ------------------ SINGLE PREDICTION ------------------

@app.post("/predict", response_model=PredictionOutput)
async def predict(transaction: TransactionInput):
    _ensure_model_loaded()
    start = time.time()

    try:
        tx = transaction.dict()
        result = model_loader.predict_transaction(tx)
        latency_ms = (time.time() - start) * 1000

        REQUEST_ERRORS.append(0)
        record_prediction(
        is_fraud=result["is_fraud"],
        probability=result["fraud_probability"],
        risk_level=result["risk_level"],
        endpoint="predict",
        )

        record_latency(latency_ms / 1000)

        
        log_prediction(
            amount=tx["Amount"],
            fraud_probability=result["fraud_probability"],
            is_fraud=result["is_fraud"],
            risk_level=result["risk_level"],
            latency_ms=latency_ms,
        )



        return PredictionOutput(
            is_fraud=result["is_fraud"],
            fraud_probability=result["fraud_probability"],
            risk_level=result["risk_level"],
            timestamp=datetime.now().isoformat(),
            response_time_ms=round(latency_ms, 2),
            model_version="1.0.0",
        )

    except Exception as e:
      REQUEST_ERRORS.append(1)
      #api_errors.inc()
      record_error()   # ← FIXED
      logger.error(f"Prediction failed: {e}", exc_info=True)
      raise HTTPException(status_code=500, detail="Prediction failed")

# ------------------ BATCH PREDICTION ------------------

@app.post("/predict_batch", response_model=BatchPredictionOutput)
async def predict_batch(batch: BatchPredictionInput):
    _ensure_model_loaded()
    start = time.time()

    try:
        transactions = [t.dict() for t in batch.transactions]
        results = model_loader.predict_batch(transactions)

        predictions = []
        fraud_count = 0

        for tx, r in zip(transactions, results):
            record_prediction(
            is_fraud=r["is_fraud"],
            probability=r["fraud_probability"],
            risk_level=r["risk_level"],
            endpoint="predict_batch",
            )


            log_prediction(
                amount=tx["Amount"],
                fraud_probability=r["fraud_probability"],
                is_fraud=r["is_fraud"],
                risk_level=r["risk_level"],
                latency_ms=0,
            )

            predictions.append(
                PredictionOutput(
                    is_fraud=r["is_fraud"],
                    fraud_probability=r["fraud_probability"],
                    risk_level=r["risk_level"],
                    timestamp=datetime.now().isoformat(),
                    response_time_ms=0,
                    model_version="1.0.0",
                )
            )

        REQUEST_ERRORS.append(0)

        return BatchPredictionOutput(
            predictions=predictions,
            total_transactions=len(transactions),
            fraud_count=fraud_count,
            total_processing_time_ms=round((time.time() - start) * 1000, 2),
        )

    except Exception as e:
        REQUEST_ERRORS.append(1)
        #api_errors.inc()
        record_error()   # ← FIXED
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Batch prediction failed")
# ------------------ METRICS ------------------

@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    if not PREDICTION_LOG_PATH.exists():
        return MetricsResponse(0, 0, 0.0, 0.0, 0.0)

    df = pd.read_csv(PREDICTION_LOG_PATH)
    if df.empty:
        return MetricsResponse(0, 0, 0.0, 0.0, 0.0)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    last_hour = df[df["timestamp"] > datetime.now() - pd.Timedelta(hours=1)]

    total = len(df)
    frauds = int(df["is_fraud"].sum())

    update_fraud_rate(frauds / total)
    update_uptime(time.time() - start_time)


    return MetricsResponse(
        total_predictions=total,
        fraud_detected=frauds,
        fraud_rate=frauds / total,
        average_latency_ms=round(df["latency_ms"].mean(), 2),
        requests_per_minute=round(len(last_hour) / 60, 2),
    )

@app.get("/prometheus-metrics")
async def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ------------------------------------------------------------------
# Local run
# ------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
