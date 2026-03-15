"""
Minimal Prometheus Metrics
File: monitoring/prometheus_metrics.py

Phase 2 scope:
- API traffic
- Latency
- Errors
- Fraud signal
"""

from prometheus_client import Counter, Histogram, Gauge

# Core API Metrics

# Total prediction requests
predictions_total = Counter(
    "fraud_predictions_total",
    "Total number of prediction requests",
    ["endpoint", "risk_level"],
)

# Fraud predictions count
fraud_detected_total = Counter(
    "fraud_detected_total", "Total number of transactions flagged as fraud"
)

# API errors
api_errors_total = Counter("api_errors_total", "Total API errors")

# Prediction latency (seconds)
prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

# Business / Signal Metrics

# Current fraud rate (rolling)
fraud_rate = Gauge("fraud_rate", "Current fraud rate (fraud / total predictions)")

# System Health

# API uptime
api_uptime_seconds = Gauge("api_uptime_seconds", "API uptime in seconds")

# Helper update functions


def record_prediction(
    is_fraud: bool,
    probability: float = None,
    risk_level: str = "unknown",
    endpoint: str = "default",
):
    # Using labels allows you to filter metrics in Prometheus/Grafana
    predictions_total.labels(endpoint=endpoint, risk_level=risk_level).inc()

    if is_fraud:
        fraud_detected_total.inc()


def record_latency(latency_seconds: float):
    prediction_latency_seconds.observe(latency_seconds)


def record_error():
    api_errors_total.inc()


def update_fraud_rate(rate: float):
    fraud_rate.set(rate)


def update_uptime(uptime_seconds: float):
    api_uptime_seconds.set(uptime_seconds)
