from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np

# Evidently imports for the HTML report
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.utils import load_config, get_logger
from monitoring.alerting import AlertManager

logger = get_logger(__name__)


# Helper: JSON Serializer for NumPy types
def json_converter(obj):
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    return str(obj)


# Helper: Prediction Drift
def check_prediction_drift(
    prediction_log_path: str,
    window_days: int,
    threshold: float,
) -> tuple[bool, dict]:
    df = pd.read_csv(prediction_log_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    recent = df[df["timestamp"] >= datetime.now() - pd.Timedelta(days=window_days)]

    if len(recent) < 50:
        return False, {}

    fraud_rate = recent["is_fraud"].mean()
    # Cast to standard bool for JSON safety
    drift_detected = bool(fraud_rate > threshold)

    return drift_detected, {
        "window_days": window_days,
        "fraud_rate": float(fraud_rate),
        "threshold": threshold,
        "detected": drift_detected,
    }


# Main Job
def drift_monitoring_job(
    config_path: str = "monitoring/drift_config.yaml",
    prediction_log_path: str = "logs/predictions.csv",
) -> dict:
    logger.info("=" * 80)
    logger.info("DRIFT MONITORING JOB STARTED")
    logger.info("=" * 80)

    config = load_config(config_path)
    alert = AlertManager()

    results = {
        "timestamp": datetime.now().isoformat(),
        "data_drift": None,
        "prediction_drift": None,
    }

    # 1. Data Drift & HTML Report Generation
    ref_path = Path("data/processed/train_reference.csv")
    curr_path = Path("data/processed/val.csv")

    if ref_path.exists() and curr_path.exists():
        ref = pd.read_csv(ref_path)
        curr = pd.read_csv(curr_path)

        # Basic manual check
        drift_score = abs(ref.mean().mean() - curr.mean().mean())
        threshold = config["thresholds"]["data_drift_score"]

        results["data_drift"] = {
            "drift_score": float(drift_score),
            "threshold": threshold,
            "detected": bool(drift_score > threshold),
        }

        # --- EVIDENTLY SECTION ---
        logger.info("Generating Evidently HTML Report...")
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=curr)

        report_dir = Path("reports/drift")
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"data_drift_report_{timestamp_str}.html"
        report.save_html(str(report_path))
        logger.info(f"Visual report saved to: {report_path}")
        # -------------------------

        if drift_score > threshold:
            alert.send_alert(
                alert_type="data_drift",
                severity="WARNING",
                message=f"Data drift detected (score={drift_score:.4f})",
            )
    else:
        logger.warning("Reference/current data missing — skipping data drift")

    # 2. Prediction Drift
    if Path(prediction_log_path).exists():
        detected, info = check_prediction_drift(
            prediction_log_path=prediction_log_path,
            window_days=config["drift_detection"]["current_window_days"],
            threshold=config["thresholds"]["prediction_drift_score"],
        )
        results["prediction_drift"] = info
        if detected:
            alert.send_alert(
                alert_type="prediction_drift",
                severity="WARNING",
                message="Prediction drift detected",
            )

    # Save JSON results
    Path("logs").mkdir(exist_ok=True)
    with open("logs/drift_monitoring_results.json", "w") as f:
        json.dump(results, f, indent=2, default=json_converter)

    logger.info("DRIFT MONITORING JOB COMPLETED")
    return results


if __name__ == "__main__":
    output = drift_monitoring_job()
    print(json.dumps(output, indent=2, default=json_converter))
