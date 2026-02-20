"""
Automated Training Pipeline with Prefect
File: pipelines/training_pipeline.py
"""

import sys
from pathlib import Path
from typing import Dict
from datetime import datetime
import json

import pandas as pd
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_config, setup_logging, get_logger
from src.data_preprocessing import (
    load_data, split_data, scale_features, handle_imbalance
)
from src.feature_engineering import engineer_all_features
from src.train import train_model
from src.evaluate import evaluate_model
from monitoring.drift_detection import generate_data_drift_report
from monitoring.alerting import AlertManager

setup_logging()
logger = get_logger(__name__)

# ------------------- Tasks -------------------

@task
def preprocess_data(config: Dict):
    df = load_data(config)
    df = engineer_all_features(df, config)
    train_df, val_df, test_df = split_data(df, config)

    target = config["features"]["target"]
    features = [c for c in df.columns if c != target]

    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    X_test, y_test = test_df[features], test_df[target]

    X_train, X_val, X_test, scaler = scale_features(
        X_train, X_val, X_test, config["features"]["numerical"]
    )

    if config["training"].get("use_smote", False):
        X_train, y_train = handle_imbalance(X_train, y_train, config)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


@task
def check_drift(X_train, y_train, pipeline_cfg):
    ref_path = pipeline_cfg["reference_data"]["path"]
    if not Path(ref_path).exists():
        return False, 0.0

    ref = pd.read_csv(ref_path)
    current = pd.concat([X_train, y_train], axis=1)

    report = generate_data_drift_report(ref, current)
    return (
        report["dataset_drift_score"]
        > pipeline_cfg["triggers"]["data_drift_threshold"],
        report["dataset_drift_score"],
    )


@task
def train_task(X_train, y_train, X_val, y_val, config):
    return train_model(X_train, y_train, X_val, y_val, config)


@task
def evaluate_task(model, scaler, X_test, y_test, config):
    return evaluate_model(
        config["api"]["model_path"],
        config["api"]["scaler_path"],
        X_test,
        y_test,
        config,
    )


# ------------------- Flow -------------------

@flow(
    name="fraud_detection_training_pipeline",
    task_runner=SequentialTaskRunner(),
)
def training_pipeline(force_retrain: bool = False, skip_drift_check: bool = False):

    config = load_config("config/config.yaml")
    pipeline_cfg = load_config("pipelines/pipeline_config.yaml")
    alert = AlertManager(config)

    alert.send_alert("pipeline_start", "Training pipeline started", "INFO")

    start_time = datetime.now()
    results = {"status": "FAILED"}

    try:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess_data(config)

        drift_detected = False
        drift_score = 0.0

        if not skip_drift_check:
            drift_detected, drift_score = check_drift(X_train, y_train, pipeline_cfg)
            if drift_detected:
                alert.send_alert(
                    "data_drift",
                    f"Data drift detected (score={drift_score:.4f})",
                    "WARNING",
                )

        if not (force_retrain or skip_drift_check or drift_detected):
            alert.send_alert(
                "training_skipped",
                "No drift detected, training skipped",
                "INFO",
            )
            return {"status": "SKIPPED"}

        model, train_metrics = train_task(X_train, y_train, X_val, y_val, config)
        test_metrics = evaluate_task(model, scaler, X_test, y_test, config)

        if (
            test_metrics["f1_score"]
            >= pipeline_cfg["triggers"]["min_f1_score"]
            and test_metrics["recall"]
            >= pipeline_cfg["triggers"]["min_recall"]
        ):
            alert.send_alert(
                "model_registered",
                f"Model registered (F1={test_metrics['f1_score']:.4f})",
                "INFO",
            )
            registered = True
        else:
            alert.send_alert(
                "model_not_registered",
                "Model metrics below threshold",
                "WARNING",
            )
            registered = False

        results.update(
            {
                "status": "SUCCESS",
                "drift_detected": drift_detected,
                "drift_score": drift_score,
                "metrics": test_metrics,
                "model_registered": registered,
            }
        )

    except Exception as e:
        alert.send_alert("pipeline_failed", str(e), "CRITICAL", force=True)
        raise

    finally:
        results["duration_seconds"] = (
            datetime.now() - start_time
        ).total_seconds()

        Path("logs").mkdir(exist_ok=True)
        with open("logs/pipeline_results.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


# ------------------- Deployment -------------------

def create_deployment():
    Deployment.build_from_flow(
        flow=training_pipeline,
        name="weekly-training",
        schedule=CronSchedule(cron="0 0 * * 0"),
        work_queue_name="fraud-detection",
    ).apply()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--skip-drift-check", action="store_true")
    parser.add_argument("--create-deployment", action="store_true")
    args = parser.parse_args()

    if args.create_deployment:
        create_deployment()
    else:
        print(
            json.dumps(
                training_pipeline(
                    force_retrain=args.force_retrain,
                    skip_drift_check=args.skip_drift_check,
                ),
                indent=2,
            )
        )
