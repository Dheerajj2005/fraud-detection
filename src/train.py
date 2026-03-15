import time
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from src.utils import (
    get_logger,
    plot_confusion_matrix,
    plot_feature_importance,
)
from src.data_preprocessing import (
    load_data,
    split_data,
    scale_features,
    handle_imbalance,
    prepare_features_and_target,
    save_processed_data,
)
from src.feature_engineering import engineer_all_features

logger = get_logger(__name__)


# Metrics
def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
    }


def _log_metrics(metrics: Dict[str, float]) -> None:
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
        logger.info(f"{k}: {v:.4f}")


# Training
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: Dict[str, Any],
) -> Tuple[lgb.Booster, Dict[str, float]]:
    """Train LightGBM model and log to MLflow."""
    logger.info("Starting model training")
    start_time = time.time()

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params(config["model"]["params"])
        mlflow.log_params(
            {
                "n_estimators": config["training"]["n_estimators"],
                "use_smote": config["training"]["use_smote"],
                "smote_ratio": config["training"]["smote_ratio"],
                "train_size": len(X_train),
                "val_size": len(X_val),
            }
        )

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            config["model"]["params"],
            train_data,
            num_boost_round=config["training"]["n_estimators"],
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(config["training"]["early_stopping_rounds"]),
                lgb.log_evaluation(period=100),
            ],
        )

        mlflow.log_metric("training_time_seconds", time.time() - start_time)

        y_proba = model.predict(X_val)
        y_pred = (y_proba >= config["api"]["threshold"]).astype(int)

        metrics = calculate_metrics(y_val, y_pred, y_proba)
        _log_metrics(metrics)

        # Artifacts
        fig_dir = Path("reports/figures")
        fig_dir.mkdir(parents=True, exist_ok=True)

        cm_path = fig_dir / "confusion_matrix.png"
        plot_confusion_matrix(
            confusion_matrix(y_val, y_pred),
            str(cm_path),
            labels=["Legitimate", "Fraud"],
        )
        mlflow.log_artifact(str(cm_path))

        fi_path = fig_dir / "feature_importance.png"
        plot_feature_importance(model, X_train.columns.tolist(), str(fi_path))
        mlflow.log_artifact(str(fi_path))

        report = classification_report(
            y_val, y_pred, target_names=["Legitimate", "Fraud"]
        )
        report_path = Path("reports/classification_report.txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
        mlflow.log_artifact(str(report_path))

        model_path = Path(config["api"]["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        mlflow.lightgbm.log_model(
            model,
            "model",
            registered_model_name=config["mlflow"]["model_name"],
        )

        logger.info("Model training completed")
        return model, metrics


# Pipeline
def main():
    from src.utils import load_config, setup_logging

    setup_logging(log_level="INFO", log_file="logs/training.log")

    logger.info("Starting training pipeline")

    config = load_config("config/config.yaml")

    df = load_data(config)
    df = engineer_all_features(df, config)

    train_df, val_df, test_df = split_data(df, config)
    save_processed_data(train_df, val_df, test_df, config)

    X_train, y_train = prepare_features_and_target(train_df, config)
    X_val, y_val = prepare_features_and_target(val_df, config)
    X_test, y_test = prepare_features_and_target(test_df, config)

    X_train, X_val, X_test, _ = scale_features(
        X_train,
        X_val,
        X_test,
        numerical_features=["Amount", "Time", "amount_scaled"],
        scaler_path=config["api"]["scaler_path"],
    )

    X_train, y_train = handle_imbalance(X_train, y_train, config)

    model, metrics = train_model(X_train, y_train, X_val, y_val, config)

    logger.info("Training complete")
    return model, metrics


if __name__ == "__main__":
    main()
