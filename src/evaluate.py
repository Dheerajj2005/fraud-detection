import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import (
    get_logger,
    plot_confusion_matrix,
    plot_roc_curve,
)

logger = get_logger(__name__)


# Evaluation
def evaluate_model(
    model_path: str,
    scaler_path: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate trained model on test dataset."""

    logger.info("Loading model and scaler")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Scaling
    numerical_features = ["Amount", "Time", "amount_scaled"]
    X_test = X_test.copy()
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    # Predictions
    y_proba = model.predict(X_test)
    threshold = config["api"]["threshold"]
    y_pred = (y_proba >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Core metrics
    metrics: Dict[str, Any] = {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
    }

    # Threshold analysis
    thresholds = np.linspace(0.05, 0.95, 19)
    threshold_rows = []

    best_f1 = -1
    best_threshold = threshold

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_test, preds, zero_division=0)

        threshold_rows.append(
            {
                "threshold": float(t),
                "precision": precision_score(y_test, preds, zero_division=0),
                "recall": recall_score(y_test, preds, zero_division=0),
                "f1_score": f1,
            }
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    metrics["threshold_analysis"] = {
        "optimal_threshold": float(best_threshold),
        "optimal_f1": float(best_f1),
        "all_thresholds": threshold_rows,
    }

    # Business impact analysis
    TP_BENEFIT = 500.0  # money saved per fraud caught
    FP_COST = 5.0  # cost of false alarm
    FN_COST = 800.0  # cost of missed fraud

    total_benefit = tp * TP_BENEFIT
    total_cost = (fp * FP_COST) + (fn * FN_COST)
    net_benefit = total_benefit - total_cost

    metrics["business_metrics"] = {
        "true_positive_benefit": tp * TP_BENEFIT,
        "false_positive_cost": fp * FP_COST,
        "false_negative_cost": fn * FN_COST,
        "total_benefit": total_benefit,
        "total_cost": total_cost,
        "net_benefit": net_benefit,
        "cost_per_transaction": total_cost / len(y_test) if len(y_test) else 0.0,
    }

    # Plots
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        cm,
        str(fig_dir / "test_confusion_matrix.png"),
        labels=["Legitimate", "Fraud"],
        normalize=False,
    )

    plot_confusion_matrix(
        cm,
        str(fig_dir / "test_confusion_matrix_normalized.png"),
        labels=["Legitimate", "Fraud"],
        normalize=True,
    )

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plot_roc_curve(
        fpr,
        tpr,
        metrics["roc_auc"],
        str(fig_dir / "test_roc_curve.png"),
    )

    # Save metrics
    metrics_dir = Path("reports/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(
        f"Evaluation complete | ROC-AUC={metrics['roc_auc']:.4f} "
        f"| Best threshold={best_threshold:.2f}"
    )

    return metrics


# CLI
if __name__ == "__main__":
    from src.utils import load_config, setup_logging
    from src.data_preprocessing import prepare_features_and_target
    from src.feature_engineering import engineer_all_features

    setup_logging(log_level="INFO")
    config = load_config()

    test_df = pd.read_csv(Path(config["data"]["processed_path"]) / "test.csv")
    test_df = engineer_all_features(test_df, config)

    X_test, y_test = prepare_features_and_target(test_df, config)

    evaluate_model(
        config["api"]["model_path"],
        config["api"]["scaler_path"],
        X_test,
        y_test,
        config,
    )
