import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml


# General utilities
def ensure_dir(path: Path | str) -> None:
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def format_number(value: float, decimals: int = 4) -> str:
    """Format float for display."""
    return f"{value:.{decimals}f}"


# Configuration
def load_config(path: str = "config/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# Logging
def setup_logging(
    log_level: str = "INFO",
    log_file: str | None = None,
) -> None:
    """Configure root logger."""
    handlers = [logging.StreamHandler()]

    if log_file:
        ensure_dir(Path(log_file).parent)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Get named logger."""
    return logging.getLogger(name)


# Metrics
def save_metrics(metrics: dict, path: str) -> None:
    """Save metrics dictionary as JSON."""
    ensure_dir(Path(path).parent)
    metrics["timestamp"] = datetime.now().isoformat()

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(path: str) -> dict:
    """Load metrics from JSON file."""
    with open(path) as f:
        return json.load(f)


# Plotting helpers
def _save_figure(fig: plt.Figure, path: str) -> None:
    ensure_dir(Path(path).parent)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    path: str,
    labels: List[str],
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> None:
    """Save confusion matrix heatmap."""
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    _save_figure(fig, path)


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    path: str,
    title: str = "ROC Curve",
) -> None:
    """Save ROC curve plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()

    _save_figure(fig, path)


def plot_feature_importance(
    model,
    feature_names: List[str],
    path: str,
    top_n: int = 20,
    title: str = "Feature Importance",
) -> None:
    """Save feature importance bar plot."""
    # 1. Check if it's a standard Sklearn model (has the attribute)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    # 2. Check if it's a LightGBM/XGBoost model (uses a method)
    elif hasattr(model, "feature_importance"):
        importances = model.feature_importance(importance_type="gain")
    else:
        raise ValueError(
            "Model does not expose feature importance attributes or methods"
        )

    # Sort and get top N
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices], color="skyblue")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance (Gain)")
    ax.set_title(title)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Assuming _save_figure is a helper in your utils,
    # otherwise use: plt.savefig(path); plt.close()
    _save_figure(fig, path)
