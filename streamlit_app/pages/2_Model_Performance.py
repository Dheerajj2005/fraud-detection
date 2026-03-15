"""
Model Performance page showing training and evaluation metrics.
Refactored for hybrid-safety and redundancy removal.
"""

import streamlit as st
import json
from pathlib import Path
import pandas as pd

# --- CONFIG & PATHS ---
st.set_page_config(page_title="Model Performance", page_icon="📊", layout="wide")

st.title("📊 Model Performance")
st.markdown("Comprehensive model evaluation metrics and visualizations")

metrics_path = Path("reports/metrics/test_metrics.json")
figures_path = Path("reports/figures")


# --- HELPER FUNCTIONS ---
def compute_fpr(metrics):
    fp = metrics.get("false_positives", 0)
    tn = metrics.get("true_negatives", 0)
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def get_accuracy(metrics):
    acc = metrics.get("accuracy")
    if acc is None:
        try:
            tp, tn = metrics["true_positives"], metrics["true_negatives"]
            fp, fn = metrics["false_positives"], metrics["false_negatives"]
            return (tp + tn) / (tp + tn + fp + fn)
        except KeyError:
            return 0.0
    return acc


# --- DATA LOADING ---
if not metrics_path.exists():
    st.warning("⚠️ No evaluation metrics found. Please run model evaluation first.")
    st.code("python src/evaluate.py", language="bash")
    st.stop()

with open(metrics_path, "r") as f:
    metrics = json.load(f)

# Derive missing metrics safely
accuracy = get_accuracy(metrics)
fpr = (
    metrics.get("false_positive_rate")
    if "false_positive_rate" in metrics
    else compute_fpr(metrics)
)

# --- UI SECTIONS ---

## 🎯 Performance Overview
st.markdown("## 🎯 Performance Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", f"{accuracy:.2%}", help="Overall classification accuracy")
with col2:
    st.metric(
        "Recall (Fraud)",
        f"{metrics.get('recall', 0):.2%}",
        help="Percentage of frauds detected",
    )
with col3:
    st.metric(
        "Precision (Fraud)",
        f"{metrics.get('precision', 0):.2%}",
        help="Percentage of correct fraud predictions",
    )
with col4:
    st.metric(
        "F1-Score",
        f"{metrics.get('f1_score', 0):.2%}",
        help="Harmonic mean of precision and recall",
    )

st.markdown("---")

## 📈 ROC and PR AUC
col1, col2 = st.columns(2)
with col1:
    st.metric("ROC-AUC Score", f"{metrics.get('roc_auc', 0):.4f}")
with col2:
    st.metric("PR-AUC Score", f"{metrics.get('pr_auc', 0):.4f}")

st.markdown("---")

## 📊 Confusion Matrix
st.markdown("## 📊 Confusion Matrix")
col_img1, col_img2 = st.columns(2)


def display_image(path, caption, col):
    if path.exists():
        col.image(str(path), caption=caption)
    else:
        col.warning(f"Image not found: {path.name}")


display_image(
    figures_path / "test_confusion_matrix.png", "Confusion Matrix (Counts)", col_img1
)
display_image(
    figures_path / "test_confusion_matrix_normalized.png",
    "Confusion Matrix (Normalized)",
    col_img2,
)

# Raw Counts
c1, c2, c3, c4 = st.columns(4)
c1.metric("True Positives", f"{metrics.get('true_positives', 0):,}")
c2.metric("True Negatives", f"{metrics.get('true_negatives', 0):,}")
c3.metric("False Positives", f"{metrics.get('false_positives', 0):,}")
c4.metric("False Negatives", f"{metrics.get('false_negatives', 0):,}")

st.markdown("---")

## 🎚️ Threshold Analysis (Optional Rendering)
st.markdown("## 🎚️ Threshold Analysis")
if "threshold_analysis" in metrics:
    ta = metrics["threshold_analysis"]
    col1, col2 = st.columns(2)
    col1.metric("Optimal Threshold", f"{ta['optimal_threshold']:.3f}")
    col2.metric("F1 at Optimal", f"{ta['optimal_f1']:.4f}")

    with st.expander("📋 Detailed Threshold Metrics"):
        st.dataframe(
            pd.DataFrame(ta["all_thresholds"]).style.format("{:.4f}"),
            use_container_width=True,
        )
else:
    st.info("ℹ️ Threshold analysis data not available in metrics file.")

st.markdown("---")

## 💰 Business Impact (Optional Rendering)
st.markdown("## 💰 Business Impact")
if "business_metrics" in metrics:
    bm = metrics["business_metrics"]
    b1, b2, b3 = st.columns(3)
    b1.metric("Total Benefit", f"${bm['total_benefit']:,.2f}")
    b2.metric("Total Cost", f"${bm['total_cost']:,.2f}")
    b3.metric(
        "Net Benefit", f"${bm['net_benefit']:,.2f}", delta=f"${bm['net_benefit']:,.2f}"
    )

    cost_data = pd.DataFrame(
        {
            "Category": ["TP Benefit", "FP Cost", "FN Cost"],
            "Amount": [
                bm["true_positive_benefit"],
                -bm["false_positive_cost"],
                -bm["false_negative_cost"],
            ],
        }
    )
    st.bar_chart(cost_data.set_index("Category"))
else:
    st.info("ℹ️ Business impact metrics not computed for this run.")

# --- SIDEBAR SUMMARY ---
with st.sidebar:
    st.markdown("### 📊 Performance Summary")

    # Status Indicators
    if metrics.get("recall", 0) >= 0.96:
        st.success(f"✅ Recall: {metrics['recall']:.2%}")
    else:
        st.warning(f"⚠️ Recall: {metrics.get('recall', 0):.2%}")

    if fpr <= 0.025:
        st.success(f"✅ FPR: {fpr:.2%}")
    else:
        st.warning(f"⚠️ FPR: {fpr:.2%}")

    st.markdown("---")
    st.markdown(
        f"""
    - **Detection Rate:** {metrics.get('recall', 0):.1%}
    - **Precision:** {metrics.get('precision', 0):.1%}
    - **False Alarms:** {fpr:.2%}
    """
    )
